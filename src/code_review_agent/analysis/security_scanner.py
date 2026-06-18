"""Security scanner integration with Bandit and Semgrep.

Runs security-focused static analysis tools in Docker sandboxes
to detect vulnerabilities, hardcoded secrets, injection flaws, and
other security anti-patterns in pull request code.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from code_review_agent.sandbox.docker_sandbox import DockerSandbox, SandboxResult
from code_review_agent.logging import get_logger

logger = get_logger("security_scanner")


@dataclass
class SecurityFinding:
    """A security issue identified by a scanner.

    Attributes:
        file_path: File where the vulnerability exists.
        line_number: Line number of the finding.
        severity: critical | high | medium | low.
        confidence: How confident the tool is (high | medium | low).
        title: Short title/name of the vulnerability.
        description: Detailed explanation of the issue.
        cwe_id: Common Weakness Enumeration ID, if available.
        tool: Which scanner found this (bandit, semgrep, secrets).
        rule_id: Scanner-specific rule identifier.
        code_snippet: The offending code snippet.
        remediation: Suggested fix or mitigation.
    """

    file_path: str
    line_number: int
    severity: str
    confidence: str = "medium"
    title: str = ""
    description: str = ""
    cwe_id: str | None = None
    tool: str = ""
    rule_id: str = ""
    code_snippet: str = ""
    remediation: str = ""

    @property
    def is_critical(self) -> bool:
        """Check if this is a critical finding."""
        return self.severity == "critical" or (
            self.severity == "high" and self.confidence == "high"
        )


@dataclass
class SecurityScanResult:
    """Aggregated results from all security scanners.

    Attributes:
        findings: All security findings across all tools.
        tools_run: Which scanners were executed.
        errors: Errors from scanners that failed.
        scan_duration: Total scan duration in seconds.
    """

    findings: list[SecurityFinding] = field(default_factory=list)
    tools_run: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    scan_duration: float = 0.0

    @property
    def critical_findings(self) -> list[SecurityFinding]:
        """Get only critical-severity findings."""
        return [f for f in self.findings if f.severity == "critical"]

    @property
    def high_findings(self) -> list[SecurityFinding]:
        """Get only high-severity findings."""
        return [f for f in self.findings if f.severity == "high"]

    @property
    def has_blocking_issues(self) -> bool:
        """Check if there are issues that should block the PR."""
        return bool(self.critical_findings) or len(self.high_findings) >= 2

    def findings_for_file(self, file_path: str) -> list[SecurityFinding]:
        """Get findings for a specific file."""
        return [f for f in self.findings if f.file_path == file_path]


# Common secret patterns for regex-based detection
SECRET_PATTERNS: list[tuple[str, str, str]] = [
    (r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"][A-Za-z0-9_\-]{16,}['\"]", "Possible API key", "SEC-SECRETS-001"),
    (r"(?i)(secret|password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{8,}['\"]", "Hardcoded secret/password", "SEC-SECRETS-002"),
    (r"(?i)aws[_-]?(secret|access)[_-]?key\s*[=:]\s*['\"][A-Za-z0-9/+=]{20,}['\"]", "AWS credential", "SEC-SECRETS-003"),
    (r"ghp_[A-Za-z0-9]{36}", "GitHub personal access token", "SEC-SECRETS-004"),
    (r"sk-[A-Za-z0-9]{32,}", "OpenAI API key pattern", "SEC-SECRETS-005"),
    (r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*", "Bearer token in code", "SEC-SECRETS-006"),
    (r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----", "Private key in source", "SEC-SECRETS-007"),
]


class SecurityScanner:
    """Orchestrates security scanning using multiple tools.

    Runs Bandit (Python), Semgrep (multi-language), and regex-based
    secret detection against PR source files.

    Attributes:
        sandbox: Docker sandbox for tool execution.
        enabled_tools: Which security tools to run.
    """

    def __init__(
        self,
        sandbox: DockerSandbox,
        enabled_tools: list[str] | None = None,
    ) -> None:
        """Initialize the security scanner.

        Args:
            sandbox: Configured Docker sandbox.
            enabled_tools: Tools to run. Defaults to ["bandit", "semgrep", "secrets"].
        """
        self.sandbox = sandbox
        self.enabled_tools = enabled_tools or ["bandit", "semgrep", "secrets"]

    async def scan(
        self,
        source_files: dict[str, str],
        language: str | None = None,
    ) -> SecurityScanResult:
        """Run all enabled security scanners against source files.

        Args:
            source_files: Mapping of file paths to file contents.
            language: Primary language (used to select relevant tools).

        Returns:
            Aggregated SecurityScanResult.
        """
        import time

        start = time.time()
        result = SecurityScanResult()

        # Always run secret detection (no Docker needed)
        if "secrets" in self.enabled_tools:
            secrets = self._scan_secrets(source_files)
            result.findings.extend(secrets)
            result.tools_run.append("secrets")

        # Run Bandit for Python files
        if "bandit" in self.enabled_tools and self._has_python_files(source_files):
            bandit_findings = await self._run_bandit(source_files)
            result.findings.extend(bandit_findings)
            result.tools_run.append("bandit")

        # Run Semgrep for multi-language scanning
        if "semgrep" in self.enabled_tools:
            semgrep_findings = await self._run_semgrep(source_files, language)
            result.findings.extend(semgrep_findings)
            result.tools_run.append("semgrep")

        result.scan_duration = round(time.time() - start, 2)

        logger.info(
            "security scan complete",
            tools=result.tools_run,
            findings=len(result.findings),
            critical=len(result.critical_findings),
            high=len(result.high_findings),
            duration_s=result.scan_duration,
        )

        return result

    def _scan_secrets(self, source_files: dict[str, str]) -> list[SecurityFinding]:
        """Scan for hardcoded secrets using regex patterns.

        Runs entirely in-process (no Docker needed).

        Args:
            source_files: File contents to scan.

        Returns:
            List of secret-related findings.
        """
        findings: list[SecurityFinding] = []

        for file_path, content in source_files.items():
            # Skip likely non-source files
            if any(file_path.endswith(ext) for ext in (".md", ".txt", ".json", ".yml", ".yaml")):
                if not file_path.endswith(".env.example"):
                    continue

            for line_num, line in enumerate(content.split("\n"), start=1):
                for pattern, title, rule_id in SECRET_PATTERNS:
                    if re.search(pattern, line):
                        # Skip if it looks like a placeholder or example
                        if self._is_likely_placeholder(line):
                            continue

                        findings.append(
                            SecurityFinding(
                                file_path=file_path,
                                line_number=line_num,
                                severity="critical",
                                confidence="medium",
                                title=title,
                                description=f"Potential secret detected: {title}. Secrets should never be committed to source control.",
                                tool="secrets",
                                rule_id=rule_id,
                                code_snippet=line.strip()[:100],
                                remediation="Use environment variables or a secret management service. Add this file to .gitignore if it contains real credentials.",
                            )
                        )

        return findings

    @staticmethod
    def _is_likely_placeholder(line: str) -> bool:
        """Check if a matched line is likely a placeholder/example.

        Args:
            line: The source line to check.

        Returns:
            True if it looks like a placeholder value.
        """
        placeholder_indicators = [
            "xxx", "placeholder", "example", "your_", "changeme",
            "replace_me", "<your", "TODO", "FIXME", "test",
            "dummy", "sample", "fake",
        ]
        line_lower = line.lower()
        return any(indicator in line_lower for indicator in placeholder_indicators)

    async def _run_bandit(self, source_files: dict[str, str]) -> list[SecurityFinding]:
        """Run Bandit security scanner for Python code.

        Args:
            source_files: Source files to scan.

        Returns:
            Parsed Bandit findings.
        """
        python_files = {
            k: v for k, v in source_files.items() if k.endswith(".py")
        }
        if not python_files:
            return []

        result = self.sandbox.run_with_source(
            command="pip install bandit -q && bandit -r . -f json -ll 2>/dev/null || true",
            source_files=python_files,
            image="python:3.11-slim",
        )

        if not result.stdout:
            return []

        return self._parse_bandit_output(result.stdout)

    def _parse_bandit_output(self, output: str) -> list[SecurityFinding]:
        """Parse Bandit JSON output into SecurityFinding objects.

        Args:
            output: Raw Bandit JSON output.

        Returns:
            List of parsed security findings.
        """
        findings: list[SecurityFinding] = []

        try:
            # Find JSON in output
            json_start = output.find("{")
            if json_start == -1:
                return findings

            data = json.loads(output[json_start:])
            results = data.get("results", [])

            for item in results:
                # Map Bandit severity levels
                severity_map = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}
                severity = severity_map.get(item.get("issue_severity", ""), "medium")

                # Promote to critical if high severity + high confidence
                confidence = item.get("issue_confidence", "MEDIUM").lower()
                if severity == "high" and confidence == "high":
                    severity = "critical"

                file_path = item.get("filename", "")
                if file_path.startswith("./"):
                    file_path = file_path[2:]

                findings.append(
                    SecurityFinding(
                        file_path=file_path,
                        line_number=item.get("line_number", 0),
                        severity=severity,
                        confidence=confidence,
                        title=item.get("issue_text", ""),
                        description=item.get("issue_text", ""),
                        cwe_id=item.get("issue_cwe", {}).get("id"),
                        tool="bandit",
                        rule_id=item.get("test_id", ""),
                        code_snippet=item.get("code", "").strip()[:200],
                        remediation=item.get("more_info", ""),
                    )
                )

        except json.JSONDecodeError:
            logger.debug("failed to parse bandit output")

        return findings

    async def _run_semgrep(
        self, source_files: dict[str, str], language: str | None
    ) -> list[SecurityFinding]:
        """Run Semgrep with security-focused rulesets.

        Args:
            source_files: Source files to scan.
            language: Primary language for rule selection.

        Returns:
            Parsed Semgrep findings.
        """
        # Select appropriate rulesets
        rulesets = ["p/security-audit", "p/secrets"]
        if language == "python":
            rulesets.append("p/python")
        elif language in ("javascript", "typescript"):
            rulesets.append("p/javascript")

        rules_arg = " ".join(f"--config {r}" for r in rulesets)

        result = self.sandbox.run_with_source(
            command=f"pip install semgrep -q && semgrep {rules_arg} --json . 2>/dev/null || true",
            source_files=source_files,
            image="python:3.11-slim",
        )

        if not result.stdout:
            return []

        return self._parse_semgrep_output(result.stdout)

    def _parse_semgrep_output(self, output: str) -> list[SecurityFinding]:
        """Parse Semgrep JSON output into SecurityFinding objects.

        Args:
            output: Raw Semgrep JSON output.

        Returns:
            List of parsed security findings.
        """
        findings: list[SecurityFinding] = []

        try:
            json_start = output.find("{")
            if json_start == -1:
                return findings

            data = json.loads(output[json_start:])
            results = data.get("results", [])

            for item in results:
                extra = item.get("extra", {})
                metadata = extra.get("metadata", {})

                # Map semgrep severity
                semgrep_severity = extra.get("severity", "WARNING")
                severity_map = {"ERROR": "high", "WARNING": "medium", "INFO": "low"}
                severity = severity_map.get(semgrep_severity, "medium")

                # Check metadata for higher severity indicators
                if metadata.get("impact") == "HIGH" or "critical" in str(metadata.get("cwe", [])).lower():
                    severity = "critical"

                file_path = item.get("path", "")
                if file_path.startswith("./"):
                    file_path = file_path[2:]

                cwe_list = metadata.get("cwe", [])
                cwe_id = cwe_list[0] if cwe_list else None

                findings.append(
                    SecurityFinding(
                        file_path=file_path,
                        line_number=item.get("start", {}).get("line", 0),
                        severity=severity,
                        confidence=metadata.get("confidence", "medium").lower(),
                        title=extra.get("message", "")[:200],
                        description=extra.get("message", ""),
                        cwe_id=cwe_id,
                        tool="semgrep",
                        rule_id=item.get("check_id", ""),
                        code_snippet=extra.get("lines", "").strip()[:200],
                        remediation=metadata.get("fix", "") or metadata.get("references", [""])[0] if metadata.get("references") else "",
                    )
                )

        except (json.JSONDecodeError, KeyError, IndexError):
            logger.debug("failed to parse semgrep output")

        return findings

    @staticmethod
    def _has_python_files(source_files: dict[str, str]) -> bool:
        """Check if any source files are Python."""
        return any(f.endswith(".py") for f in source_files)
