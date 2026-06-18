"""Static analysis and test runner using Docker sandboxes.

Coordinates running linters, type checkers, and test suites in
isolated containers, then parses their output into structured results.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Any

from code_review_agent.sandbox.docker_sandbox import DockerSandbox, SandboxResult
from code_review_agent.logging import get_logger

logger = get_logger("analysis_runner")


@dataclass
class LintIssue:
    """A single issue reported by a linter or static analysis tool.

    Attributes:
        file_path: File where the issue was found.
        line: Line number.
        column: Column number.
        severity: error, warning, info, or convention.
        message: Description of the issue.
        rule_id: Linter rule identifier.
        tool: Which tool reported this issue.
    """

    file_path: str
    line: int
    column: int = 0
    severity: str = "warning"
    message: str = ""
    rule_id: str = ""
    tool: str = ""


@dataclass
class AnalysisResult:
    """Aggregated results from all static analysis tools.

    Attributes:
        issues: All lint/analysis issues found.
        test_passed: Whether tests passed (None if not run).
        test_output: Raw test runner output.
        tools_run: Which tools were executed.
        errors: Any errors from tools that failed to run.
    """

    issues: list[LintIssue] = field(default_factory=list)
    test_passed: bool | None = None
    test_output: str = ""
    tools_run: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Number of error-level issues."""
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        """Number of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == "warning")

    def issues_for_file(self, file_path: str) -> list[LintIssue]:
        """Get issues for a specific file."""
        return [i for i in self.issues if i.file_path == file_path]


class AnalysisRunner:
    """Runs static analysis and tests in Docker sandboxes.

    Detects the project's language and tooling, then executes
    appropriate linters and test commands in isolated containers.

    Attributes:
        sandbox: Docker sandbox for command execution.
    """

    def __init__(self, sandbox: DockerSandbox) -> None:
        """Initialize the analysis runner.

        Args:
            sandbox: Configured Docker sandbox.
        """
        self.sandbox = sandbox

    async def run_analysis(
        self,
        source_files: dict[str, str],
        language: str,
        run_tests: bool = True,
    ) -> AnalysisResult:
        """Run full static analysis pipeline for a set of source files.

        Args:
            source_files: Mapping of file paths to file contents.
            language: Primary programming language.
            run_tests: Whether to attempt running tests.

        Returns:
            Aggregated AnalysisResult.
        """
        result = AnalysisResult()

        if language == "python":
            await self._run_python_analysis(source_files, result, run_tests)
        elif language in ("javascript", "typescript"):
            await self._run_js_ts_analysis(source_files, result, run_tests)
        else:
            logger.debug("no analysis configured for language", language=language)

        logger.info(
            "analysis complete",
            language=language,
            tools_run=result.tools_run,
            issues=len(result.issues),
            errors=result.error_count,
            warnings=result.warning_count,
        )

        return result

    async def _run_python_analysis(
        self,
        source_files: dict[str, str],
        result: AnalysisResult,
        run_tests: bool,
    ) -> None:
        """Run Python-specific analysis tools.

        Runs ruff (linting), mypy (type checking), and optionally pytest.

        Args:
            source_files: Source file contents.
            result: Result object to populate.
            run_tests: Whether to run pytest.
        """
        # --- Ruff (linter + formatter check) ---
        ruff_result = self.sandbox.run_with_source(
            command="pip install ruff -q && ruff check --output-format=json .",
            source_files=source_files,
            image="python:3.11-slim",
            setup_command=None,
        )
        result.tools_run.append("ruff")

        if ruff_result.stdout:
            issues = self._parse_ruff_output(ruff_result.stdout)
            result.issues.extend(issues)

        if not ruff_result.success and not ruff_result.stdout:
            result.errors.append(f"ruff failed: {ruff_result.stderr[:200]}")

        # --- Mypy (type checking) ---
        mypy_result = self.sandbox.run_with_source(
            command="pip install mypy -q && mypy --ignore-missing-imports --no-error-summary . 2>&1 || true",
            source_files=source_files,
            image="python:3.11-slim",
        )
        result.tools_run.append("mypy")

        if mypy_result.stdout:
            issues = self._parse_mypy_output(mypy_result.stdout)
            result.issues.extend(issues)

        # --- Pytest (tests) ---
        if run_tests and self._has_test_files(source_files, "python"):
            test_result = self.sandbox.run_with_source(
                command="pip install pytest -q && python -m pytest -x --tb=short 2>&1",
                source_files=source_files,
                image="python:3.11-slim",
            )
            result.tools_run.append("pytest")
            result.test_passed = test_result.success
            result.test_output = test_result.output[:5000]

    async def _run_js_ts_analysis(
        self,
        source_files: dict[str, str],
        result: AnalysisResult,
        run_tests: bool,
    ) -> None:
        """Run JavaScript/TypeScript analysis tools.

        Runs eslint and optionally jest/vitest.

        Args:
            source_files: Source file contents.
            result: Result object to populate.
            run_tests: Whether to run tests.
        """
        # Check if package.json exists for dependency context
        has_package_json = "package.json" in source_files

        # --- ESLint ---
        eslint_cmd = (
            "npm init -y > /dev/null 2>&1 && "
            "npm install eslint @eslint/js -q 2>/dev/null && "
            "npx eslint --format json . 2>/dev/null || true"
        )
        eslint_result = self.sandbox.run_with_source(
            command=eslint_cmd,
            source_files=source_files,
            image="node:20-slim",
        )
        result.tools_run.append("eslint")

        if eslint_result.stdout:
            issues = self._parse_eslint_output(eslint_result.stdout)
            result.issues.extend(issues)

        # --- Tests ---
        if run_tests and has_package_json:
            test_result = self.sandbox.run_with_source(
                command="npm install -q 2>/dev/null && npm test 2>&1 || true",
                source_files=source_files,
                image="node:20-slim",
            )
            result.tools_run.append("npm test")
            result.test_passed = test_result.success
            result.test_output = test_result.output[:5000]

    def _parse_ruff_output(self, output: str) -> list[LintIssue]:
        """Parse ruff JSON output into LintIssue objects.

        Args:
            output: Raw ruff JSON output.

        Returns:
            List of parsed lint issues.
        """
        issues: list[LintIssue] = []
        try:
            data = json.loads(output)
            for item in data:
                issues.append(
                    LintIssue(
                        file_path=item.get("filename", ""),
                        line=item.get("location", {}).get("row", 0),
                        column=item.get("location", {}).get("column", 0),
                        severity="error" if item.get("code", "").startswith("E") else "warning",
                        message=item.get("message", ""),
                        rule_id=item.get("code", ""),
                        tool="ruff",
                    )
                )
        except json.JSONDecodeError:
            logger.debug("failed to parse ruff output as JSON")
        return issues

    def _parse_mypy_output(self, output: str) -> list[LintIssue]:
        """Parse mypy text output into LintIssue objects.

        Mypy format: file.py:line: severity: message

        Args:
            output: Raw mypy output.

        Returns:
            List of parsed lint issues.
        """
        issues: list[LintIssue] = []
        pattern = re.compile(r"^(.+?):(\d+):\s*(error|warning|note):\s*(.+)$")

        for line in output.split("\n"):
            match = pattern.match(line.strip())
            if match:
                severity = match.group(3)
                if severity == "note":
                    severity = "info"
                issues.append(
                    LintIssue(
                        file_path=match.group(1),
                        line=int(match.group(2)),
                        severity=severity,
                        message=match.group(4),
                        tool="mypy",
                    )
                )
        return issues

    def _parse_eslint_output(self, output: str) -> list[LintIssue]:
        """Parse ESLint JSON output into LintIssue objects.

        Args:
            output: Raw ESLint JSON output.

        Returns:
            List of parsed lint issues.
        """
        issues: list[LintIssue] = []
        try:
            # Find JSON array in output (may have non-JSON prefix)
            json_start = output.find("[")
            if json_start == -1:
                return issues
            data = json.loads(output[json_start:])

            for file_result in data:
                file_path = file_result.get("filePath", "")
                # Make path relative
                if "/workspace/" in file_path:
                    file_path = file_path.split("/workspace/", 1)[1]

                for msg in file_result.get("messages", []):
                    severity = "error" if msg.get("severity", 1) == 2 else "warning"
                    issues.append(
                        LintIssue(
                            file_path=file_path,
                            line=msg.get("line", 0),
                            column=msg.get("column", 0),
                            severity=severity,
                            message=msg.get("message", ""),
                            rule_id=msg.get("ruleId", "") or "",
                            tool="eslint",
                        )
                    )
        except json.JSONDecodeError:
            logger.debug("failed to parse eslint output as JSON")
        return issues

    @staticmethod
    def _has_test_files(source_files: dict[str, str], language: str) -> bool:
        """Check if the source files include test files.

        Args:
            source_files: File mapping.
            language: Programming language.

        Returns:
            True if test files are present.
        """
        test_patterns: dict[str, list[str]] = {
            "python": ["test_", "_test.py", "tests/"],
            "javascript": [".test.", ".spec.", "__tests__/"],
            "typescript": [".test.", ".spec.", "__tests__/"],
        }

        patterns = test_patterns.get(language, [])
        return any(
            any(p in path for p in patterns)
            for path in source_files.keys()
        )
