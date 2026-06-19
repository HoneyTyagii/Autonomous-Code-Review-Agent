"""Bug prediction module using code change heuristics.

Applies research-backed heuristics to identify code changes that are
statistically more likely to introduce bugs. Uses signals like:
- Size and complexity of changes
- Churn patterns (many changes to same area)
- Anti-patterns in diff (error handling removal, etc.)
- Structural risk indicators

References:
- "Predicting Faults from Cached History" (Kim et al.)
- "A Large-Scale Empirical Study of Just-in-Time Defect Prediction" (Kamei et al.)
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any

from code_review_agent.github.diff_parser import (
    FileDiff,
    PullRequestDiff,
    DiffHunk,
    LineType,
)
from code_review_agent.logging import get_logger

logger = get_logger("bug_predictor")


class RiskLevel(str, Enum):
    """Risk level for bug prediction."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class BugRiskSignal:
    """A single risk signal identified in the code change.

    Attributes:
        file_path: File exhibiting the signal.
        line_number: Approximate line location.
        risk_level: Severity of the risk signal.
        signal_type: Category of the risk signal.
        description: Human-readable explanation.
        confidence: How confident the predictor is (0.0-1.0).
    """

    file_path: str
    line_number: int | None = None
    risk_level: RiskLevel = RiskLevel.MEDIUM
    signal_type: str = ""
    description: str = ""
    confidence: float = 0.5


@dataclass
class BugPrediction:
    """Complete bug prediction analysis for a pull request.

    Attributes:
        overall_risk_score: Normalized risk score (0.0-1.0).
        risk_level: Overall risk classification.
        signals: Individual risk signals detected.
        high_risk_files: Files ranked by bug likelihood.
        recommendations: Actionable recommendations.
    """

    overall_risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    signals: list[BugRiskSignal] = field(default_factory=list)
    high_risk_files: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def has_critical_risk(self) -> bool:
        """Check for any critical risk signals."""
        return any(s.risk_level == RiskLevel.CRITICAL for s in self.signals)

    @property
    def signal_count_by_level(self) -> dict[str, int]:
        """Count signals grouped by risk level."""
        counts: dict[str, int] = {}
        for signal in self.signals:
            counts[signal.risk_level.value] = counts.get(signal.risk_level.value, 0) + 1
        return counts


class BugPredictor:
    """Predicts bug likelihood using code change heuristics.

    Analyzes the structure and content of code changes to assign
    risk scores based on empirically-validated signals that correlate
    with defect introduction.

    The heuristics are grouped into:
    - Size heuristics: large changes are riskier
    - Complexity heuristics: complex diffs are riskier
    - Pattern heuristics: specific anti-patterns in the diff
    - Structural heuristics: changes to risky areas of code
    """

    # Thresholds for size-based risk
    LARGE_FILE_CHANGES = 200  # lines changed in a single file
    LARGE_PR_TOTAL = 500  # total lines changed
    MANY_FILES_THRESHOLD = 15  # files changed

    # Keywords indicating risky areas
    RISKY_AREA_KEYWORDS = [
        "auth", "login", "password", "token", "session",
        "payment", "billing", "charge", "transaction",
        "encrypt", "decrypt", "hash", "crypto",
        "database", "migration", "schema",
        "permission", "access", "role", "admin",
    ]

    def predict(self, diff: PullRequestDiff) -> BugPrediction:
        """Run bug prediction analysis on a pull request diff.

        Args:
            diff: Parsed PR diff with all file changes.

        Returns:
            BugPrediction with risk scores, signals, and recommendations.
        """
        prediction = BugPrediction()

        # Run all heuristic checks
        self._check_size_heuristics(diff, prediction)
        self._check_complexity_heuristics(diff, prediction)
        self._check_pattern_heuristics(diff, prediction)
        self._check_structural_heuristics(diff, prediction)

        # Calculate overall risk score
        prediction.overall_risk_score = self._calculate_risk_score(prediction)
        prediction.risk_level = self._score_to_level(prediction.overall_risk_score)

        # Identify highest-risk files
        prediction.high_risk_files = self._rank_risky_files(diff, prediction)

        # Generate recommendations
        prediction.recommendations = self._generate_recommendations(prediction)

        logger.info(
            "bug prediction complete",
            risk_score=round(prediction.overall_risk_score, 2),
            risk_level=prediction.risk_level.value,
            signals=len(prediction.signals),
        )

        return prediction

    def _check_size_heuristics(
        self, diff: PullRequestDiff, prediction: BugPrediction
    ) -> None:
        """Check size-based risk signals.

        Large changes are harder to review and more likely to contain bugs.
        """
        # Total PR size
        total_changes = diff.total_additions + diff.total_deletions
        if total_changes > self.LARGE_PR_TOTAL:
            prediction.signals.append(
                BugRiskSignal(
                    file_path="(entire PR)",
                    risk_level=RiskLevel.HIGH,
                    signal_type="large_change",
                    description=f"Very large PR ({total_changes} lines changed). Large changes correlate with higher defect rates.",
                    confidence=0.7,
                )
            )

        # Many files changed
        if diff.total_files_changed > self.MANY_FILES_THRESHOLD:
            prediction.signals.append(
                BugRiskSignal(
                    file_path="(entire PR)",
                    risk_level=RiskLevel.MEDIUM,
                    signal_type="many_files",
                    description=f"Changes span {diff.total_files_changed} files. Scattered changes are harder to reason about.",
                    confidence=0.6,
                )
            )

        # Individual large files
        for file_diff in diff.files:
            if file_diff.total_changes > self.LARGE_FILE_CHANGES:
                prediction.signals.append(
                    BugRiskSignal(
                        file_path=file_diff.filename,
                        risk_level=RiskLevel.MEDIUM,
                        signal_type="large_file_change",
                        description=f"File has {file_diff.total_changes} lines changed. Consider breaking into smaller changes.",
                        confidence=0.6,
                    )
                )

    def _check_complexity_heuristics(
        self, diff: PullRequestDiff, prediction: BugPrediction
    ) -> None:
        """Check complexity-based risk signals.

        Changes with high entropy (many interleaved additions/deletions)
        are riskier than clean additions or simple replacements.
        """
        for file_diff in diff.files:
            for hunk in file_diff.hunks:
                # Check for high-entropy hunks (many interleaved add/remove)
                entropy = self._calculate_hunk_entropy(hunk)
                if entropy > 0.7 and len(hunk.lines) > 10:
                    prediction.signals.append(
                        BugRiskSignal(
                            file_path=file_diff.filename,
                            line_number=hunk.new_start,
                            risk_level=RiskLevel.MEDIUM,
                            signal_type="high_entropy",
                            description="Complex interleaved changes (high entropy). Difficult to verify correctness.",
                            confidence=0.5,
                        )
                    )

                # Check for large hunks without context
                if len(hunk.lines) > 50 and hunk.old_count == 0:
                    prediction.signals.append(
                        BugRiskSignal(
                            file_path=file_diff.filename,
                            line_number=hunk.new_start,
                            risk_level=RiskLevel.LOW,
                            signal_type="large_addition",
                            description="Large block of new code without modification to existing logic.",
                            confidence=0.4,
                        )
                    )

    def _check_pattern_heuristics(
        self, diff: PullRequestDiff, prediction: BugPrediction
    ) -> None:
        """Check for anti-pattern signals in the diff content.

        Looks for specific patterns that historically correlate with bugs.
        """
        for file_diff in diff.files:
            if file_diff.is_deleted:
                continue

            for hunk in file_diff.hunks:
                self._check_error_handling_removal(file_diff, hunk, prediction)
                self._check_null_check_removal(file_diff, hunk, prediction)
                self._check_todo_fixme_addition(file_diff, hunk, prediction)
                self._check_commented_code(file_diff, hunk, prediction)
                self._check_broad_exception_handling(file_diff, hunk, prediction)

    def _check_error_handling_removal(
        self, file_diff: FileDiff, hunk: DiffHunk, prediction: BugPrediction
    ) -> None:
        """Detect removal of error handling code."""
        error_patterns = re.compile(
            r"(try|catch|except|finally|rescue|throw|raise|Error|Exception)",
            re.IGNORECASE,
        )
        removed_error_lines = 0
        for line in hunk.removed_lines:
            if error_patterns.search(line.content):
                removed_error_lines += 1

        if removed_error_lines >= 2:
            prediction.signals.append(
                BugRiskSignal(
                    file_path=file_diff.filename,
                    line_number=hunk.new_start,
                    risk_level=RiskLevel.HIGH,
                    signal_type="error_handling_removed",
                    description=f"Error handling code removed ({removed_error_lines} lines). This may introduce unhandled exceptions.",
                    confidence=0.75,
                )
            )

    def _check_null_check_removal(
        self, file_diff: FileDiff, hunk: DiffHunk, prediction: BugPrediction
    ) -> None:
        """Detect removal of null/None/undefined checks."""
        null_patterns = re.compile(
            r"(is None|is not None|!= None|== None|=== null|!== null|!= null|== null|\?\?|\.\\?)",
        )
        for line in hunk.removed_lines:
            if null_patterns.search(line.content):
                prediction.signals.append(
                    BugRiskSignal(
                        file_path=file_diff.filename,
                        line_number=line.old_line_number,
                        risk_level=RiskLevel.HIGH,
                        signal_type="null_check_removed",
                        description="Null/None check removed. May cause NullPointerException or TypeError at runtime.",
                        confidence=0.7,
                    )
                )
                break  # One signal per hunk is enough

    def _check_todo_fixme_addition(
        self, file_diff: FileDiff, hunk: DiffHunk, prediction: BugPrediction
    ) -> None:
        """Detect addition of TODO/FIXME/HACK comments."""
        todo_pattern = re.compile(r"#.*\b(TODO|FIXME|HACK|XXX|WORKAROUND)\b", re.IGNORECASE)
        for line in hunk.added_lines:
            if todo_pattern.search(line.content):
                prediction.signals.append(
                    BugRiskSignal(
                        file_path=file_diff.filename,
                        line_number=line.new_line_number,
                        risk_level=RiskLevel.LOW,
                        signal_type="todo_added",
                        description="TODO/FIXME comment added. Indicates known incomplete or problematic code.",
                        confidence=0.4,
                    )
                )
                break

    def _check_commented_code(
        self, file_diff: FileDiff, hunk: DiffHunk, prediction: BugPrediction
    ) -> None:
        """Detect addition of commented-out code blocks."""
        commented_code_count = 0
        code_indicators = re.compile(r"^\s*[#//]\s*(if|for|while|def|class|return|import|var|let|const)\s")

        for line in hunk.added_lines:
            if code_indicators.match(line.content):
                commented_code_count += 1

        if commented_code_count >= 3:
            prediction.signals.append(
                BugRiskSignal(
                    file_path=file_diff.filename,
                    line_number=hunk.new_start,
                    risk_level=RiskLevel.LOW,
                    signal_type="commented_code",
                    description=f"Commented-out code added ({commented_code_count} lines). Dead code should be removed, not commented.",
                    confidence=0.5,
                )
            )

    def _check_broad_exception_handling(
        self, file_diff: FileDiff, hunk: DiffHunk, prediction: BugPrediction
    ) -> None:
        """Detect addition of overly broad exception handling."""
        broad_patterns = [
            re.compile(r"except\s*:"),  # bare except
            re.compile(r"except\s+Exception\s*:"),  # catch-all Exception
            re.compile(r"catch\s*\(\s*(e|err|error)?\s*\)\s*\{?\s*\}"),  # empty catch block
        ]

        for line in hunk.added_lines:
            for pattern in broad_patterns:
                if pattern.search(line.content):
                    prediction.signals.append(
                        BugRiskSignal(
                            file_path=file_diff.filename,
                            line_number=line.new_line_number,
                            risk_level=RiskLevel.MEDIUM,
                            signal_type="broad_exception",
                            description="Overly broad exception handling. May silently swallow important errors.",
                            confidence=0.6,
                        )
                    )
                    return  # One per hunk

    def _check_structural_heuristics(
        self, diff: PullRequestDiff, prediction: BugPrediction
    ) -> None:
        """Check structural risk signals.

        Identifies changes to security-sensitive or high-impact areas.
        """
        for file_diff in diff.files:
            filename_lower = file_diff.filename.lower()

            # Changes to security-sensitive files
            for keyword in self.RISKY_AREA_KEYWORDS:
                if keyword in filename_lower:
                    prediction.signals.append(
                        BugRiskSignal(
                            file_path=file_diff.filename,
                            risk_level=RiskLevel.MEDIUM,
                            signal_type="sensitive_area",
                            description=f"Changes to security/business-critical area ('{keyword}' in path). Requires careful review.",
                            confidence=0.6,
                        )
                    )
                    break

            # Changes to configuration files
            config_patterns = [
                "config", ".env", "settings", "docker-compose",
                "nginx", "apache", "Dockerfile",
            ]
            if any(p in filename_lower for p in config_patterns):
                if file_diff.total_changes > 5:
                    prediction.signals.append(
                        BugRiskSignal(
                            file_path=file_diff.filename,
                            risk_level=RiskLevel.MEDIUM,
                            signal_type="config_change",
                            description="Configuration file modified. Misconfigurations can cause production outages.",
                            confidence=0.5,
                        )
                    )

    @staticmethod
    def _calculate_hunk_entropy(hunk: DiffHunk) -> float:
        """Calculate the entropy (interleaving) of a hunk.

        High entropy means additions and deletions are highly interleaved,
        which is harder to review and more bug-prone.

        Args:
            hunk: A diff hunk.

        Returns:
            Entropy score between 0.0 (all one type) and 1.0 (fully interleaved).
        """
        if not hunk.lines:
            return 0.0

        transitions = 0
        prev_type = None
        for line in hunk.lines:
            if line.line_type == LineType.CONTEXT:
                continue
            if prev_type is not None and line.line_type != prev_type:
                transitions += 1
            prev_type = line.line_type

        non_context = sum(1 for l in hunk.lines if l.line_type != LineType.CONTEXT)
        if non_context <= 1:
            return 0.0

        return transitions / (non_context - 1)

    @staticmethod
    def _calculate_risk_score(prediction: BugPrediction) -> float:
        """Calculate overall risk score from individual signals.

        Weighted average based on signal severity and confidence.

        Args:
            prediction: Prediction with populated signals.

        Returns:
            Normalized score between 0.0 and 1.0.
        """
        if not prediction.signals:
            return 0.0

        level_weights = {
            RiskLevel.CRITICAL: 1.0,
            RiskLevel.HIGH: 0.75,
            RiskLevel.MEDIUM: 0.4,
            RiskLevel.LOW: 0.15,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for signal in prediction.signals:
            weight = level_weights.get(signal.risk_level, 0.3)
            weighted_sum += weight * signal.confidence
            total_weight += 1.0

        # Normalize to 0-1, with diminishing returns for many signals
        raw_score = weighted_sum / max(total_weight, 1.0)
        # Apply sigmoid-like scaling for many signals
        signal_multiplier = min(1.0, len(prediction.signals) / 5.0)

        return min(1.0, raw_score * (0.5 + 0.5 * signal_multiplier))

    @staticmethod
    def _score_to_level(score: float) -> RiskLevel:
        """Convert a numeric risk score to a risk level.

        Args:
            score: Risk score (0.0-1.0).

        Returns:
            Corresponding RiskLevel.
        """
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.35:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _rank_risky_files(
        self, diff: PullRequestDiff, prediction: BugPrediction
    ) -> list[str]:
        """Rank files by bug risk based on signal density.

        Args:
            diff: The PR diff.
            prediction: Prediction with signals.

        Returns:
            File paths ordered by descending risk.
        """
        file_scores: dict[str, float] = {}

        for signal in prediction.signals:
            if signal.file_path == "(entire PR)":
                continue
            file_scores[signal.file_path] = (
                file_scores.get(signal.file_path, 0.0) + signal.confidence
            )

        # Sort by score descending
        ranked = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        return [f for f, _ in ranked[:10]]

    @staticmethod
    def _generate_recommendations(prediction: BugPrediction) -> list[str]:
        """Generate actionable recommendations based on signals.

        Args:
            prediction: Populated prediction.

        Returns:
            List of recommendation strings.
        """
        recommendations: list[str] = []
        signal_types = {s.signal_type for s in prediction.signals}

        if "large_change" in signal_types:
            recommendations.append(
                "Consider breaking this PR into smaller, focused changes for easier review."
            )

        if "error_handling_removed" in signal_types:
            recommendations.append(
                "Error handling was removed. Ensure error cases are still properly handled."
            )

        if "null_check_removed" in signal_types:
            recommendations.append(
                "Null/None checks were removed. Verify callers cannot pass null values."
            )

        if "sensitive_area" in signal_types:
            recommendations.append(
                "Changes touch security-sensitive code. Consider requesting a security-focused review."
            )

        if "high_entropy" in signal_types:
            recommendations.append(
                "Some changes have complex interleaved modifications. Extra attention to logic correctness is recommended."
            )

        if "broad_exception" in signal_types:
            recommendations.append(
                "Broad exception handling detected. Consider catching specific exceptions and logging errors."
            )

        if not recommendations and prediction.overall_risk_score < 0.2:
            recommendations.append("Low-risk change. Standard review should suffice.")

        return recommendations
