"""Enumerations used across database models and the review engine."""

from enum import Enum


class ReviewStatus(str, Enum):
    """Lifecycle status of a code review."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReviewDecision(str, Enum):
    """Final decision rendered by the review agent."""

    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    COMMENT = "comment"


class CommentSeverity(str, Enum):
    """Severity level of a review comment."""

    CRITICAL = "critical"  # Blocking issues: security, data loss, crashes
    HIGH = "high"  # Significant bugs or design problems
    MEDIUM = "medium"  # Code quality, maintainability concerns
    LOW = "low"  # Minor style or readability suggestions
    INFO = "info"  # Informational, non-actionable notes


class CommentCategory(str, Enum):
    """Category classifying the nature of a review comment."""

    SECURITY = "security"
    BUG = "bug"
    PERFORMANCE = "performance"
    STYLE = "style"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    BEST_PRACTICE = "best_practice"
    NAMING = "naming"
    COMPLEXITY = "complexity"
