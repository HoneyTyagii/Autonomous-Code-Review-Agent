"""Structured output schemas for the review engine.

Defines the JSON schemas that the LLM must produce, enabling
reliable parsing of review results into actionable comments.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommentOutput:
    """A single review comment produced by the LLM.

    Attributes:
        file_path: File the comment targets.
        line_number: Line number in the new file.
        severity: critical | high | medium | low | info.
        category: security | bug | performance | style | etc.
        message: Human-readable explanation of the issue.
        suggested_fix: Optional code suggestion to fix the issue.
        rule_id: Related coding standard rule, if any.
    """

    file_path: str
    line_number: int | None
    severity: str
    category: str
    message: str
    suggested_fix: str | None = None
    rule_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommentOutput":
        """Create from a dictionary (LLM JSON output)."""
        return cls(
            file_path=data.get("file_path", ""),
            line_number=data.get("line_number"),
            severity=data.get("severity", "medium"),
            category=data.get("category", "best_practice"),
            message=data.get("message", ""),
            suggested_fix=data.get("suggested_fix"),
            rule_id=data.get("rule_id"),
        )


@dataclass
class FileReviewOutput:
    """Review output for a single file.

    Attributes:
        file_path: The reviewed file path.
        summary: Brief summary of findings for this file.
        comments: List of specific comments/issues found.
    """

    file_path: str
    summary: str = ""
    comments: list[CommentOutput] = field(default_factory=list)


@dataclass
class ReviewOutput:
    """Complete structured output from the review engine.

    Attributes:
        summary: Overall review summary text.
        decision: approve | request_changes | comment.
        confidence: Confidence in the decision (0.0-1.0).
        file_reviews: Per-file review results.
        total_comments: Total number of comments generated.
        critical_count: Number of critical severity comments.
        high_count: Number of high severity comments.
    """

    summary: str = ""
    decision: str = "comment"
    confidence: float = 0.5
    file_reviews: list[FileReviewOutput] = field(default_factory=list)

    @property
    def total_comments(self) -> int:
        """Total comments across all files."""
        return sum(len(fr.comments) for fr in self.file_reviews)

    @property
    def critical_count(self) -> int:
        """Number of critical severity issues."""
        return sum(
            1 for fr in self.file_reviews
            for c in fr.comments if c.severity == "critical"
        )

    @property
    def high_count(self) -> int:
        """Number of high severity issues."""
        return sum(
            1 for fr in self.file_reviews
            for c in fr.comments if c.severity == "high"
        )

    @property
    def all_comments(self) -> list[CommentOutput]:
        """Flat list of all comments from all files."""
        return [c for fr in self.file_reviews for c in fr.comments]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewOutput":
        """Parse the full LLM JSON output into a ReviewOutput."""
        file_reviews = []
        for fr_data in data.get("file_reviews", []):
            comments = [
                CommentOutput.from_dict(c)
                for c in fr_data.get("comments", [])
            ]
            file_reviews.append(
                FileReviewOutput(
                    file_path=fr_data.get("file_path", ""),
                    summary=fr_data.get("summary", ""),
                    comments=comments,
                )
            )

        return cls(
            summary=data.get("summary", ""),
            decision=data.get("decision", "comment"),
            confidence=data.get("confidence", 0.5),
            file_reviews=file_reviews,
        )


# JSON schema passed to LLM for structured output
REVIEW_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["summary", "decision", "confidence", "file_reviews"],
    "properties": {
        "summary": {
            "type": "string",
            "description": "Overall review summary (2-4 sentences).",
        },
        "decision": {
            "type": "string",
            "enum": ["approve", "request_changes", "comment"],
            "description": "Final review decision.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in the decision.",
        },
        "file_reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["file_path", "summary", "comments"],
                "properties": {
                    "file_path": {"type": "string"},
                    "summary": {"type": "string"},
                    "comments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["file_path", "line_number", "severity", "category", "message"],
                            "properties": {
                                "file_path": {"type": "string"},
                                "line_number": {"type": ["integer", "null"]},
                                "severity": {
                                    "type": "string",
                                    "enum": ["critical", "high", "medium", "low", "info"],
                                },
                                "category": {
                                    "type": "string",
                                    "enum": [
                                        "security", "bug", "performance", "style",
                                        "maintainability", "documentation", "testing",
                                        "best_practice", "naming", "complexity",
                                    ],
                                },
                                "message": {"type": "string"},
                                "suggested_fix": {"type": ["string", "null"]},
                                "rule_id": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
            },
        },
    },
}
