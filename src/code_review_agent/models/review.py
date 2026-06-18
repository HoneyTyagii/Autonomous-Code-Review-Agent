"""Review and ReviewComment ORM models.

Persists the full history of code reviews performed by the agent,
including individual comments, decisions, and metrics. This history
powers the review memory and learning features.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    String,
    Integer,
    Text,
    ForeignKey,
    Enum as SAEnum,
    Float,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from code_review_agent.models.base import Base, TimestampMixin, UUIDMixin
from code_review_agent.models.enums import (
    ReviewStatus,
    ReviewDecision,
    CommentSeverity,
    CommentCategory,
)

if TYPE_CHECKING:
    from code_review_agent.models.repository import Repository


class Review(Base, UUIDMixin, TimestampMixin):
    """A single code review performed on a pull request.

    Captures the complete state and outcome of a review run: which PR
    was reviewed, the agent's decision, timing metrics, and links to
    all generated comments.

    Attributes:
        repository_id: Foreign key to the reviewed repository.
        pr_number: The pull request number.
        pr_title: Title of the pull request at review time.
        head_sha: Commit SHA that was reviewed.
        author: PR author's GitHub login.
        status: Current status of the review.
        decision: Final review decision (approve/request_changes/comment).
        summary: The agent's summary review text.
        files_reviewed: Number of files analyzed.
        lines_added: Total lines added in the PR.
        lines_deleted: Total lines removed in the PR.
        duration_seconds: How long the review took.
        llm_tokens_used: Total LLM tokens consumed.
        github_review_id: ID of the posted GitHub review.
        error_message: Error details if the review failed.
    """

    __tablename__ = "reviews"

    repository_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    pr_number: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    pr_title: Mapped[str] = mapped_column(String(1024), default="")
    head_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    author: Mapped[str] = mapped_column(String(255), default="")

    status: Mapped[ReviewStatus] = mapped_column(
        SAEnum(ReviewStatus),
        default=ReviewStatus.PENDING,
        nullable=False,
        index=True,
    )
    decision: Mapped[ReviewDecision | None] = mapped_column(
        SAEnum(ReviewDecision),
        nullable=True,
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    files_reviewed: Mapped[int] = mapped_column(Integer, default=0)
    lines_added: Mapped[int] = mapped_column(Integer, default=0)
    lines_deleted: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    llm_tokens_used: Mapped[int] = mapped_column(Integer, default=0)

    github_review_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    repository: Mapped[Repository] = relationship(
        "Repository", back_populates="reviews"
    )
    comments: Mapped[list[ReviewComment]] = relationship(
        "ReviewComment",
        back_populates="review",
        cascade="all, delete-orphan",
    )

    @property
    def comment_count(self) -> int:
        """Total number of comments in this review."""
        return len(self.comments)

    def __repr__(self) -> str:
        """Return a string representation of the review."""
        return (
            f"<Review pr#{self.pr_number} "
            f"status={self.status.value} decision={self.decision}>"
        )


class ReviewComment(Base, UUIDMixin, TimestampMixin):
    """An individual comment generated during a code review.

    Each comment targets a specific file and line, carries a severity
    and category classification, and may include a suggested fix patch.

    Attributes:
        review_id: Foreign key to the parent review.
        file_path: Path of the file the comment refers to.
        line_number: Line number in the new file version.
        diff_position: Position in the diff for GitHub inline placement.
        severity: How serious the issue is.
        category: The type of issue identified.
        message: The human-readable comment text.
        suggested_fix: Optional code suggestion or patch.
        rule_id: Identifier of the rule/standard that triggered it.
        is_resolved: Whether the issue was addressed.
        posted_to_github: Whether this was successfully posted.
        github_comment_id: ID of the posted GitHub comment.
    """

    __tablename__ = "review_comments"

    review_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("reviews.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    line_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diff_position: Mapped[int | None] = mapped_column(Integer, nullable=True)

    severity: Mapped[CommentSeverity] = mapped_column(
        SAEnum(CommentSeverity),
        default=CommentSeverity.MEDIUM,
        nullable=False,
        index=True,
    )
    category: Mapped[CommentCategory] = mapped_column(
        SAEnum(CommentCategory),
        default=CommentCategory.BEST_PRACTICE,
        nullable=False,
    )

    message: Mapped[str] = mapped_column(Text, nullable=False)
    suggested_fix: Mapped[str | None] = mapped_column(Text, nullable=True)
    rule_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    is_resolved: Mapped[bool] = mapped_column(default=False, nullable=False)
    posted_to_github: Mapped[bool] = mapped_column(default=False, nullable=False)
    github_comment_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    review: Mapped[Review] = relationship("Review", back_populates="comments")

    def __repr__(self) -> str:
        """Return a string representation of the comment."""
        return (
            f"<ReviewComment {self.file_path}:{self.line_number} "
            f"severity={self.severity.value}>"
        )
