"""Repository ORM model.

Tracks repositories the agent has reviewed, along with their
configuration, coding standards reference, and indexing state.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, Boolean, Text, BigInteger
from sqlalchemy.orm import Mapped, mapped_column, relationship

from code_review_agent.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from code_review_agent.models.review import Review


class Repository(Base, UUIDMixin, TimestampMixin):
    """A GitHub repository tracked by the review agent.

    Stores repository identity, the GitHub App installation that grants
    access, configuration flags, and the state of the RAG index built
    for repository understanding.

    Attributes:
        github_id: GitHub's numeric repository ID.
        full_name: Repository full name (owner/repo).
        owner: Repository owner login.
        name: Repository name.
        installation_id: GitHub App installation ID granting access.
        default_branch: The repository's default branch.
        is_active: Whether automated reviews are enabled.
        is_indexed: Whether the RAG index has been built.
        indexed_commit_sha: The commit SHA the index was last built from.
        coding_standards_doc: Path or reference to coding standards.
        review_config: JSON configuration for review behavior.
    """

    __tablename__ = "repositories"

    github_id: Mapped[int] = mapped_column(
        BigInteger, unique=True, index=True, nullable=False
    )
    full_name: Mapped[str] = mapped_column(
        String(512), unique=True, index=True, nullable=False
    )
    owner: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    installation_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    default_branch: Mapped[str] = mapped_column(String(255), default="main")

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_indexed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    indexed_commit_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)

    coding_standards_doc: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_config: Mapped[str | None] = mapped_column(Text, nullable=True)

    total_reviews: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relationships
    reviews: Mapped[list[Review]] = relationship(
        "Review",
        back_populates="repository",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """Return a string representation of the repository."""
        return f"<Repository {self.full_name} (active={self.is_active})>"
