"""initial schema: repositories, reviews, review_comments

Revision ID: 0001
Revises:
Create Date: 2026-01-01 00:00:00.000000

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the initial database schema."""
    # --- repositories ---
    op.create_table(
        "repositories",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("github_id", sa.BigInteger(), nullable=False),
        sa.Column("full_name", sa.String(length=512), nullable=False),
        sa.Column("owner", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("installation_id", sa.BigInteger(), nullable=False),
        sa.Column("default_branch", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_indexed", sa.Boolean(), nullable=False),
        sa.Column("indexed_commit_sha", sa.String(length=40), nullable=True),
        sa.Column("coding_standards_doc", sa.Text(), nullable=True),
        sa.Column("review_config", sa.Text(), nullable=True),
        sa.Column("total_reviews", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_repositories_github_id", "repositories", ["github_id"], unique=True)
    op.create_index("ix_repositories_full_name", "repositories", ["full_name"], unique=True)
    op.create_index("ix_repositories_installation_id", "repositories", ["installation_id"])

    # --- reviews ---
    op.create_table(
        "reviews",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("repository_id", sa.Uuid(), nullable=False),
        sa.Column("pr_number", sa.Integer(), nullable=False),
        sa.Column("pr_title", sa.String(length=1024), nullable=True),
        sa.Column("head_sha", sa.String(length=40), nullable=False),
        sa.Column("author", sa.String(length=255), nullable=True),
        sa.Column(
            "status",
            sa.Enum("PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "SKIPPED", name="reviewstatus"),
            nullable=False,
        ),
        sa.Column(
            "decision",
            sa.Enum("APPROVE", "REQUEST_CHANGES", "COMMENT", name="reviewdecision"),
            nullable=True,
        ),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("files_reviewed", sa.Integer(), nullable=True),
        sa.Column("lines_added", sa.Integer(), nullable=True),
        sa.Column("lines_deleted", sa.Integer(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("llm_tokens_used", sa.Integer(), nullable=True),
        sa.Column("github_review_id", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_reviews_repository_id", "reviews", ["repository_id"])
    op.create_index("ix_reviews_pr_number", "reviews", ["pr_number"])
    op.create_index("ix_reviews_status", "reviews", ["status"])

    # --- review_comments ---
    op.create_table(
        "review_comments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("review_id", sa.Uuid(), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=False),
        sa.Column("line_number", sa.Integer(), nullable=True),
        sa.Column("diff_position", sa.Integer(), nullable=True),
        sa.Column(
            "severity",
            sa.Enum("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO", name="commentseverity"),
            nullable=False,
        ),
        sa.Column(
            "category",
            sa.Enum(
                "SECURITY", "BUG", "PERFORMANCE", "STYLE", "MAINTAINABILITY",
                "DOCUMENTATION", "TESTING", "BEST_PRACTICE", "NAMING", "COMPLEXITY",
                name="commentcategory",
            ),
            nullable=False,
        ),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("suggested_fix", sa.Text(), nullable=True),
        sa.Column("rule_id", sa.String(length=255), nullable=True),
        sa.Column("is_resolved", sa.Boolean(), nullable=False),
        sa.Column("posted_to_github", sa.Boolean(), nullable=False),
        sa.Column("github_comment_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["review_id"], ["reviews.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_review_comments_review_id", "review_comments", ["review_id"])
    op.create_index("ix_review_comments_severity", "review_comments", ["severity"])


def downgrade() -> None:
    """Drop the initial database schema."""
    op.drop_table("review_comments")
    op.drop_table("reviews")
    op.drop_table("repositories")
    sa.Enum(name="commentcategory").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="commentseverity").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="reviewdecision").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="reviewstatus").drop(op.get_bind(), checkfirst=True)
