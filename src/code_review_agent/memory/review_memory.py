"""Review memory for storing and retrieving past review patterns.

Persists review outcomes, comments, and patterns in both the relational
database (structured queries) and vector store (semantic search) to
enable learning from historical reviews.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
import uuid

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from code_review_agent.core.schemas import ReviewOutput, CommentOutput
from code_review_agent.github.pr_fetcher import PRContext
from code_review_agent.models.enums import (
    ReviewStatus,
    ReviewDecision,
    CommentSeverity,
    CommentCategory,
)
from code_review_agent.models.review import Review, ReviewComment
from code_review_agent.models.repository import Repository
from code_review_agent.rag.embeddings import EmbeddingService
from code_review_agent.rag.vector_store import VectorStoreClient
from code_review_agent.logging import get_logger

logger = get_logger("review_memory")


@dataclass
class ReviewPattern:
    """A recurring pattern observed across past reviews.

    Attributes:
        pattern_type: Category of pattern (e.g., "common_issue", "author_style").
        description: Human-readable description.
        frequency: How often this pattern occurs.
        files_affected: File patterns where this recurs.
        example_comment: Example comment for this pattern.
    """

    pattern_type: str
    description: str
    frequency: int = 0
    files_affected: list[str] = field(default_factory=list)
    example_comment: str = ""


@dataclass
class AuthorProfile:
    """Profile of a PR author based on past reviews.

    Attributes:
        author: GitHub login.
        total_prs_reviewed: Number of PRs reviewed from this author.
        common_issues: Most frequent issue categories.
        approval_rate: Percentage of PRs approved on first review.
        avg_issues_per_pr: Average number of issues found per PR.
    """

    author: str
    total_prs_reviewed: int = 0
    common_issues: list[str] = field(default_factory=list)
    approval_rate: float = 0.0
    avg_issues_per_pr: float = 0.0


class ReviewMemory:
    """Stores and queries review history for learning and context.

    Combines relational storage (PostgreSQL) for structured queries
    with vector storage (ChromaDB) for semantic similarity search
    across past review comments.

    Attributes:
        embedding_service: Service for embedding review comments.
        vector_store: ChromaDB client for semantic search.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreClient,
    ) -> None:
        """Initialize review memory.

        Args:
            embedding_service: Configured embedding service.
            vector_store: ChromaDB client.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def store_review(
        self,
        session: AsyncSession,
        pr_context: PRContext,
        review_output: ReviewOutput,
        duration_seconds: float = 0.0,
        tokens_used: int = 0,
    ) -> Review:
        """Persist a completed review to the database and vector store.

        Args:
            session: Database session.
            pr_context: PR context that was reviewed.
            review_output: The review results.
            duration_seconds: How long the review took.
            tokens_used: Total LLM tokens consumed.

        Returns:
            The persisted Review model instance.
        """
        # Find or create repository record
        repo = await self._get_or_create_repo(session, pr_context)

        # Map decision string to enum
        decision_map = {
            "approve": ReviewDecision.APPROVE,
            "request_changes": ReviewDecision.REQUEST_CHANGES,
            "comment": ReviewDecision.COMMENT,
        }

        # Create Review record
        review = Review(
            id=uuid.uuid4(),
            repository_id=repo.id,
            pr_number=pr_context.pr_number,
            pr_title=pr_context.title,
            head_sha=pr_context.head_sha,
            author=pr_context.author,
            status=ReviewStatus.COMPLETED,
            decision=decision_map.get(review_output.decision),
            summary=review_output.summary,
            files_reviewed=len(pr_context.reviewable_files),
            lines_added=pr_context.diff.total_additions,
            lines_deleted=pr_context.diff.total_deletions,
            duration_seconds=duration_seconds,
            llm_tokens_used=tokens_used,
        )
        session.add(review)

        # Create ReviewComment records
        severity_map = {s.value: s for s in CommentSeverity}
        category_map = {c.value: c for c in CommentCategory}

        for comment in review_output.all_comments:
            review_comment = ReviewComment(
                id=uuid.uuid4(),
                review_id=review.id,
                file_path=comment.file_path,
                line_number=comment.line_number,
                severity=severity_map.get(comment.severity, CommentSeverity.MEDIUM),
                category=category_map.get(comment.category, CommentCategory.BEST_PRACTICE),
                message=comment.message,
                suggested_fix=comment.suggested_fix,
                rule_id=comment.rule_id,
            )
            session.add(review_comment)

        # Update repo stats
        repo.total_reviews += 1
        await session.flush()

        # Store comments in vector store for semantic retrieval
        await self._store_comments_in_vector_store(
            pr_context.repo_full_name, review, review_output
        )

        logger.info(
            "review stored",
            repo=pr_context.repo_full_name,
            pr=pr_context.pr_number,
            comments=len(review_output.all_comments),
        )

        return review

    async def get_similar_past_comments(
        self,
        repo_full_name: str,
        code_snippet: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Find past review comments similar to a code snippet.

        Uses semantic search to find comments made on similar code,
        helping the review engine avoid repeating or contradicting
        past feedback.

        Args:
            repo_full_name: Repository to search within.
            code_snippet: Code to find similar past comments for.
            n_results: Maximum results to return.

        Returns:
            List of past comment data with metadata.
        """
        embedding = await self.embedding_service.embed(code_snippet)

        results = self.vector_store.query(
            collection_name=VectorStoreClient.COLLECTION_REVIEWS,
            query_embedding=embedding,
            n_results=n_results,
            where={"repo": repo_full_name},
        )

        return [
            {
                "comment": r.content,
                "severity": r.metadata.get("severity", ""),
                "category": r.metadata.get("category", ""),
                "file_path": r.metadata.get("file_path", ""),
                "similarity": r.score,
            }
            for r in results
        ]

    async def get_author_profile(
        self, session: AsyncSession, author: str, repo_id: uuid.UUID
    ) -> AuthorProfile:
        """Build a profile of an author based on past reviews.

        Args:
            session: Database session.
            author: GitHub login of the PR author.
            repo_id: Repository UUID.

        Returns:
            AuthorProfile with historical statistics.
        """
        # Count total reviews for this author
        total_query = select(func.count()).where(
            and_(Review.author == author, Review.repository_id == repo_id)
        )
        total_result = await session.execute(total_query)
        total_prs = total_result.scalar() or 0

        # Count approvals
        approved_query = select(func.count()).where(
            and_(
                Review.author == author,
                Review.repository_id == repo_id,
                Review.decision == ReviewDecision.APPROVE,
            )
        )
        approved_result = await session.execute(approved_query)
        approved = approved_result.scalar() or 0

        # Get most common issue categories
        category_query = (
            select(ReviewComment.category, func.count().label("cnt"))
            .join(Review)
            .where(and_(Review.author == author, Review.repository_id == repo_id))
            .group_by(ReviewComment.category)
            .order_by(func.count().desc())
            .limit(5)
        )
        category_result = await session.execute(category_query)
        common_issues = [row[0].value for row in category_result.fetchall()]

        # Average issues per PR
        comment_count_query = select(func.count()).select_from(ReviewComment).join(Review).where(
            and_(Review.author == author, Review.repository_id == repo_id)
        )
        comment_result = await session.execute(comment_count_query)
        total_comments = comment_result.scalar() or 0
        avg_issues = total_comments / max(total_prs, 1)

        return AuthorProfile(
            author=author,
            total_prs_reviewed=total_prs,
            common_issues=common_issues,
            approval_rate=approved / max(total_prs, 1),
            avg_issues_per_pr=round(avg_issues, 1),
        )

    async def get_repository_patterns(
        self, session: AsyncSession, repo_id: uuid.UUID, limit: int = 10
    ) -> list[ReviewPattern]:
        """Identify recurring review patterns in a repository.

        Args:
            session: Database session.
            repo_id: Repository UUID.
            limit: Maximum patterns to return.

        Returns:
            List of identified patterns ordered by frequency.
        """
        # Find most common comment messages (grouped by category + similar message)
        pattern_query = (
            select(
                ReviewComment.category,
                ReviewComment.file_path,
                ReviewComment.message,
                func.count().label("frequency"),
            )
            .join(Review)
            .where(Review.repository_id == repo_id)
            .group_by(
                ReviewComment.category,
                ReviewComment.file_path,
                ReviewComment.message,
            )
            .order_by(func.count().desc())
            .limit(limit)
        )

        result = await session.execute(pattern_query)
        rows = result.fetchall()

        patterns: list[ReviewPattern] = []
        for row in rows:
            category, file_path, message, frequency = row
            if frequency >= 2:  # Only include recurring patterns
                patterns.append(
                    ReviewPattern(
                        pattern_type=category.value if hasattr(category, "value") else str(category),
                        description=message[:200],
                        frequency=frequency,
                        files_affected=[file_path],
                        example_comment=message,
                    )
                )

        return patterns

    async def _store_comments_in_vector_store(
        self,
        repo_full_name: str,
        review: Review,
        review_output: ReviewOutput,
    ) -> None:
        """Store review comments in the vector store for semantic retrieval.

        Args:
            repo_full_name: Repository name.
            review: The persisted Review record.
            review_output: Review output with comments.
        """
        comments = review_output.all_comments
        if not comments:
            return

        texts: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for i, comment in enumerate(comments):
            # Create a rich text representation for embedding
            text = (
                f"[{comment.severity}] [{comment.category}] "
                f"{comment.file_path}: {comment.message}"
            )
            texts.append(text)

            doc_id = hashlib.sha256(
                f"{repo_full_name}:{review.id}:{i}".encode()
            ).hexdigest()[:16]
            ids.append(doc_id)

            metadatas.append({
                "repo": repo_full_name,
                "severity": comment.severity,
                "category": comment.category,
                "file_path": comment.file_path,
                "pr_number": str(review.pr_number),
                "author": review.author,
            })

        # Embed and store
        embeddings = await self.embedding_service.embed_many(texts)
        self.vector_store.add_documents(
            collection_name=VectorStoreClient.COLLECTION_REVIEWS,
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def _get_or_create_repo(
        self, session: AsyncSession, pr_context: PRContext
    ) -> Repository:
        """Get or create a Repository record.

        Args:
            session: Database session.
            pr_context: PR context with repo metadata.

        Returns:
            The Repository model instance.
        """
        query = select(Repository).where(
            Repository.full_name == pr_context.repo_full_name
        )
        result = await session.execute(query)
        repo = result.scalar_one_or_none()

        if repo is None:
            repo = Repository(
                id=uuid.uuid4(),
                github_id=0,  # Will be updated on first webhook
                full_name=pr_context.repo_full_name,
                owner=pr_context.owner,
                name=pr_context.repo,
                installation_id=pr_context.installation_id,
                default_branch=pr_context.base_branch,
                is_active=True,
                is_indexed=False,
                total_reviews=0,
            )
            session.add(repo)
            await session.flush()

        return repo
