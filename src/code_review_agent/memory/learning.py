"""Review learning engine that improves over time.

Analyzes past review patterns to provide context-aware suggestions,
avoid repeated feedback, and adapt to repository-specific conventions.
"""

from dataclasses import dataclass, field
from typing import Any
import uuid

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from code_review_agent.memory.review_memory import (
    ReviewMemory,
    ReviewPattern,
    AuthorProfile,
)
from code_review_agent.models.review import Review, ReviewComment
from code_review_agent.models.enums import ReviewDecision, CommentSeverity
from code_review_agent.logging import get_logger

logger = get_logger("learning")


@dataclass
class LearningContext:
    """Context derived from past reviews to inform current review.

    Attributes:
        author_profile: Profile of the current PR author.
        repo_patterns: Recurring patterns in this repository.
        similar_past_comments: Past comments on similar code.
        suppress_rules: Rules that should be suppressed (false positive history).
        emphasis_areas: Areas to pay extra attention to.
    """

    author_profile: AuthorProfile | None = None
    repo_patterns: list[ReviewPattern] = field(default_factory=list)
    similar_past_comments: list[dict[str, Any]] = field(default_factory=list)
    suppress_rules: list[str] = field(default_factory=list)
    emphasis_areas: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Format learning context for inclusion in LLM prompt.

        Returns:
            Formatted context string.
        """
        parts: list[str] = []

        if self.author_profile and self.author_profile.total_prs_reviewed > 0:
            profile = self.author_profile
            parts.append(
                f"## Author History ({profile.author}):\n"
                f"- PRs reviewed: {profile.total_prs_reviewed}\n"
                f"- Approval rate: {profile.approval_rate:.0%}\n"
                f"- Common issues: {', '.join(profile.common_issues[:3])}\n"
                f"- Avg issues/PR: {profile.avg_issues_per_pr}"
            )

        if self.repo_patterns:
            patterns_text = "\n".join(
                f"- [{p.pattern_type}] {p.description} (seen {p.frequency}x)"
                for p in self.repo_patterns[:5]
            )
            parts.append(f"## Recurring Repository Patterns:\n{patterns_text}")

        if self.similar_past_comments:
            comments_text = "\n".join(
                f"- [{c['severity']}] {c['comment'][:100]}"
                for c in self.similar_past_comments[:3]
            )
            parts.append(
                f"## Similar Past Comments (avoid repeating unless still relevant):\n"
                f"{comments_text}"
            )

        if self.emphasis_areas:
            parts.append(
                f"## Areas Requiring Extra Attention:\n"
                f"- " + "\n- ".join(self.emphasis_areas)
            )

        if self.suppress_rules:
            parts.append(
                f"## Suppressed Rules (known false positives in this repo):\n"
                f"- " + "\n- ".join(self.suppress_rules)
            )

        return "\n\n".join(parts) if parts else ""


class ReviewLearner:
    """Learns from past reviews to improve future review quality.

    Provides contextual learning signals by:
    - Tracking author patterns (common mistakes, style preferences)
    - Identifying repository conventions (recurring standards)
    - Finding similar past comments (avoid repetition)
    - Detecting false positive patterns (suppress noise)
    - Highlighting areas needing extra attention

    Attributes:
        memory: Review memory for data retrieval.
    """

    # Minimum reviews needed before learning signals are reliable
    MIN_REVIEWS_FOR_LEARNING = 3

    def __init__(self, memory: ReviewMemory) -> None:
        """Initialize the review learner.

        Args:
            memory: Configured ReviewMemory instance.
        """
        self.memory = memory

    async def build_learning_context(
        self,
        session: AsyncSession,
        repo_id: uuid.UUID,
        author: str,
        code_snippets: list[str],
    ) -> LearningContext:
        """Build a learning context for a new review.

        Gathers historical data and patterns to inform the
        current review process.

        Args:
            session: Database session.
            repo_id: Repository UUID.
            author: PR author's GitHub login.
            code_snippets: Representative code from the current PR.

        Returns:
            LearningContext with all gathered intelligence.
        """
        context = LearningContext()

        # Get author profile
        context.author_profile = await self.memory.get_author_profile(
            session, author, repo_id
        )

        # Get repository patterns
        context.repo_patterns = await self.memory.get_repository_patterns(
            session, repo_id
        )

        # Find similar past comments
        if code_snippets:
            combined_snippet = "\n".join(code_snippets[:3])[:2000]
            context.similar_past_comments = (
                await self.memory.get_similar_past_comments(
                    repo_full_name="",  # Will be derived from repo_id context
                    code_snippet=combined_snippet,
                    n_results=5,
                )
            )

        # Determine emphasis areas based on author history
        context.emphasis_areas = self._determine_emphasis_areas(context)

        # Detect rules to suppress
        context.suppress_rules = await self._detect_false_positives(session, repo_id)

        logger.info(
            "learning context built",
            author=author,
            patterns=len(context.repo_patterns),
            similar_comments=len(context.similar_past_comments),
            emphasis_areas=len(context.emphasis_areas),
        )

        return context

    def _determine_emphasis_areas(self, context: LearningContext) -> list[str]:
        """Determine areas needing extra attention based on author history.

        Args:
            context: Partially built learning context.

        Returns:
            List of area descriptions to emphasize.
        """
        areas: list[str] = []

        if context.author_profile:
            profile = context.author_profile

            # If author frequently has security issues, emphasize security
            if "security" in profile.common_issues:
                areas.append(
                    "Author has had security issues in past PRs — review security carefully."
                )

            # If high average issues, be thorough
            if profile.avg_issues_per_pr > 5:
                areas.append(
                    "Author's PRs typically have many issues — thorough review recommended."
                )

            # If low approval rate, common patterns may recur
            if profile.approval_rate < 0.3 and profile.total_prs_reviewed >= self.MIN_REVIEWS_FOR_LEARNING:
                areas.append(
                    "Author's PRs are frequently sent back for changes — check for recurring patterns."
                )

        # Check repository patterns for high-frequency issues
        for pattern in context.repo_patterns:
            if pattern.frequency >= 5:
                areas.append(
                    f"Recurring issue in this repo: {pattern.description[:80]}"
                )

        return areas[:5]  # Cap at 5 emphasis areas

    async def _detect_false_positives(
        self, session: AsyncSession, repo_id: uuid.UUID
    ) -> list[str]:
        """Detect rules that consistently produce false positives.

        A rule is considered a false positive pattern if:
        - It appears in multiple reviews
        - The associated PRs were ultimately approved without changes

        Args:
            session: Database session.
            repo_id: Repository UUID.

        Returns:
            List of rule_ids to suppress.
        """
        # Find comments on PRs that were ultimately approved
        # (suggesting the comment may have been a false positive)
        query = (
            select(ReviewComment.rule_id, func.count().label("cnt"))
            .join(Review)
            .where(
                and_(
                    Review.repository_id == repo_id,
                    Review.decision == ReviewDecision.APPROVE,
                    ReviewComment.rule_id.isnot(None),
                    ReviewComment.severity.in_([
                        CommentSeverity.LOW,
                        CommentSeverity.INFO,
                    ]),
                )
            )
            .group_by(ReviewComment.rule_id)
            .having(func.count() >= 3)
        )

        result = await session.execute(query)
        rows = result.fetchall()

        suppress_rules = [row[0] for row in rows if row[0]]

        if suppress_rules:
            logger.debug(
                "false positive rules detected",
                rules=suppress_rules,
            )

        return suppress_rules

    async def record_feedback(
        self,
        session: AsyncSession,
        review_id: uuid.UUID,
        comment_id: uuid.UUID,
        was_helpful: bool,
        was_resolved: bool = False,
    ) -> None:
        """Record human feedback on a review comment.

        Used to track which comments were helpful vs noise,
        improving future reviews.

        Args:
            session: Database session.
            review_id: Parent review ID.
            comment_id: The comment receiving feedback.
            was_helpful: Whether the comment was considered useful.
            was_resolved: Whether the issue was actually fixed.
        """
        query = select(ReviewComment).where(ReviewComment.id == comment_id)
        result = await session.execute(query)
        comment = result.scalar_one_or_none()

        if comment:
            comment.is_resolved = was_resolved
            # In a full implementation, we'd store helpfulness in a
            # separate feedback table for more nuanced learning
            await session.flush()

            logger.debug(
                "feedback recorded",
                comment_id=str(comment_id),
                helpful=was_helpful,
                resolved=was_resolved,
            )
