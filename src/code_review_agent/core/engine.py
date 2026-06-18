"""Core review engine that orchestrates the full review pipeline.

Coordinates the entire review process: fetching context, retrieving
standards, running analysis, generating LLM review, and producing
the final structured output.
"""

import json
import time
from typing import Any

from code_review_agent.core.prompts import PromptBuilder
from code_review_agent.core.schemas import (
    REVIEW_OUTPUT_SCHEMA,
    ReviewOutput,
    FileReviewOutput,
    CommentOutput,
)
from code_review_agent.github.pr_fetcher import PRContext
from code_review_agent.llm.base import BaseLLM, LLMResponse
from code_review_agent.rag.retriever import ContextRetriever, RetrievalContext
from code_review_agent.logging import get_logger

logger = get_logger("review_engine")

# Threshold for switching to per-file review mode
LARGE_PR_FILE_THRESHOLD = 10
LARGE_PR_LINE_THRESHOLD = 1000


class ReviewEngine:
    """Orchestrates the AI code review pipeline.

    Manages the full lifecycle of a review:
    1. Retrieve relevant context from the vector store
    2. Build structured prompts with diff and standards
    3. Call LLM for review generation
    4. Parse and validate structured output
    5. Apply decision logic

    For large PRs, splits into per-file reviews and merges results.

    Attributes:
        llm: The LLM provider for generating reviews.
        retriever: Context retriever for RAG lookups.
        prompt_builder: Prompt construction utility.
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: ContextRetriever | None = None,
    ) -> None:
        """Initialize the review engine.

        Args:
            llm: Configured LLM provider.
            retriever: Optional context retriever for RAG.
        """
        self.llm = llm
        self.retriever = retriever
        self.prompt_builder = PromptBuilder()

    async def review(self, pr_context: PRContext) -> ReviewOutput:
        """Execute a full code review on a pull request.

        Decides between whole-PR and per-file review strategies based
        on the PR size, then orchestrates the review pipeline.

        Args:
            pr_context: Complete PR context with diff and metadata.

        Returns:
            Structured ReviewOutput with all findings and decision.
        """
        start_time = time.time()

        logger.info(
            "starting review",
            repo=pr_context.repo_full_name,
            pr=pr_context.pr_number,
            files=pr_context.diff.total_files_changed,
            additions=pr_context.diff.total_additions,
        )

        # Retrieve RAG context
        retrieval_context = await self._retrieve_context(pr_context)

        # Choose review strategy based on PR size
        reviewable_files = pr_context.reviewable_files
        is_large_pr = (
            len(reviewable_files) > LARGE_PR_FILE_THRESHOLD
            or pr_context.diff.total_additions > LARGE_PR_LINE_THRESHOLD
        )

        if is_large_pr:
            logger.info("using per-file review strategy", files=len(reviewable_files))
            output = await self._review_per_file(pr_context, retrieval_context)
        else:
            logger.info("using whole-PR review strategy")
            output = await self._review_whole_pr(pr_context, retrieval_context)

        duration = time.time() - start_time
        logger.info(
            "review complete",
            pr=pr_context.pr_number,
            decision=output.decision,
            comments=output.total_comments,
            critical=output.critical_count,
            high=output.high_count,
            duration_s=round(duration, 2),
        )

        return output

    async def _review_whole_pr(
        self,
        pr_context: PRContext,
        retrieval_context: RetrievalContext | None,
    ) -> ReviewOutput:
        """Review the entire PR in a single LLM call.

        Args:
            pr_context: PR context.
            retrieval_context: RAG context.

        Returns:
            Parsed ReviewOutput.
        """
        messages = self.prompt_builder.build_review_messages(
            pr_context, retrieval_context
        )

        response = await self.llm.generate_structured(
            messages=messages,
            response_schema=REVIEW_OUTPUT_SCHEMA,
        )

        output = self._parse_response(response)
        return self._apply_decision_logic(output)

    async def _review_per_file(
        self,
        pr_context: PRContext,
        retrieval_context: RetrievalContext | None,
    ) -> ReviewOutput:
        """Review each file individually and merge results.

        Used for large PRs where the full diff exceeds context limits.

        Args:
            pr_context: PR context.
            retrieval_context: RAG context.

        Returns:
            Merged ReviewOutput from all file reviews.
        """
        all_file_reviews: list[FileReviewOutput] = []

        for file_diff in pr_context.reviewable_files:
            try:
                messages = self.prompt_builder.build_file_review_messages(
                    file_diff=file_diff,
                    pr_context=pr_context,
                    retrieval_context=retrieval_context,
                )

                response = await self.llm.generate_structured(
                    messages=messages,
                    response_schema=REVIEW_OUTPUT_SCHEMA,
                )

                file_output = self._parse_response(response)
                all_file_reviews.extend(file_output.file_reviews)

            except Exception as e:
                logger.warning(
                    "file review failed",
                    file=file_diff.filename,
                    error=str(e),
                )
                continue

        # Merge into a single output
        merged = ReviewOutput(file_reviews=all_file_reviews)
        merged = self._generate_summary(merged, pr_context)
        return self._apply_decision_logic(merged)

    async def _retrieve_context(
        self, pr_context: PRContext
    ) -> RetrievalContext | None:
        """Retrieve relevant context from the vector store.

        Args:
            pr_context: PR context to generate queries from.

        Returns:
            Retrieved context, or None if retriever is unavailable.
        """
        if not self.retriever:
            return None

        try:
            # Use changed code as query material
            query_texts: list[str] = []
            for file_diff in pr_context.reviewable_files[:5]:
                # Use added lines as query
                for hunk in file_diff.hunks:
                    added_content = "\n".join(
                        line.content for line in hunk.added_lines
                    )
                    if added_content.strip():
                        query_texts.append(added_content[:500])

            if not query_texts:
                return None

            return await self.retriever.retrieve_context(
                repo_full_name=pr_context.repo_full_name,
                query_texts=query_texts,
            )

        except Exception as e:
            logger.warning("context retrieval failed", error=str(e))
            return None

    def _parse_response(self, response: LLMResponse) -> ReviewOutput:
        """Parse the LLM JSON response into a ReviewOutput.

        Handles malformed JSON gracefully by attempting cleanup.

        Args:
            response: Raw LLM response.

        Returns:
            Parsed ReviewOutput.
        """
        content = response.content.strip()

        # Strip potential markdown code fencing
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            content = content.strip()

        try:
            data = json.loads(content)
            return ReviewOutput.from_dict(data)
        except json.JSONDecodeError as e:
            logger.warning(
                "failed to parse LLM JSON response",
                error=str(e),
                content_preview=content[:200],
            )
            # Return a minimal output with the raw content as summary
            return ReviewOutput(
                summary=f"Review generated but output parsing failed: {content[:500]}",
                decision="comment",
                confidence=0.3,
            )

    def _apply_decision_logic(self, output: ReviewOutput) -> ReviewOutput:
        """Apply deterministic decision logic based on findings.

        Overrides the LLM's decision when the evidence clearly
        supports a different conclusion.

        Args:
            output: The raw ReviewOutput from LLM.

        Returns:
            ReviewOutput with adjusted decision if needed.
        """
        if output.critical_count > 0:
            output.decision = "request_changes"
            output.confidence = max(output.confidence, 0.9)
        elif output.high_count >= 3:
            output.decision = "request_changes"
            output.confidence = max(output.confidence, 0.8)
        elif output.total_comments == 0:
            output.decision = "approve"
            output.confidence = max(output.confidence, 0.8)
        elif output.critical_count == 0 and output.high_count == 0:
            # Only low/medium issues — approve with comments
            if output.decision == "request_changes":
                output.decision = "comment"

        return output

    def _generate_summary(
        self, output: ReviewOutput, pr_context: PRContext
    ) -> ReviewOutput:
        """Generate a summary for merged per-file reviews.

        Args:
            output: Merged review output without summary.
            pr_context: PR context for metadata.

        Returns:
            Output with generated summary.
        """
        total = output.total_comments
        critical = output.critical_count
        high = output.high_count

        if total == 0:
            output.summary = (
                f"Reviewed {len(pr_context.reviewable_files)} files with "
                f"+{pr_context.diff.total_additions}/-{pr_context.diff.total_deletions} "
                f"changes. No issues found."
            )
        else:
            output.summary = (
                f"Reviewed {len(pr_context.reviewable_files)} files. "
                f"Found {total} issue(s): "
                f"{critical} critical, {high} high, "
                f"{total - critical - high} medium/low."
            )

        return output
