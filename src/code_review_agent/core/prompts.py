"""Prompt construction for the code review LLM chain.

Builds structured prompts that provide the LLM with full context:
PR metadata, diff content, coding standards, related code, and
explicit instructions for producing actionable review comments.
"""

from code_review_agent.github.diff_parser import FileDiff, PullRequestDiff
from code_review_agent.github.pr_fetcher import PRContext
from code_review_agent.llm.base import Message, MessageRole
from code_review_agent.rag.retriever import RetrievalContext
from code_review_agent.logging import get_logger

logger = get_logger("prompts")

SYSTEM_PROMPT = """\
You are an expert code reviewer for a software engineering team. Your job is to \
review pull request changes and provide constructive, actionable feedback.

Your review should:
1. Identify bugs, security vulnerabilities, and logic errors
2. Check adherence to coding standards and best practices
3. Evaluate code maintainability and readability
4. Suggest performance improvements where relevant
5. Note missing error handling or edge cases
6. Identify opportunities for better naming or documentation

Guidelines:
- Be specific: reference exact file paths and line numbers
- Be constructive: explain WHY something is an issue, not just WHAT
- Prioritize: focus on impactful issues, not nitpicks
- Suggest fixes: provide concrete code suggestions when possible
- Be concise: keep comments focused and actionable
- Avoid false positives: only flag genuine concerns

Severity levels:
- critical: Security vulnerabilities, data loss, crashes in production
- high: Bugs, significant logic errors, major design problems
- medium: Code quality, maintainability, missing error handling
- low: Style, minor readability, optional improvements
- info: Observations, positive feedback, informational notes

Decision criteria:
- approve: No critical/high issues, code is ready to merge
- request_changes: Critical or multiple high-severity issues found
- comment: Only medium/low issues, or uncertain about severity
"""


class PromptBuilder:
    """Constructs prompts for the review LLM from PR context and RAG results.

    Assembles system instructions, PR metadata, diff content, coding standards,
    and related code into a structured message sequence for the LLM.
    """

    # Maximum diff size before truncation (characters)
    MAX_DIFF_SIZE = 50_000
    MAX_FILE_CONTENT_SIZE = 20_000

    def build_review_messages(
        self,
        pr_context: PRContext,
        retrieval_context: RetrievalContext | None = None,
    ) -> list[Message]:
        """Build the full message sequence for a code review.

        Args:
            pr_context: Complete PR context with diff and metadata.
            retrieval_context: Retrieved RAG context (standards, related code).

        Returns:
            List of Messages ready for the LLM.
        """
        messages: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ]

        # Build user message with all context
        user_content = self._build_user_message(pr_context, retrieval_context)
        messages.append(Message(role=MessageRole.USER, content=user_content))

        logger.debug(
            "prompt built",
            messages=len(messages),
            total_chars=sum(len(m.content) for m in messages),
        )

        return messages

    def build_file_review_messages(
        self,
        file_diff: FileDiff,
        pr_context: PRContext,
        retrieval_context: RetrievalContext | None = None,
    ) -> list[Message]:
        """Build messages for reviewing a single file in detail.

        Used for large PRs where reviewing file-by-file is more effective.

        Args:
            file_diff: The specific file's diff to review.
            pr_context: Overall PR context for metadata.
            retrieval_context: Retrieved context for this file.

        Returns:
            List of Messages for single-file review.
        """
        messages: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ]

        parts: list[str] = []

        # PR context header
        parts.append(self._format_pr_header(pr_context))

        # File diff
        parts.append(f"\n## File Under Review: {file_diff.filename}")
        parts.append(f"Status: {file_diff.status} | "
                     f"+{file_diff.additions} -{file_diff.deletions}")
        parts.append(f"\n### Diff:\n```\n{file_diff.patch[:self.MAX_DIFF_SIZE]}\n```")

        # Full file content if available
        full_content = pr_context.file_contents.get(file_diff.filename)
        if full_content:
            truncated = full_content[:self.MAX_FILE_CONTENT_SIZE]
            parts.append(f"\n### Full File Content:\n```\n{truncated}\n```")

        # Standards
        if retrieval_context and retrieval_context.coding_standards:
            parts.append(
                f"\n## Applicable Coding Standards:\n"
                f"{retrieval_context.format_standards()}"
            )

        parts.append(
            "\n## Instructions:\n"
            "Review ONLY this file. Produce a JSON response with your findings."
        )

        messages.append(Message(role=MessageRole.USER, content="\n".join(parts)))
        return messages

    def _build_user_message(
        self,
        pr_context: PRContext,
        retrieval_context: RetrievalContext | None,
    ) -> str:
        """Assemble the full user message with all review context.

        Args:
            pr_context: PR context.
            retrieval_context: RAG context.

        Returns:
            Formatted user message string.
        """
        parts: list[str] = []

        # PR metadata header
        parts.append(self._format_pr_header(pr_context))

        # Coding standards from RAG
        if retrieval_context and retrieval_context.coding_standards:
            parts.append(
                f"\n## Coding Standards to Enforce:\n"
                f"{retrieval_context.format_standards()}"
            )

        # Related code context from RAG
        if retrieval_context and retrieval_context.related_code:
            parts.append(
                f"\n## Related Code Context (from same repository):\n"
                f"{retrieval_context.format_related_code()}"
            )

        # Diff content
        parts.append("\n## Pull Request Changes:")
        diff_text = self._format_diff(pr_context)
        parts.append(diff_text)

        # Final instruction
        parts.append(
            "\n## Instructions:\n"
            "Review all the changes above. For each issue found, specify the exact "
            "file path, line number, severity, category, and a clear message explaining "
            "the problem and how to fix it. If the code looks good, approve it.\n\n"
            "Respond with a JSON object matching the required schema."
        )

        return "\n".join(parts)

    def _format_pr_header(self, pr_context: PRContext) -> str:
        """Format PR metadata as a header section.

        Args:
            pr_context: PR context object.

        Returns:
            Formatted header string.
        """
        header = (
            f"## Pull Request: {pr_context.repo_full_name}#{pr_context.pr_number}\n"
            f"**Title:** {pr_context.title}\n"
            f"**Author:** {pr_context.author}\n"
            f"**Branch:** {pr_context.head_branch} → {pr_context.base_branch}\n"
            f"**Files Changed:** {pr_context.diff.total_files_changed}\n"
            f"**Lines:** +{pr_context.diff.total_additions} "
            f"-{pr_context.diff.total_deletions}\n"
        )

        if pr_context.description:
            desc = pr_context.description[:2000]
            header += f"\n**Description:**\n{desc}\n"

        if pr_context.labels:
            header += f"**Labels:** {', '.join(pr_context.labels)}\n"

        return header

    def _format_diff(self, pr_context: PRContext) -> str:
        """Format the PR diff for inclusion in the prompt.

        Includes only reviewable files and truncates if too large.

        Args:
            pr_context: PR context with diff.

        Returns:
            Formatted diff string.
        """
        parts: list[str] = []
        total_size = 0

        for file_diff in pr_context.reviewable_files:
            file_section = (
                f"\n### {file_diff.filename} "
                f"({file_diff.status}, +{file_diff.additions} -{file_diff.deletions})\n"
                f"```diff\n{file_diff.patch}\n```\n"
            )

            # Check if adding this file would exceed the limit
            if total_size + len(file_section) > self.MAX_DIFF_SIZE:
                remaining = len(pr_context.reviewable_files) - len(parts)
                parts.append(
                    f"\n... ({remaining} more files truncated due to size) ..."
                )
                break

            parts.append(file_section)
            total_size += len(file_section)

        return "\n".join(parts)
