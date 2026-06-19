"""Auto-patch generation with unified diff output.

Generates concrete code fixes as unified diff patches that can be
directly applied to the repository or presented as GitHub suggested
changes. Uses the LLM to transform review comments with suggested
fixes into applicable patches.
"""

from dataclasses import dataclass, field
from typing import Any
import difflib

from code_review_agent.core.schemas import CommentOutput, ReviewOutput
from code_review_agent.github.pr_fetcher import PRContext
from code_review_agent.llm.base import BaseLLM, Message, MessageRole
from code_review_agent.logging import get_logger

logger = get_logger("patch_generator")


@dataclass
class Patch:
    """A single code patch that can be applied to fix an issue.

    Attributes:
        file_path: Target file for the patch.
        original_code: The original code lines being replaced.
        fixed_code: The corrected code lines.
        start_line: Starting line number in the original file.
        end_line: Ending line number in the original file.
        unified_diff: The patch in unified diff format.
        description: What this patch fixes.
        comment_index: Index of the review comment this patch addresses.
    """

    file_path: str
    original_code: str
    fixed_code: str
    start_line: int
    end_line: int
    unified_diff: str = ""
    description: str = ""
    comment_index: int | None = None

    @property
    def is_valid(self) -> bool:
        """Check if the patch has meaningful content."""
        return bool(
            self.file_path
            and self.fixed_code.strip()
            and self.original_code != self.fixed_code
        )

    @property
    def lines_added(self) -> int:
        """Number of lines added by this patch."""
        return len(self.fixed_code.splitlines())

    @property
    def lines_removed(self) -> int:
        """Number of lines removed by this patch."""
        return len(self.original_code.splitlines())

    def to_github_suggestion(self) -> str:
        """Format as a GitHub suggested change (markdown).

        Returns:
            GitHub-flavored markdown suggestion block.
        """
        return f"```suggestion\n{self.fixed_code}\n```"


@dataclass
class PatchSet:
    """Collection of patches for a pull request.

    Attributes:
        patches: All generated patches.
        pr_number: The PR these patches target.
        repo_full_name: Repository identifier.
    """

    patches: list[Patch] = field(default_factory=list)
    pr_number: int = 0
    repo_full_name: str = ""

    @property
    def valid_patches(self) -> list[Patch]:
        """Get only valid, applicable patches."""
        return [p for p in self.patches if p.is_valid]

    @property
    def total_files_patched(self) -> int:
        """Number of unique files with patches."""
        return len({p.file_path for p in self.valid_patches})

    def patches_for_file(self, file_path: str) -> list[Patch]:
        """Get patches targeting a specific file."""
        return [p for p in self.valid_patches if p.file_path == file_path]

    def to_combined_diff(self) -> str:
        """Combine all patches into a single unified diff string.

        Returns:
            Combined diff output suitable for `git apply`.
        """
        parts: list[str] = []
        for patch in self.valid_patches:
            if patch.unified_diff:
                parts.append(patch.unified_diff)
        return "\n".join(parts)


PATCH_SYSTEM_PROMPT = """\
You are a code fixing assistant. Given a code review comment identifying an issue \
and the relevant source code, generate ONLY the corrected code that fixes the issue.

Rules:
- Output ONLY the fixed code lines, nothing else
- Maintain the same indentation and style as the original
- Make the minimal change necessary to fix the issue
- Do not add unrelated changes or refactoring
- If the fix requires adding new lines, include them
- If the fix requires removing lines, omit them from output
"""


class PatchGenerator:
    """Generates code patches from review comments.

    Uses the LLM to transform review comments with suggested fixes into
    concrete, applicable code patches in unified diff format.

    Attributes:
        llm: LLM provider for generating fixes.
    """

    # Context lines to include around the target area
    CONTEXT_LINES = 3

    def __init__(self, llm: BaseLLM) -> None:
        """Initialize the patch generator.

        Args:
            llm: Configured LLM provider.
        """
        self.llm = llm

    async def generate_patches(
        self,
        review_output: ReviewOutput,
        pr_context: PRContext,
    ) -> PatchSet:
        """Generate patches for all comments that have suggested fixes.

        Args:
            review_output: The completed review with comments.
            pr_context: PR context with file contents.

        Returns:
            PatchSet with generated patches.
        """
        patch_set = PatchSet(
            pr_number=pr_context.pr_number,
            repo_full_name=pr_context.repo_full_name,
        )

        for idx, comment in enumerate(review_output.all_comments):
            # Only generate patches for comments with suggestions or fixable issues
            if not self._is_patchable(comment):
                continue

            file_content = pr_context.file_contents.get(comment.file_path)
            if not file_content:
                continue

            patch = await self._generate_single_patch(
                comment=comment,
                file_content=file_content,
                comment_index=idx,
            )

            if patch and patch.is_valid:
                patch_set.patches.append(patch)

        logger.info(
            "patches generated",
            total_comments=len(review_output.all_comments),
            patches_generated=len(patch_set.valid_patches),
            files_patched=patch_set.total_files_patched,
        )

        return patch_set

    async def _generate_single_patch(
        self,
        comment: CommentOutput,
        file_content: str,
        comment_index: int,
    ) -> Patch | None:
        """Generate a patch for a single review comment.

        Args:
            comment: The review comment with issue details.
            file_content: Full file content for context.
            comment_index: Index of the comment in the review.

        Returns:
            Generated Patch, or None if generation fails.
        """
        lines = file_content.splitlines(keepends=True)
        line_num = comment.line_number

        if not line_num or line_num > len(lines):
            return None

        # Extract the code region around the target line
        start = max(0, line_num - 1 - self.CONTEXT_LINES)
        end = min(len(lines), line_num + self.CONTEXT_LINES)
        original_region = lines[start:end]
        original_code = "".join(original_region)

        # If the comment already has a suggested fix, use it directly
        if comment.suggested_fix:
            fixed_code = comment.suggested_fix
            # Ensure proper newline at end
            if not fixed_code.endswith("\n"):
                fixed_code += "\n"
        else:
            # Ask LLM to generate the fix
            fixed_code = await self._llm_generate_fix(
                comment=comment,
                original_code=original_code,
                file_path=comment.file_path,
            )
            if not fixed_code:
                return None

        # Generate unified diff
        unified_diff = self._create_unified_diff(
            file_path=comment.file_path,
            original_lines=original_region,
            fixed_code=fixed_code,
            start_line=start + 1,
        )

        return Patch(
            file_path=comment.file_path,
            original_code=original_code,
            fixed_code=fixed_code,
            start_line=start + 1,
            end_line=end,
            unified_diff=unified_diff,
            description=comment.message,
            comment_index=comment_index,
        )

    async def _llm_generate_fix(
        self,
        comment: CommentOutput,
        original_code: str,
        file_path: str,
    ) -> str | None:
        """Use the LLM to generate a fix for the identified issue.

        Args:
            comment: Review comment describing the issue.
            original_code: The original code that needs fixing.
            file_path: File path for context.

        Returns:
            The fixed code string, or None on failure.
        """
        user_prompt = (
            f"File: {file_path}\n"
            f"Issue ({comment.severity}, {comment.category}): {comment.message}\n\n"
            f"Original code:\n```\n{original_code}\n```\n\n"
            f"Generate the corrected code that fixes this issue. "
            f"Output ONLY the fixed code, no explanation."
        )

        try:
            response = await self.llm.generate(
                messages=[
                    Message(role=MessageRole.SYSTEM, content=PATCH_SYSTEM_PROMPT),
                    Message(role=MessageRole.USER, content=user_prompt),
                ],
                temperature=0.0,
                max_tokens=1024,
            )

            fixed_code = response.content.strip()

            # Strip markdown code fences if present
            if fixed_code.startswith("```"):
                lines = fixed_code.split("\n")
                fixed_code = "\n".join(lines[1:-1]) if len(lines) > 2 else fixed_code

            return fixed_code if fixed_code else None

        except Exception as e:
            logger.warning("LLM fix generation failed", error=str(e))
            return None

    @staticmethod
    def _create_unified_diff(
        file_path: str,
        original_lines: list[str],
        fixed_code: str,
        start_line: int,
    ) -> str:
        """Create a unified diff from original and fixed code.

        Args:
            file_path: File path for diff headers.
            original_lines: Original source lines.
            fixed_code: The replacement code.
            start_line: Line number where the region starts.

        Returns:
            Unified diff string.
        """
        # Ensure consistent line endings
        original = [line if line.endswith("\n") else line + "\n" for line in original_lines]
        fixed = fixed_code.splitlines(keepends=True)
        if fixed and not fixed[-1].endswith("\n"):
            fixed[-1] += "\n"

        diff = difflib.unified_diff(
            original,
            fixed,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="\n",
        )

        return "".join(diff)

    @staticmethod
    def _is_patchable(comment: CommentOutput) -> bool:
        """Determine if a comment is suitable for patch generation.

        Only generates patches for actionable issues with clear fixes.

        Args:
            comment: The review comment.

        Returns:
            True if a patch should be attempted.
        """
        # Always patch if there's already a suggested fix
        if comment.suggested_fix:
            return True

        # Only auto-generate patches for certain categories/severities
        patchable_categories = {
            "security", "bug", "performance", "style", "naming",
        }
        patchable_severities = {"critical", "high", "medium"}

        return (
            comment.category in patchable_categories
            and comment.severity in patchable_severities
            and comment.line_number is not None
        )
