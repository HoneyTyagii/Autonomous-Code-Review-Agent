"""Unified diff parser for GitHub pull request patches.

Parses GitHub's unified diff format into structured objects that
the review engine can reason about: hunks, added/removed lines,
and position mapping for inline comments.
"""

from dataclasses import dataclass, field
from enum import Enum
import re

from code_review_agent.logging import get_logger

logger = get_logger("diff_parser")

# Regex for unified diff hunk headers: @@ -old_start,old_count +new_start,new_count @@
HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)


class LineType(str, Enum):
    """Type of line in a diff."""

    ADDED = "added"
    REMOVED = "removed"
    CONTEXT = "context"
    HUNK_HEADER = "hunk_header"


@dataclass
class DiffLine:
    """A single line in a diff hunk.

    Attributes:
        content: The line content (without the +/- prefix).
        line_type: Whether the line was added, removed, or is context.
        old_line_number: Line number in the old file (None for added lines).
        new_line_number: Line number in the new file (None for removed lines).
        diff_position: Position in the diff for GitHub inline comments.
    """

    content: str
    line_type: LineType
    old_line_number: int | None = None
    new_line_number: int | None = None
    diff_position: int = 0


@dataclass
class DiffHunk:
    """A hunk within a file diff.

    Represents a contiguous block of changes with surrounding context.

    Attributes:
        old_start: Starting line number in the old file.
        old_count: Number of lines from the old file.
        new_start: Starting line number in the new file.
        new_count: Number of lines in the new file.
        header_text: Any function/class context after the @@ markers.
        lines: The individual diff lines within this hunk.
    """

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header_text: str = ""
    lines: list[DiffLine] = field(default_factory=list)

    @property
    def added_lines(self) -> list[DiffLine]:
        """Get only the added lines in this hunk."""
        return [line for line in self.lines if line.line_type == LineType.ADDED]

    @property
    def removed_lines(self) -> list[DiffLine]:
        """Get only the removed lines in this hunk."""
        return [line for line in self.lines if line.line_type == LineType.REMOVED]

    @property
    def changed_line_numbers(self) -> list[int]:
        """Get new file line numbers where changes occurred."""
        return [
            line.new_line_number
            for line in self.lines
            if line.line_type == LineType.ADDED and line.new_line_number is not None
        ]


@dataclass
class FileDiff:
    """Parsed diff for a single file in a pull request.

    Attributes:
        filename: Path of the file relative to repository root.
        status: Change status (added, removed, modified, renamed).
        additions: Total number of added lines.
        deletions: Total number of removed lines.
        patch: Raw unified diff patch text.
        hunks: Parsed diff hunks.
        previous_filename: Original filename if file was renamed.
        language: Detected programming language based on extension.
    """

    filename: str
    status: str
    additions: int = 0
    deletions: int = 0
    patch: str = ""
    hunks: list[DiffHunk] = field(default_factory=list)
    previous_filename: str | None = None
    language: str | None = None

    @property
    def is_new_file(self) -> bool:
        """Check if this file was newly created."""
        return self.status == "added"

    @property
    def is_deleted(self) -> bool:
        """Check if this file was deleted."""
        return self.status == "removed"

    @property
    def is_renamed(self) -> bool:
        """Check if this file was renamed."""
        return self.status == "renamed"

    @property
    def extension(self) -> str:
        """Get the file extension."""
        parts = self.filename.rsplit(".", 1)
        return parts[1] if len(parts) > 1 else ""

    @property
    def total_changes(self) -> int:
        """Total number of changed lines."""
        return self.additions + self.deletions

    def get_line_at_position(self, position: int) -> DiffLine | None:
        """Get the diff line at a specific GitHub diff position.

        Args:
            position: The position number in the diff (1-indexed).

        Returns:
            The DiffLine at that position, or None if not found.
        """
        for hunk in self.hunks:
            for line in hunk.lines:
                if line.diff_position == position:
                    return line
        return None


@dataclass
class PullRequestDiff:
    """Complete parsed diff for an entire pull request.

    Attributes:
        files: List of all file diffs in the PR.
        total_additions: Sum of all additions across files.
        total_deletions: Sum of all deletions across files.
    """

    files: list[FileDiff] = field(default_factory=list)

    @property
    def total_additions(self) -> int:
        """Total lines added across all files."""
        return sum(f.additions for f in self.files)

    @property
    def total_deletions(self) -> int:
        """Total lines removed across all files."""
        return sum(f.deletions for f in self.files)

    @property
    def total_files_changed(self) -> int:
        """Number of files changed."""
        return len(self.files)

    @property
    def modified_files(self) -> list[FileDiff]:
        """Files that were modified (not added or deleted)."""
        return [f for f in self.files if f.status == "modified"]

    @property
    def added_files(self) -> list[FileDiff]:
        """Files that were newly added."""
        return [f for f in self.files if f.status == "added"]

    @property
    def deleted_files(self) -> list[FileDiff]:
        """Files that were deleted."""
        return [f for f in self.files if f.status == "removed"]

    def get_file(self, filename: str) -> FileDiff | None:
        """Get a specific file diff by filename.

        Args:
            filename: The file path to look up.

        Returns:
            The FileDiff if found, None otherwise.
        """
        for f in self.files:
            if f.filename == filename:
                return f
        return None

    def filter_by_language(self, language: str) -> list[FileDiff]:
        """Get files filtered by programming language.

        Args:
            language: The language to filter by.

        Returns:
            List of FileDiff objects matching the language.
        """
        return [f for f in self.files if f.language == language]


# Language detection by file extension
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "jsx": "javascript",
    "java": "java",
    "go": "go",
    "rs": "rust",
    "rb": "ruby",
    "php": "php",
    "c": "c",
    "cpp": "cpp",
    "h": "c",
    "hpp": "cpp",
    "cs": "csharp",
    "swift": "swift",
    "kt": "kotlin",
    "scala": "scala",
    "yml": "yaml",
    "yaml": "yaml",
    "json": "json",
    "toml": "toml",
    "md": "markdown",
    "sh": "bash",
    "bash": "bash",
    "sql": "sql",
    "dockerfile": "docker",
}


def detect_language(filename: str) -> str | None:
    """Detect programming language from filename.

    Args:
        filename: The file path to detect language for.

    Returns:
        Language string or None if unrecognized.
    """
    # Check for special filenames
    basename = filename.rsplit("/", 1)[-1].lower()
    if basename == "dockerfile":
        return "docker"
    if basename == "makefile":
        return "make"

    # Check extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return EXTENSION_LANGUAGE_MAP.get(ext)


def parse_patch(patch: str) -> list[DiffHunk]:
    """Parse a unified diff patch string into structured hunks.

    Handles GitHub's unified diff format with proper line number tracking
    and position mapping for inline review comments.

    Args:
        patch: The raw unified diff patch text from GitHub API.

    Returns:
        List of parsed DiffHunk objects.
    """
    if not patch:
        return []

    hunks: list[DiffHunk] = []
    current_hunk: DiffHunk | None = None
    position = 0  # GitHub's diff position counter (1-indexed across all hunks)
    old_line = 0
    new_line = 0

    for raw_line in patch.split("\n"):
        # Check for hunk header
        match = HUNK_HEADER_RE.match(raw_line)
        if match:
            position += 1
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) else 1
            new_start = int(match.group(3))
            new_count = int(match.group(4)) if match.group(4) else 1
            header_text = match.group(5).strip()

            current_hunk = DiffHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                header_text=header_text,
            )
            hunks.append(current_hunk)
            old_line = old_start
            new_line = new_start
            continue

        if current_hunk is None:
            continue

        position += 1

        if raw_line.startswith("+"):
            # Added line
            diff_line = DiffLine(
                content=raw_line[1:],
                line_type=LineType.ADDED,
                old_line_number=None,
                new_line_number=new_line,
                diff_position=position,
            )
            current_hunk.lines.append(diff_line)
            new_line += 1

        elif raw_line.startswith("-"):
            # Removed line
            diff_line = DiffLine(
                content=raw_line[1:],
                line_type=LineType.REMOVED,
                old_line_number=old_line,
                new_line_number=None,
                diff_position=position,
            )
            current_hunk.lines.append(diff_line)
            old_line += 1

        elif raw_line.startswith(" ") or raw_line == "":
            # Context line
            content = raw_line[1:] if raw_line.startswith(" ") else raw_line
            diff_line = DiffLine(
                content=content,
                line_type=LineType.CONTEXT,
                old_line_number=old_line,
                new_line_number=new_line,
                diff_position=position,
            )
            current_hunk.lines.append(diff_line)
            old_line += 1
            new_line += 1

        elif raw_line.startswith("\\"):
            # "\ No newline at end of file" - skip but count position
            pass

    return hunks


def parse_github_files(files_data: list[dict]) -> PullRequestDiff:
    """Parse GitHub API file change data into a structured PullRequestDiff.

    Takes the response from GitHub's "List pull request files" endpoint
    and parses each file's patch into structured diff objects.

    Args:
        files_data: List of file objects from GitHub API
                    (GET /repos/{owner}/{repo}/pulls/{pr}/files).

    Returns:
        A fully parsed PullRequestDiff with all files and hunks.
    """
    pr_diff = PullRequestDiff()

    for file_data in files_data:
        filename = file_data.get("filename", "")
        patch = file_data.get("patch", "")

        file_diff = FileDiff(
            filename=filename,
            status=file_data.get("status", "modified"),
            additions=file_data.get("additions", 0),
            deletions=file_data.get("deletions", 0),
            patch=patch,
            hunks=parse_patch(patch),
            previous_filename=file_data.get("previous_filename"),
            language=detect_language(filename),
        )

        pr_diff.files.append(file_diff)

    logger.info(
        "parsed PR diff",
        files=len(pr_diff.files),
        additions=pr_diff.total_additions,
        deletions=pr_diff.total_deletions,
    )

    return pr_diff
