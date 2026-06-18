"""Pull request fetcher that coordinates diff retrieval and parsing.

Orchestrates fetching PR metadata, file changes, and full file content
from GitHub, then assembles everything into a rich context object
for the review engine.
"""

import base64
from dataclasses import dataclass, field
from typing import Any

from code_review_agent.github.client import GitHubClient
from code_review_agent.github.diff_parser import (
    FileDiff,
    PullRequestDiff,
    parse_github_files,
)
from code_review_agent.logging import get_logger

logger = get_logger("pr_fetcher")


@dataclass
class PRContext:
    """Complete context for reviewing a pull request.

    Aggregates all information needed by the review engine:
    PR metadata, parsed diffs, and optionally full file contents.

    Attributes:
        owner: Repository owner.
        repo: Repository name.
        pr_number: Pull request number.
        installation_id: GitHub App installation ID.
        title: PR title.
        description: PR body/description.
        base_branch: Target branch the PR merges into.
        head_branch: Source branch with the changes.
        head_sha: Latest commit SHA on the head branch.
        author: PR author's GitHub login.
        diff: Parsed pull request diff.
        file_contents: Map of filename -> full file content for context.
        labels: PR labels.
    """

    owner: str
    repo: str
    pr_number: int
    installation_id: int
    title: str = ""
    description: str = ""
    base_branch: str = ""
    head_branch: str = ""
    head_sha: str = ""
    author: str = ""
    diff: PullRequestDiff = field(default_factory=PullRequestDiff)
    file_contents: dict[str, str] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)

    @property
    def repo_full_name(self) -> str:
        """Get the full repository name (owner/repo)."""
        return f"{self.owner}/{self.repo}"

    @property
    def changed_filenames(self) -> list[str]:
        """Get list of all changed file paths."""
        return [f.filename for f in self.diff.files]

    @property
    def reviewable_files(self) -> list[FileDiff]:
        """Get files that should be reviewed (excludes binary/generated)."""
        skip_patterns = {
            "package-lock.json",
            "yarn.lock",
            "poetry.lock",
            "Pipfile.lock",
            "pnpm-lock.yaml",
            "go.sum",
        }
        skip_extensions = {
            "png", "jpg", "jpeg", "gif", "svg", "ico",
            "woff", "woff2", "ttf", "eot",
            "min.js", "min.css",
        }

        reviewable = []
        for f in self.diff.files:
            basename = f.filename.rsplit("/", 1)[-1]
            if basename in skip_patterns:
                continue
            if f.extension in skip_extensions:
                continue
            if f.is_deleted:
                continue
            reviewable.append(f)

        return reviewable


class PRFetcher:
    """Fetches and assembles complete PR context from GitHub.

    Coordinates multiple GitHub API calls to build a rich context
    object containing everything needed for an informed code review.

    Attributes:
        client: Authenticated GitHub API client.
    """

    def __init__(self, client: GitHubClient) -> None:
        """Initialize the PR fetcher.

        Args:
            client: Configured GitHubClient instance.
        """
        self.client = client

    async def fetch_pr_context(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        installation_id: int,
        fetch_full_content: bool = True,
        max_file_size: int = 100_000,
    ) -> PRContext:
        """Fetch complete PR context for review.

        Retrieves PR metadata, file changes, and optionally the full
        content of modified files for deeper analysis.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.
            installation_id: GitHub App installation ID.
            fetch_full_content: Whether to fetch full file contents.
            max_file_size: Maximum file size (bytes) to fetch content for.

        Returns:
            Complete PRContext ready for the review engine.
        """
        logger.info(
            "fetching PR context",
            repo=f"{owner}/{repo}",
            pr_number=pr_number,
        )

        # Fetch PR metadata
        pr_data = await self.client.get_pull_request(
            owner, repo, pr_number, installation_id
        )

        # Fetch file changes
        files_data = await self.client.get_pull_request_files(
            owner, repo, pr_number, installation_id
        )

        # Parse the diff
        diff = parse_github_files(files_data)

        # Build context
        context = PRContext(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            installation_id=installation_id,
            title=pr_data.get("title", ""),
            description=pr_data.get("body", "") or "",
            base_branch=pr_data.get("base", {}).get("ref", ""),
            head_branch=pr_data.get("head", {}).get("ref", ""),
            head_sha=pr_data.get("head", {}).get("sha", ""),
            author=pr_data.get("user", {}).get("login", ""),
            diff=diff,
            labels=[label.get("name", "") for label in pr_data.get("labels", [])],
        )

        # Optionally fetch full file contents for modified files
        if fetch_full_content:
            await self._fetch_file_contents(
                context=context,
                files_data=files_data,
                max_file_size=max_file_size,
            )

        logger.info(
            "PR context assembled",
            pr_number=pr_number,
            files_changed=diff.total_files_changed,
            additions=diff.total_additions,
            deletions=diff.total_deletions,
            files_with_content=len(context.file_contents),
        )

        return context

    async def _fetch_file_contents(
        self,
        context: PRContext,
        files_data: list[dict[str, Any]],
        max_file_size: int,
    ) -> None:
        """Fetch full file contents for reviewable files.

        Retrieves the full content of each modified file at the HEAD
        commit of the PR branch, for deeper analysis beyond just the diff.

        Args:
            context: The PR context to populate with file contents.
            files_data: Raw file data from GitHub API.
            max_file_size: Maximum file size to fetch.
        """
        for file_data in files_data:
            filename = file_data.get("filename", "")
            status = file_data.get("status", "")

            # Skip deleted files and large files
            if status == "removed":
                continue

            size = file_data.get("size", 0)
            if size and size > max_file_size:
                logger.debug(
                    "skipping large file",
                    filename=filename,
                    size=size,
                )
                continue

            # Skip binary and generated files
            file_diff = context.diff.get_file(filename)
            if file_diff and file_diff.extension in {
                "png", "jpg", "jpeg", "gif", "svg", "ico",
                "woff", "woff2", "ttf", "eot", "pdf",
            }:
                continue

            try:
                content_data = await self.client.get_repository_content(
                    owner=context.owner,
                    repo=context.repo,
                    path=filename,
                    installation_id=context.installation_id,
                    ref=context.head_sha,
                )

                # Decode base64 content
                if content_data.get("encoding") == "base64":
                    raw_content = content_data.get("content", "")
                    decoded = base64.b64decode(raw_content).decode("utf-8", errors="replace")
                    context.file_contents[filename] = decoded
                else:
                    logger.debug(
                        "unsupported encoding",
                        filename=filename,
                        encoding=content_data.get("encoding"),
                    )

            except Exception as e:
                logger.warning(
                    "failed to fetch file content",
                    filename=filename,
                    error=str(e),
                )
