"""High-level GitHub API client for pull request operations.

Wraps httpx with authenticated requests, retry logic, and
rate-limit awareness for interacting with the GitHub REST API.
"""

from typing import Any

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from code_review_agent.github.auth import GitHubAppAuth
from code_review_agent.logging import get_logger

logger = get_logger("github_client")

# Retry configuration
MAX_RETRIES = 3
RETRY_MIN_WAIT = 1  # seconds
RETRY_MAX_WAIT = 10  # seconds


class GitHubClient:
    """Authenticated GitHub API client for code review operations.

    Provides methods for PR interactions: fetching PRs, posting comments,
    creating reviews, and managing review state. Handles authentication
    token refresh and rate limiting automatically.

    Attributes:
        auth: The GitHub App authentication handler.
        api_url: The GitHub API base URL.
    """

    def __init__(self, auth: GitHubAppAuth) -> None:
        """Initialize the GitHub client.

        Args:
            auth: Configured GitHubAppAuth instance for token management.
        """
        self.auth = auth
        self.api_url = auth.api_url

    async def _get_headers(self, installation_id: int) -> dict[str, str]:
        """Get authenticated request headers.

        Args:
            installation_id: The installation to authenticate as.

        Returns:
            Headers dict with authorization and API version.
        """
        token = await self.auth.get_installation_token(installation_id)
        return {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    )
    async def _request(
        self,
        method: str,
        url: str,
        installation_id: int,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an authenticated request to the GitHub API with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, etc.).
            url: Full API URL.
            installation_id: Installation ID for authentication.
            **kwargs: Additional arguments passed to httpx.

        Returns:
            The HTTP response.

        Raises:
            httpx.HTTPStatusError: If the response indicates an error after retries.
        """
        headers = await self._get_headers(installation_id)
        if "headers" in kwargs:
            kwargs["headers"].update(headers)
        else:
            kwargs["headers"] = headers

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(method, url, **kwargs)

        # Log rate limit info
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining and int(remaining) < 100:
            logger.warning(
                "github rate limit low",
                remaining=remaining,
                limit=response.headers.get("X-RateLimit-Limit"),
            )

        response.raise_for_status()
        return response

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        installation_id: int,
    ) -> dict[str, Any]:
        """Fetch pull request details.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.
            installation_id: GitHub App installation ID.

        Returns:
            Pull request data as a dictionary.
        """
        url = f"{self.api_url}/repos/{owner}/{repo}/pulls/{pr_number}"
        response = await self._request("GET", url, installation_id)
        return response.json()

    async def get_pull_request_files(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        installation_id: int,
    ) -> list[dict[str, Any]]:
        """Fetch the list of files changed in a pull request.

        Handles pagination to retrieve all changed files.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.
            installation_id: GitHub App installation ID.

        Returns:
            List of file change objects with patch data.
        """
        files: list[dict[str, Any]] = []
        page = 1
        per_page = 100

        while True:
            url = (
                f"{self.api_url}/repos/{owner}/{repo}/pulls/{pr_number}/files"
                f"?page={page}&per_page={per_page}"
            )
            response = await self._request("GET", url, installation_id)
            page_files = response.json()

            if not page_files:
                break

            files.extend(page_files)

            if len(page_files) < per_page:
                break

            page += 1

        logger.info("fetched PR files", pr_number=pr_number, file_count=len(files))
        return files

    async def create_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        installation_id: int,
        body: str,
        event: str = "COMMENT",
        comments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a pull request review with optional inline comments.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.
            installation_id: GitHub App installation ID.
            body: The review summary body text.
            event: Review action - APPROVE, REQUEST_CHANGES, or COMMENT.
            comments: Optional list of inline review comments with position data.

        Returns:
            The created review response data.
        """
        url = f"{self.api_url}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        payload: dict[str, Any] = {
            "body": body,
            "event": event,
        }
        if comments:
            payload["comments"] = comments

        response = await self._request(
            "POST", url, installation_id, json=payload
        )

        logger.info(
            "review created",
            pr_number=pr_number,
            event=event,
            comment_count=len(comments) if comments else 0,
        )
        return response.json()

    async def post_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        installation_id: int,
        body: str,
    ) -> dict[str, Any]:
        """Post a general comment on a pull request.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.
            installation_id: GitHub App installation ID.
            body: The comment body text (Markdown supported).

        Returns:
            The created comment response data.
        """
        url = f"{self.api_url}/repos/{owner}/{repo}/issues/{pr_number}/comments"
        response = await self._request(
            "POST", url, installation_id, json={"body": body}
        )

        logger.info("comment posted", pr_number=pr_number)
        return response.json()

    async def get_repository_content(
        self,
        owner: str,
        repo: str,
        path: str,
        installation_id: int,
        ref: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a file's content from the repository.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: File path within the repository.
            installation_id: GitHub App installation ID.
            ref: Optional git ref (branch, tag, or SHA).

        Returns:
            File content response data (base64 encoded content).
        """
        url = f"{self.api_url}/repos/{owner}/{repo}/contents/{path}"
        params = {}
        if ref:
            params["ref"] = ref

        response = await self._request(
            "GET", url, installation_id, params=params
        )
        return response.json()

    async def get_repository_tree(
        self,
        owner: str,
        repo: str,
        tree_sha: str,
        installation_id: int,
        recursive: bool = True,
    ) -> dict[str, Any]:
        """Fetch the repository file tree.

        Args:
            owner: Repository owner.
            repo: Repository name.
            tree_sha: The SHA of the tree to fetch (use branch HEAD).
            installation_id: GitHub App installation ID.
            recursive: Whether to fetch the tree recursively.

        Returns:
            Tree response with file paths and metadata.
        """
        url = f"{self.api_url}/repos/{owner}/{repo}/git/trees/{tree_sha}"
        params = {"recursive": "1"} if recursive else {}

        response = await self._request(
            "GET", url, installation_id, params=params
        )
        return response.json()
