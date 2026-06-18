"""GitHub App authentication using JWT and installation access tokens.

Implements the two-step GitHub App authentication flow:
1. Generate a JWT signed with the App's private key
2. Exchange the JWT for a short-lived installation access token

Reference: https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app
"""

import time
from pathlib import Path

import jwt
import httpx

from code_review_agent.logging import get_logger

logger = get_logger("github_auth")


class GitHubAppAuth:
    """Handles GitHub App authentication via JWT and installation tokens.

    Generates JWTs signed with the App's RSA private key and exchanges
    them for short-lived installation access tokens scoped to specific
    repository installations.

    Attributes:
        app_id: The GitHub App's numeric ID.
        private_key: The PEM-encoded RSA private key.
        api_url: The GitHub API base URL.
    """

    # JWT expires after 10 minutes (GitHub maximum)
    JWT_EXPIRATION_SECONDS = 600

    # Buffer before token expiry to trigger refresh (60 seconds)
    TOKEN_REFRESH_BUFFER = 60

    def __init__(self, app_id: str, private_key_path: Path, api_url: str) -> None:
        """Initialize GitHub App authentication.

        Args:
            app_id: The GitHub App ID from app settings.
            private_key_path: Path to the PEM-encoded RSA private key file.
            api_url: GitHub API base URL (e.g., https://api.github.com).

        Raises:
            FileNotFoundError: If the private key file does not exist.
            ValueError: If the private key file is empty or unreadable.
        """
        self.app_id = app_id
        self.api_url = api_url.rstrip("/")

        # Load private key
        if not private_key_path.exists():
            raise FileNotFoundError(
                f"GitHub App private key not found: {private_key_path}"
            )

        key_content = private_key_path.read_text().strip()
        if not key_content:
            raise ValueError(f"GitHub App private key is empty: {private_key_path}")

        self.private_key = key_content

        # Cache for installation tokens: {installation_id: (token, expires_at)}
        self._token_cache: dict[int, tuple[str, float]] = {}

        logger.info("github app auth initialized", app_id=app_id)

    def generate_jwt(self) -> str:
        """Generate a JWT for authenticating as the GitHub App.

        Creates a short-lived JWT (max 10 minutes) signed with the App's
        private key. Used to request installation access tokens.

        Returns:
            Encoded JWT string.
        """
        now = int(time.time())
        payload = {
            "iat": now - 60,  # Issued at (60s in the past for clock drift)
            "exp": now + self.JWT_EXPIRATION_SECONDS,
            "iss": self.app_id,
        }

        token = jwt.encode(payload, self.private_key, algorithm="RS256")
        logger.debug("jwt generated", app_id=self.app_id)
        return token

    async def get_installation_token(self, installation_id: int) -> str:
        """Get an installation access token, using cache when possible.

        Checks the cache first and only requests a new token from GitHub
        if the cached one is expired or about to expire.

        Args:
            installation_id: The GitHub App installation ID for the target repo.

        Returns:
            A valid installation access token string.

        Raises:
            GitHubAuthError: If token generation fails.
        """
        # Check cache
        if installation_id in self._token_cache:
            token, expires_at = self._token_cache[installation_id]
            if time.time() < (expires_at - self.TOKEN_REFRESH_BUFFER):
                logger.debug(
                    "using cached installation token",
                    installation_id=installation_id,
                )
                return token

        # Request new token
        token = await self._request_installation_token(installation_id)
        return token

    async def _request_installation_token(self, installation_id: int) -> str:
        """Request a new installation access token from GitHub.

        Args:
            installation_id: The GitHub App installation ID.

        Returns:
            The new installation access token.

        Raises:
            GitHubAuthError: If the request fails.
        """
        app_jwt = self.generate_jwt()
        url = f"{self.api_url}/app/installations/{installation_id}/access_tokens"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {app_jwt}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )

        if response.status_code != 201:
            logger.error(
                "failed to get installation token",
                installation_id=installation_id,
                status_code=response.status_code,
                body=response.text[:200],
            )
            raise GitHubAuthError(
                f"Failed to get installation token: {response.status_code} - {response.text[:200]}"
            )

        data = response.json()
        token = data["token"]
        expires_at = data.get("expires_at", "")

        # Parse expiry and cache (tokens last ~1 hour)
        # Use a conservative 55-minute cache time
        cache_expires = time.time() + 3300
        self._token_cache[installation_id] = (token, cache_expires)

        logger.info(
            "installation token acquired",
            installation_id=installation_id,
            expires_at=expires_at,
        )

        return token

    def clear_token_cache(self, installation_id: int | None = None) -> None:
        """Clear cached installation tokens.

        Args:
            installation_id: Specific installation to clear, or None to clear all.
        """
        if installation_id is not None:
            self._token_cache.pop(installation_id, None)
        else:
            self._token_cache.clear()


class GitHubAuthError(Exception):
    """Raised when GitHub App authentication fails."""
