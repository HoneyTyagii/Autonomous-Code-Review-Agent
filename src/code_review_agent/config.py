"""Application configuration using pydantic-settings.

Loads settings from environment variables and .env files with
validation, type coercion, and sensible defaults.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables or a .env file.
    Variable names are case-insensitive and match the field names with
    optional prefix removal.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Application ---
    app_env: Environment = Environment.DEVELOPMENT
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    secret_key: str = "change-me"

    # --- GitHub App ---
    github_app_id: str = ""
    github_private_key_path: Path = Path("./github-app-key.pem")
    github_webhook_secret: str = ""
    github_api_url: str = "https://api.github.com"

    # --- LLM ---
    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # --- Embeddings ---
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"

    # --- Database ---
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/code_review_agent"
    )
    database_echo: bool = False

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Vector Store ---
    chromadb_host: str = "localhost"
    chromadb_port: int = 8001
    chromadb_persist_dir: str = "./data/chromadb"

    # --- Docker Sandbox ---
    docker_socket: str = "unix:///var/run/docker.sock"
    sandbox_timeout: int = Field(default=300, description="Sandbox timeout in seconds")
    sandbox_memory_limit: str = "512m"
    sandbox_cpu_limit: float = 1.0

    # --- Security Scanner ---
    enable_security_scan: bool = True
    security_scan_tools: str = "bandit,semgrep"

    # --- Celery ---
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == Environment.DEVELOPMENT

    @property
    def security_tools_list(self) -> list[str]:
        """Parse comma-separated security scan tools into a list."""
        return [tool.strip() for tool in self.security_scan_tools.split(",") if tool.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings singleton.

    Returns:
        The application Settings instance, cached after first creation.
    """
    return Settings()
