"""Abstract base class and shared types for LLM providers.

Defines the common interface that all LLM providers must implement,
along with request/response data structures used throughout the
review engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a chat conversation.

    Attributes:
        role: The message sender role.
        content: The message text content.
    """

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to a provider-agnostic dictionary."""
        return {"role": self.role.value, "content": self.content}


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Attributes:
        content: The generated text content.
        model: The model that produced the response.
        prompt_tokens: Tokens used in the prompt.
        completion_tokens: Tokens used in the completion.
        total_tokens: Total tokens consumed.
        finish_reason: Why generation stopped (stop, length, etc.).
        raw_response: The provider's raw response object.
    """

    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    raw_response: Any = None

    @property
    def was_truncated(self) -> bool:
        """Check if the response was cut short by token limits."""
        return self.finish_reason == "length"


@dataclass
class LLMConfig:
    """Configuration for an LLM provider.

    Attributes:
        model: Model identifier string.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in the completion.
        top_p: Nucleus sampling parameter.
        timeout: Request timeout in seconds.
    """

    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    top_p: float = 1.0
    timeout: float = 120.0


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement these methods to provide
    a consistent interface for the review engine.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize with provider configuration.

        Args:
            config: LLM configuration parameters.
        """
        self.config = config

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a completion from a list of messages.

        Args:
            messages: The conversation messages.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            The LLM response with content and usage stats.
        """

    @abstractmethod
    async def generate_structured(
        self,
        messages: list[Message],
        response_schema: dict[str, Any],
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate a structured (JSON) response matching a schema.

        Args:
            messages: The conversation messages.
            response_schema: JSON schema the response must conform to.
            temperature: Override default temperature.

        Returns:
            LLMResponse with JSON content matching the schema.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider's name identifier."""

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self.config.model

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Verify provider connectivity and return status metadata.

        Implementations should perform a lightweight, non-mutating API call
        (for example, listing available models) and measure latency. On
        failure an exception is raised so callers can mark the service as
        unhealthy.

        Returns:
            Dictionary containing at least ``status``, ``provider``,
            ``model``, and ``latency_ms`` keys.
        """
