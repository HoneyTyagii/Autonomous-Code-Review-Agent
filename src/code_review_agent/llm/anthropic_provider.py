"""Anthropic LLM provider implementation.

Wraps the Anthropic async client for Claude models with retry logic,
structured output support, and usage tracking.
"""

import json
import time
from typing import Any

import anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from code_review_agent.llm.base import (
    BaseLLM,
    LLMConfig,
    LLMResponse,
    Message,
    MessageRole,
)
from code_review_agent.logging import get_logger

logger = get_logger("llm_anthropic")


class AnthropicProvider(BaseLLM):
    """Anthropic API provider for Claude models.

    Handles the differences in Anthropic's message format (separate
    system parameter, no system role in messages array).

    Attributes:
        client: The async Anthropic client instance.
    """

    def __init__(self, api_key: str, config: LLMConfig | None = None) -> None:
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key.
            config: Optional LLM configuration.
        """
        default_config = LLMConfig(model="claude-sonnet-4-20250514")
        super().__init__(config or default_config)
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=self.config.timeout,
        )

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return "anthropic"

    async def health_check(self) -> dict[str, Any]:
        """Verify connectivity to the Anthropic API.

        Performs a lightweight call to list models and measures latency.

        Returns:
            Status metadata with provider, model, and latency_ms.

        Raises:
            anthropic.APIError: If the API call fails.
        """
        start = time.perf_counter()
        await self.client.models.list(limit=1)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        return {
            "status": "healthy",
            "provider": self.provider_name,
            "model": self.model_name,
            "latency_ms": latency_ms,
        }

    def _prepare_messages(
        self, messages: list[Message]
    ) -> tuple[str, list[dict[str, str]]]:
        """Separate system message from conversation messages.

        Anthropic's API takes the system prompt as a separate parameter,
        not as part of the messages array.

        Args:
            messages: Full message list including system message.

        Returns:
            Tuple of (system_prompt, conversation_messages).
        """
        system_prompt = ""
        conversation: list[dict[str, str]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt += msg.content + "\n"
            else:
                conversation.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        # Anthropic requires at least one user message
        if not conversation:
            conversation.append({"role": "user", "content": "Please respond."})

        return system_prompt.strip(), conversation

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(
            (anthropic.APITimeoutError, anthropic.RateLimitError, anthropic.APIConnectionError)
        ),
    )
    async def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a completion using the Anthropic API.

        Args:
            messages: Conversation messages.
            temperature: Override sampling temperature.
            max_tokens: Override max completion tokens.

        Returns:
            LLMResponse with generated content and usage data.
        """
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        system_prompt, conversation = self._prepare_messages(messages)

        response = await self.client.messages.create(
            model=self.config.model,
            system=system_prompt,
            messages=conversation,
            temperature=temp,
            max_tokens=tokens,
        )

        # Extract text content from response
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        result = LLMResponse(
            content=content,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason or "end_turn",
            raw_response=response,
        )

        logger.debug(
            "anthropic generation complete",
            model=response.model,
            tokens=result.total_tokens,
            finish_reason=result.finish_reason,
        )

        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(
            (anthropic.APITimeoutError, anthropic.RateLimitError, anthropic.APIConnectionError)
        ),
    )
    async def generate_structured(
        self,
        messages: list[Message],
        response_schema: dict[str, Any],
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate a structured JSON response using Anthropic's API.

        Adds schema instructions to the system prompt and requests
        JSON output.

        Args:
            messages: Conversation messages.
            response_schema: JSON schema for the expected response.
            temperature: Override sampling temperature.

        Returns:
            LLMResponse with JSON content.
        """
        temp = temperature if temperature is not None else self.config.temperature
        system_prompt, conversation = self._prepare_messages(messages)

        # Append schema instruction to system prompt
        schema_instruction = (
            f"\n\nYou MUST respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(response_schema, indent=2)}\n```\n"
            f"Output ONLY the JSON object, no other text."
        )
        system_prompt += schema_instruction

        response = await self.client.messages.create(
            model=self.config.model,
            system=system_prompt,
            messages=conversation,
            temperature=temp,
            max_tokens=self.config.max_tokens,
        )

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        # Clean potential markdown wrapping
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

        result = LLMResponse(
            content=content,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason or "end_turn",
            raw_response=response,
        )

        logger.debug(
            "anthropic structured generation complete",
            model=response.model,
            tokens=result.total_tokens,
        )

        return result
