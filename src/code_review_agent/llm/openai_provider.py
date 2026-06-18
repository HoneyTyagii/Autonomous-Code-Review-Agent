"""OpenAI LLM provider implementation.

Wraps the OpenAI async client with retry logic, structured output
support, and usage tracking.
"""

import json
from typing import Any

import openai
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
)
from code_review_agent.logging import get_logger

logger = get_logger("llm_openai")


class OpenAIProvider(BaseLLM):
    """OpenAI API provider for GPT-4 and other models.

    Supports both standard text generation and structured JSON output
    using OpenAI's response_format parameter.

    Attributes:
        client: The async OpenAI client instance.
    """

    def __init__(self, api_key: str, config: LLMConfig | None = None) -> None:
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key.
            config: Optional LLM configuration. Uses defaults if None.
        """
        super().__init__(config or LLMConfig(model="gpt-4o"))
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=self.config.timeout,
        )

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return "openai"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(
            (openai.APITimeoutError, openai.RateLimitError, openai.APIConnectionError)
        ),
    )
    async def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a completion using the OpenAI API.

        Args:
            messages: Conversation messages.
            temperature: Override sampling temperature.
            max_tokens: Override max completion tokens.

        Returns:
            LLMResponse with generated content and usage data.
        """
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[msg.to_dict() for msg in messages],
            temperature=temp,
            max_tokens=tokens,
            top_p=self.config.top_p,
        )

        choice = response.choices[0]
        usage = response.usage

        result = LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
            raw_response=response,
        )

        logger.debug(
            "openai generation complete",
            model=response.model,
            tokens=result.total_tokens,
            finish_reason=result.finish_reason,
        )

        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(
            (openai.APITimeoutError, openai.RateLimitError, openai.APIConnectionError)
        ),
    )
    async def generate_structured(
        self,
        messages: list[Message],
        response_schema: dict[str, Any],
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate a structured JSON response using OpenAI's response_format.

        Args:
            messages: Conversation messages.
            response_schema: JSON schema for the expected response structure.
            temperature: Override sampling temperature.

        Returns:
            LLMResponse with JSON content matching the schema.
        """
        temp = temperature if temperature is not None else self.config.temperature

        # Instruct the model to output valid JSON
        schema_instruction = Message(
            role=messages[0].role,
            content=(
                f"{messages[0].content}\n\n"
                f"You MUST respond with valid JSON matching this schema:\n"
                f"```json\n{json.dumps(response_schema, indent=2)}\n```"
            ),
        )
        modified_messages = [schema_instruction] + messages[1:]

        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[msg.to_dict() for msg in modified_messages],
            temperature=temp,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )

        choice = response.choices[0]
        usage = response.usage

        result = LLMResponse(
            content=choice.message.content or "{}",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
            raw_response=response,
        )

        logger.debug(
            "openai structured generation complete",
            model=response.model,
            tokens=result.total_tokens,
        )

        return result
