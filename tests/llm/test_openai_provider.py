"""Tests for the OpenAI LLM provider."""

from unittest.mock import AsyncMock

import pytest

from code_review_agent.llm.base import LLMConfig
from code_review_agent.llm.openai_provider import OpenAIProvider


@pytest.fixture
def provider() -> OpenAIProvider:
    """Create an OpenAI provider with a fake API key."""
    return OpenAIProvider(api_key="fake-key", config=LLMConfig(model="gpt-4o"))


@pytest.mark.asyncio
async def test_health_check_success(provider: OpenAIProvider) -> None:
    """Health check should return healthy when models.list succeeds."""
    provider.client.models.list = AsyncMock()

    result = await provider.health_check()

    assert result["status"] == "healthy"
    assert result["provider"] == "openai"
    assert result["model"] == "gpt-4o"
    assert "latency_ms" in result
    assert isinstance(result["latency_ms"], float)
    provider.client.models.list.assert_awaited_once_with(limit=1)


@pytest.mark.asyncio
async def test_health_check_failure(provider: OpenAIProvider) -> None:
    """Health check should propagate API errors."""
    import openai

    provider.client.models.list = AsyncMock(side_effect=openai.APIConnectionError("timeout"))

    with pytest.raises(openai.APIConnectionError, match="timeout"):
        await provider.health_check()
