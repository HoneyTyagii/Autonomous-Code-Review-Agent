"""Tests for the Anthropic LLM provider."""

from unittest.mock import AsyncMock

import pytest

from code_review_agent.llm.anthropic_provider import AnthropicProvider
from code_review_agent.llm.base import LLMConfig


@pytest.fixture
def provider() -> AnthropicProvider:
    """Create an Anthropic provider with a fake API key."""
    return AnthropicProvider(api_key="fake-key", config=LLMConfig(model="claude-test"))


@pytest.mark.asyncio
async def test_health_check_success(provider: AnthropicProvider) -> None:
    """Health check should return healthy when models.list succeeds."""
    provider.client.models.list = AsyncMock()

    result = await provider.health_check()

    assert result["status"] == "healthy"
    assert result["provider"] == "anthropic"
    assert result["model"] == "claude-test"
    assert "latency_ms" in result
    assert isinstance(result["latency_ms"], float)
    provider.client.models.list.assert_awaited_once_with(limit=1)


@pytest.mark.asyncio
async def test_health_check_failure(provider: AnthropicProvider) -> None:
    """Health check should propagate API errors."""
    import anthropic

    provider.client.models.list = AsyncMock(side_effect=anthropic.APIConnectionError("timeout"))

    with pytest.raises(anthropic.APIConnectionError, match="timeout"):
        await provider.health_check()
