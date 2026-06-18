"""LLM provider factory.

Creates the appropriate LLM provider instance based on application
configuration, with sensible defaults for code review tasks.
"""

from code_review_agent.config import LLMProvider, get_settings
from code_review_agent.llm.base import BaseLLM, LLMConfig
from code_review_agent.llm.openai_provider import OpenAIProvider
from code_review_agent.llm.anthropic_provider import AnthropicProvider
from code_review_agent.logging import get_logger

logger = get_logger("llm_factory")


def create_llm_provider(
    provider: LLMProvider | None = None,
    config: LLMConfig | None = None,
) -> BaseLLM:
    """Create an LLM provider from settings or explicit parameters.

    Args:
        provider: Explicit provider choice. If None, uses settings.
        config: Explicit LLM config. If None, uses settings-based defaults.

    Returns:
        Configured LLM provider instance.

    Raises:
        ValueError: If the provider is unknown or API key is missing.
    """
    settings = get_settings()
    chosen_provider = provider or settings.llm_provider

    if chosen_provider == LLMProvider.OPENAI:
        api_key = settings.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")

        llm_config = config or LLMConfig(
            model=settings.openai_model,
            temperature=0.1,
            max_tokens=4096,
        )
        logger.info("creating OpenAI provider", model=llm_config.model)
        return OpenAIProvider(api_key=api_key, config=llm_config)

    elif chosen_provider == LLMProvider.ANTHROPIC:
        api_key = settings.anthropic_api_key
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")

        llm_config = config or LLMConfig(
            model=settings.anthropic_model,
            temperature=0.1,
            max_tokens=4096,
        )
        logger.info("creating Anthropic provider", model=llm_config.model)
        return AnthropicProvider(api_key=api_key, config=llm_config)

    else:
        raise ValueError(f"Unknown LLM provider: {chosen_provider}")
