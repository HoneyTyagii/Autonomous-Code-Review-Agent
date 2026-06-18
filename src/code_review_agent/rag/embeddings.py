"""Embedding service abstraction for text vectorization.

Supports OpenAI embeddings and local Sentence Transformers models,
with batching and caching for efficient large-scale indexing.
"""

from abc import ABC, abstractmethod

import openai

from code_review_agent.config import EmbeddingProvider, get_settings
from code_review_agent.logging import get_logger

logger = get_logger("embeddings")


class BaseEmbedding(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of embeddings produced by this provider."""
        ...


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI text embedding provider.

    Uses the OpenAI API to generate text embeddings. Handles batching
    to stay within API rate limits.

    Attributes:
        model: The embedding model name.
        client: The OpenAI async client.
    """

    # Batch size limit for OpenAI API
    MAX_BATCH_SIZE = 2048

    # Known model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        """Initialize the OpenAI embedding provider.

        Args:
            api_key: OpenAI API key.
            model: The embedding model to use.
        """
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1536)
        logger.info("openai embedding initialized", model=model)

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        return self._dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts using OpenAI.

        Automatically splits into sub-batches if the input exceeds
        the maximum batch size.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i : i + self.MAX_BATCH_SIZE]
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        logger.debug("texts embedded", count=len(texts))
        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query.

        Args:
            text: The query text.

        Returns:
            The embedding vector.
        """
        response = await self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        return response.data[0].embedding


class SentenceTransformerEmbedding(BaseEmbedding):
    """Local embedding provider using Sentence Transformers.

    Runs embedding inference locally without external API calls.
    Suitable for offline use or when cost is a concern.

    Attributes:
        model_name: The HuggingFace model name.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize with a Sentence Transformers model.

        Args:
            model_name: The model to load from HuggingFace.
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("sentence transformer initialized", model=model_name)

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        return self._dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings locally using Sentence Transformers.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    async def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query.

        Args:
            text: The query text.

        Returns:
            The embedding vector.
        """
        embedding = self._model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()


class EmbeddingService:
    """Factory and facade for embedding providers.

    Creates the appropriate embedding provider based on application
    settings and provides a unified interface.
    """

    def __init__(self, provider: BaseEmbedding | None = None) -> None:
        """Initialize the embedding service.

        Args:
            provider: An explicit provider, or None to auto-configure from settings.
        """
        if provider:
            self._provider = provider
        else:
            self._provider = self._create_from_settings()

    @staticmethod
    def _create_from_settings() -> BaseEmbedding:
        """Create an embedding provider from application settings.

        Returns:
            The configured embedding provider.
        """
        settings = get_settings()

        if settings.embedding_provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbedding(
                api_key=settings.openai_api_key,
                model=settings.embedding_model,
            )
        else:
            return SentenceTransformerEmbedding(
                model_name=settings.embedding_model,
            )

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""
        return self._provider.dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: The texts to embed.

        Returns:
            List of embedding vectors.
        """
        return await self._provider.embed_texts(texts)

    async def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a query.

        Args:
            text: The query text.

        Returns:
            The embedding vector.
        """
        return await self._provider.embed_query(text)
