"""Embedding generation service supporting multiple providers.

Abstracts embedding generation behind a unified interface,
supporting OpenAI embeddings and local Sentence Transformers.
"""

from abc import ABC, abstractmethod

import openai

from code_review_agent.config import EmbeddingProvider, get_settings
from code_review_agent.logging import get_logger

logger = get_logger("embeddings")


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI API embedding provider.

    Uses OpenAI's text-embedding models for high-quality embeddings.

    Attributes:
        client: The async OpenAI client.
        model: The embedding model name.
    """

    # Known dimensions for OpenAI models
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        """Initialize the OpenAI embedder.

        Args:
            api_key: OpenAI API key.
            model: Embedding model name.
        """
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1536)

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text using OpenAI.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector.
        """
        response = await self.client.embeddings.create(
            input=[text],
            model=self.model,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts using OpenAI.

        Handles chunking for large batches (OpenAI limit: 2048 inputs).

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in same order as input.
        """
        all_embeddings: list[list[float]] = []
        batch_size = 2000  # Stay under OpenAI's 2048 limit

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await self.client.embeddings.create(
                input=batch,
                model=self.model,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local Sentence Transformer embedding provider.

    Runs embedding models locally without API calls, suitable for
    offline/private deployments.

    Attributes:
        model: The loaded SentenceTransformer model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize with a local model.

        Args:
            model_name: HuggingFace model identifier.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding locally.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate batch embeddings locally.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class EmbeddingService:
    """Unified embedding service with provider abstraction.

    Factory that creates the appropriate embedder based on configuration.

    Attributes:
        embedder: The active embedding provider instance.
    """

    def __init__(self, embedder: BaseEmbedder | None = None) -> None:
        """Initialize the embedding service.

        Args:
            embedder: Optional pre-configured embedder. If None, creates
                     one from application settings.
        """
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = self._create_from_settings()

    @staticmethod
    def _create_from_settings() -> BaseEmbedder:
        """Create an embedder from application settings.

        Returns:
            Configured embedder instance.
        """
        settings = get_settings()

        if settings.embedding_provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbedder(
                api_key=settings.openai_api_key,
                model=settings.embedding_model,
            )
        elif settings.embedding_provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return SentenceTransformerEmbedder(model_name=settings.embedding_model)
        else:
            raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")

    @property
    def dimension(self) -> int:
        """Get the embedding dimensionality."""
        return self.embedder.dimension

    async def embed(self, text: str) -> list[float]:
        """Generate a single embedding.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        return await self.embedder.embed_text(text)

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors.
        """
        return await self.embedder.embed_batch(texts)
