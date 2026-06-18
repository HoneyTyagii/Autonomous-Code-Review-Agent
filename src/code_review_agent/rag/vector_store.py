"""ChromaDB vector store client for document storage and retrieval.

Manages collections of embedded documents in ChromaDB, providing
methods for adding, querying, and managing vector data used by
the RAG pipeline.
"""

from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from code_review_agent.config import get_settings
from code_review_agent.logging import get_logger

logger = get_logger("vector_store")


@dataclass
class SearchResult:
    """A single result from a vector similarity search.

    Attributes:
        id: Document identifier.
        content: The stored document text.
        metadata: Associated metadata dictionary.
        distance: Similarity distance (lower is more similar).
        score: Normalized similarity score (0-1, higher is more similar).
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0
    score: float = 0.0


class VectorStoreClient:
    """ChromaDB client for managing vector embeddings.

    Provides a clean interface over ChromaDB for creating collections,
    adding documents with metadata, and performing similarity search.

    Attributes:
        client: The ChromaDB client instance.
    """

    # Default collection names
    COLLECTION_REPO_CODE = "repository_code"
    COLLECTION_STANDARDS = "coding_standards"
    COLLECTION_REVIEWS = "review_history"

    def __init__(self, client: chromadb.ClientAPI | None = None) -> None:
        """Initialize the vector store client.

        Args:
            client: Optional pre-configured ChromaDB client. If None,
                   creates one from application settings.
        """
        if client:
            self.client = client
        else:
            self.client = self._create_from_settings()

    @staticmethod
    def _create_from_settings() -> chromadb.ClientAPI:
        """Create a ChromaDB client from application settings.

        Returns:
            Configured ChromaDB client.
        """
        settings = get_settings()

        # Use HTTP client for connecting to a ChromaDB server
        try:
            client = chromadb.HttpClient(
                host=settings.chromadb_host,
                port=settings.chromadb_port,
            )
            # Verify connection
            client.heartbeat()
            logger.info(
                "connected to ChromaDB server",
                host=settings.chromadb_host,
                port=settings.chromadb_port,
            )
            return client
        except Exception:
            # Fall back to persistent local client
            logger.warning(
                "ChromaDB server unavailable, using local persistent storage",
                persist_dir=settings.chromadb_persist_dir,
            )
            return chromadb.PersistentClient(
                path=settings.chromadb_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

    def get_or_create_collection(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> chromadb.Collection:
        """Get or create a ChromaDB collection.

        Args:
            name: Collection name.
            metadata: Optional collection-level metadata (e.g., distance function).

        Returns:
            The ChromaDB collection.
        """
        collection_metadata = metadata or {"hnsw:space": "cosine"}
        collection = self.client.get_or_create_collection(
            name=name,
            metadata=collection_metadata,
        )
        logger.debug("collection ready", name=name, count=collection.count())
        return collection

    def add_documents(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents to a collection.

        Args:
            collection_name: Target collection name.
            ids: Unique identifiers for each document.
            documents: The document texts.
            embeddings: Pre-computed embeddings (optional if collection has embedding function).
            metadatas: Per-document metadata dictionaries.
        """
        collection = self.get_or_create_collection(collection_name)

        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
        }
        if embeddings:
            kwargs["embeddings"] = embeddings
        if metadatas:
            kwargs["metadatas"] = metadatas

        collection.upsert(**kwargs)

        logger.info(
            "documents added",
            collection=collection_name,
            count=len(ids),
        )

    def query(
        self,
        collection_name: str,
        query_embedding: list[float] | None = None,
        query_text: str | None = None,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Query a collection for similar documents.

        Args:
            collection_name: Collection to search.
            query_embedding: Embedding vector to search with.
            query_text: Text query (uses collection's embedding function).
            n_results: Maximum number of results to return.
            where: Metadata filter conditions.
            where_document: Document content filter conditions.

        Returns:
            List of SearchResult objects ordered by relevance.
        """
        collection = self.get_or_create_collection(collection_name)

        kwargs: dict[str, Any] = {"n_results": n_results}
        if query_embedding:
            kwargs["query_embeddings"] = [query_embedding]
        elif query_text:
            kwargs["query_texts"] = [query_text]
        else:
            raise ValueError("Either query_embedding or query_text must be provided")

        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = collection.query(**kwargs)

        # Parse results into SearchResult objects
        search_results: list[SearchResult] = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

            for i, doc_id in enumerate(ids):
                # Convert cosine distance to similarity score
                distance = distances[i]
                score = max(0.0, 1.0 - distance)

                search_results.append(
                    SearchResult(
                        id=doc_id,
                        content=documents[i],
                        metadata=metadatas[i] or {},
                        distance=distance,
                        score=score,
                    )
                )

        logger.debug(
            "query completed",
            collection=collection_name,
            results=len(search_results),
        )

        return search_results

    def delete_collection(self, name: str) -> None:
        """Delete an entire collection.

        Args:
            name: Collection name to delete.
        """
        try:
            self.client.delete_collection(name=name)
            logger.info("collection deleted", name=name)
        except Exception as e:
            logger.warning("failed to delete collection", name=name, error=str(e))

    def delete_documents(
        self,
        collection_name: str,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        """Delete documents from a collection.

        Args:
            collection_name: Target collection.
            ids: Specific document IDs to delete.
            where: Metadata filter to select documents for deletion.
        """
        collection = self.get_or_create_collection(collection_name)

        kwargs: dict[str, Any] = {}
        if ids:
            kwargs["ids"] = ids
        if where:
            kwargs["where"] = where

        if kwargs:
            collection.delete(**kwargs)
            logger.info("documents deleted", collection=collection_name)

    def collection_count(self, name: str) -> int:
        """Get the number of documents in a collection.

        Args:
            name: Collection name.

        Returns:
            Document count.
        """
        collection = self.get_or_create_collection(name)
        return collection.count()
