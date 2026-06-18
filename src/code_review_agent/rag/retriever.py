"""Context retriever for fetching relevant code and standards during review.

Queries the vector store to find code snippets and documentation
relevant to the current PR changes, providing the LLM with rich
contextual information for informed reviews.
"""

from dataclasses import dataclass, field
from typing import Any

from code_review_agent.rag.embeddings import EmbeddingService
from code_review_agent.rag.vector_store import VectorStoreClient, SearchResult
from code_review_agent.logging import get_logger

logger = get_logger("retriever")


@dataclass
class RetrievalContext:
    """Retrieved context for a code review.

    Attributes:
        related_code: Code snippets from the same repo relevant to the changes.
        coding_standards: Applicable coding standards and rules.
        past_reviews: Relevant past review comments for learning.
    """

    related_code: list[SearchResult] = field(default_factory=list)
    coding_standards: list[SearchResult] = field(default_factory=list)
    past_reviews: list[SearchResult] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        """Check if any context was retrieved."""
        return bool(self.related_code or self.coding_standards or self.past_reviews)

    def format_related_code(self, max_results: int = 5) -> str:
        """Format related code snippets for LLM prompt inclusion.

        Args:
            max_results: Maximum number of snippets to include.

        Returns:
            Formatted string of code context.
        """
        if not self.related_code:
            return "No related code found in repository."

        parts: list[str] = []
        for result in self.related_code[:max_results]:
            meta = result.metadata
            file_path = meta.get("file_path", "unknown")
            start = meta.get("start_line", "?")
            end = meta.get("end_line", "?")
            symbol = meta.get("symbol_name", "")

            header = f"--- {file_path} (lines {start}-{end})"
            if symbol:
                header += f" [{symbol}]"

            parts.append(f"{header}\n{result.content}")

        return "\n\n".join(parts)

    def format_standards(self, max_results: int = 5) -> str:
        """Format coding standards for LLM prompt inclusion.

        Args:
            max_results: Maximum number of standards to include.

        Returns:
            Formatted string of applicable standards.
        """
        if not self.coding_standards:
            return "No specific coding standards found."

        parts: list[str] = []
        for result in self.coding_standards[:max_results]:
            meta = result.metadata
            rule_id = meta.get("rule_id", "")
            category = meta.get("category", "general")

            header = f"[{category}]"
            if rule_id:
                header += f" ({rule_id})"

            parts.append(f"{header}: {result.content}")

        return "\n\n".join(parts)


class ContextRetriever:
    """Retrieves relevant context from the vector store for code review.

    Queries multiple collections (code, standards, past reviews) to
    assemble the most relevant context for reviewing a set of changes.

    Attributes:
        embedding_service: Service for embedding query texts.
        vector_store: ChromaDB client for similarity search.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreClient,
    ) -> None:
        """Initialize the context retriever.

        Args:
            embedding_service: Configured embedding service.
            vector_store: ChromaDB client.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def retrieve_context(
        self,
        repo_full_name: str,
        query_texts: list[str],
        n_code_results: int = 10,
        n_standards_results: int = 5,
        n_review_results: int = 5,
    ) -> RetrievalContext:
        """Retrieve all relevant context for a review.

        Searches code, standards, and past reviews collections
        using the provided query texts (typically changed code snippets).

        Args:
            repo_full_name: Repository to search within.
            query_texts: Texts representing the changes to find context for.
            n_code_results: Max related code results.
            n_standards_results: Max coding standards results.
            n_review_results: Max past review results.

        Returns:
            RetrievalContext with all retrieved information.
        """
        context = RetrievalContext()

        # Combine query texts into a single query for efficiency
        combined_query = "\n".join(query_texts[:5])  # Limit to prevent token overflow

        if not combined_query.strip():
            return context

        # Generate query embedding
        query_embedding = await self.embedding_service.embed(combined_query)

        # Search related code
        context.related_code = await self._search_code(
            repo_full_name, query_embedding, n_code_results
        )

        # Search coding standards
        context.coding_standards = await self._search_standards(
            repo_full_name, query_embedding, n_standards_results
        )

        # Search past reviews
        context.past_reviews = await self._search_reviews(
            repo_full_name, query_embedding, n_review_results
        )

        logger.info(
            "context retrieved",
            repo=repo_full_name,
            code_results=len(context.related_code),
            standards_results=len(context.coding_standards),
            review_results=len(context.past_reviews),
        )

        return context

    async def _search_code(
        self,
        repo_full_name: str,
        query_embedding: list[float],
        n_results: int,
    ) -> list[SearchResult]:
        """Search the repository code collection.

        Args:
            repo_full_name: Repository name.
            query_embedding: Query vector.
            n_results: Max results.

        Returns:
            Matching code snippets.
        """
        collection_name = (
            f"{VectorStoreClient.COLLECTION_REPO_CODE}"
            f"_{repo_full_name.replace('/', '_')}"
        )
        try:
            return self.vector_store.query(
                collection_name=collection_name,
                query_embedding=query_embedding,
                n_results=n_results,
            )
        except Exception as e:
            logger.debug("code search failed", error=str(e))
            return []

    async def _search_standards(
        self,
        repo_full_name: str,
        query_embedding: list[float],
        n_results: int,
    ) -> list[SearchResult]:
        """Search the coding standards collection.

        Args:
            repo_full_name: Repository name (for repo-specific standards).
            query_embedding: Query vector.
            n_results: Max results.

        Returns:
            Matching coding standards.
        """
        try:
            return self.vector_store.query(
                collection_name=VectorStoreClient.COLLECTION_STANDARDS,
                query_embedding=query_embedding,
                n_results=n_results,
                where={"repo": repo_full_name},
            )
        except Exception as e:
            logger.debug("standards search failed", error=str(e))
            return []

    async def _search_reviews(
        self,
        repo_full_name: str,
        query_embedding: list[float],
        n_results: int,
    ) -> list[SearchResult]:
        """Search past review comments for similar patterns.

        Args:
            repo_full_name: Repository name.
            query_embedding: Query vector.
            n_results: Max results.

        Returns:
            Relevant past review comments.
        """
        try:
            return self.vector_store.query(
                collection_name=VectorStoreClient.COLLECTION_REVIEWS,
                query_embedding=query_embedding,
                n_results=n_results,
                where={"repo": repo_full_name},
            )
        except Exception as e:
            logger.debug("reviews search failed", error=str(e))
            return []
