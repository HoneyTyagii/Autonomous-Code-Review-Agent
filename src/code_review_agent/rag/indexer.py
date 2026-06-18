"""Repository indexer for building the RAG knowledge base.

Processes repository source files into chunked, embedded documents
stored in the vector database. This enables semantic search over the
codebase to provide the review engine with relevant context.
"""

from dataclasses import dataclass
import hashlib
from typing import Any

from code_review_agent.analysis.ast_parser import TreeSitterParser, ASTAnalysis
from code_review_agent.rag.embeddings import EmbeddingService
from code_review_agent.rag.vector_store import VectorStoreClient
from code_review_agent.logging import get_logger

logger = get_logger("indexer")


@dataclass
class CodeChunk:
    """A chunk of code prepared for embedding.

    Attributes:
        id: Unique identifier (hash of repo + path + range).
        content: The code text content.
        file_path: Path within the repository.
        start_line: Starting line number.
        end_line: Ending line number.
        language: Programming language.
        symbol_name: Name of the enclosing function/class, if any.
        symbol_kind: Kind of enclosing symbol (function/class/method).
        repo_full_name: Full repository name (owner/repo).
    """

    id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str | None = None
    symbol_name: str | None = None
    symbol_kind: str | None = None
    repo_full_name: str = ""

    @property
    def metadata(self) -> dict[str, Any]:
        """Convert to metadata dict for vector store."""
        meta: dict[str, Any] = {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "repo": self.repo_full_name,
        }
        if self.language:
            meta["language"] = self.language
        if self.symbol_name:
            meta["symbol_name"] = self.symbol_name
        if self.symbol_kind:
            meta["symbol_kind"] = self.symbol_kind
        return meta


class CodeChunker:
    """Splits source code into semantically meaningful chunks.

    Uses AST analysis when available to split at function/class boundaries,
    falling back to sliding-window chunking for unsupported languages.

    Attributes:
        parser: Tree-sitter parser for AST-aware chunking.
        max_chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping lines between sliding-window chunks.
    """

    def __init__(
        self,
        max_chunk_size: int = 2000,
        overlap_lines: int = 5,
    ) -> None:
        """Initialize the code chunker.

        Args:
            max_chunk_size: Maximum characters per chunk.
            overlap_lines: Lines of overlap between window chunks.
        """
        self.parser = TreeSitterParser()
        self.max_chunk_size = max_chunk_size
        self.overlap_lines = overlap_lines

    def chunk_file(
        self,
        content: str,
        file_path: str,
        language: str | None,
        repo_full_name: str,
    ) -> list[CodeChunk]:
        """Chunk a source file into embeddable pieces.

        Attempts AST-aware chunking first (splitting at symbol boundaries),
        then falls back to sliding-window chunking.

        Args:
            content: The file's source code.
            file_path: Path within the repository.
            language: Programming language.
            repo_full_name: Full repository name.

        Returns:
            List of CodeChunk objects ready for embedding.
        """
        if not content.strip():
            return []

        chunks: list[CodeChunk] = []

        # Try AST-aware chunking for supported languages
        if language and language in TreeSitterParser.SUPPORTED_LANGUAGES:
            analysis = self.parser.parse(content, language)
            if not analysis.errors:
                chunks = self._chunk_by_symbols(
                    content, file_path, language, repo_full_name, analysis
                )

        # Fall back to sliding window if AST chunking produced nothing
        if not chunks:
            chunks = self._chunk_by_window(
                content, file_path, language, repo_full_name
            )

        return chunks

    def _chunk_by_symbols(
        self,
        content: str,
        file_path: str,
        language: str,
        repo_full_name: str,
        analysis: ASTAnalysis,
    ) -> list[CodeChunk]:
        """Chunk code at function/class boundaries using AST.

        Args:
            content: Source code.
            file_path: File path.
            language: Programming language.
            repo_full_name: Repository name.
            analysis: AST analysis results.

        Returns:
            List of symbol-based chunks.
        """
        chunks: list[CodeChunk] = []
        lines = content.split("\n")

        # Get top-level symbols (functions and classes, not methods inside classes)
        top_level_symbols = [
            s for s in analysis.symbols
            if s.parent is None
        ]

        if not top_level_symbols:
            return []

        for symbol in top_level_symbols:
            start = symbol.start_line - 1  # Convert to 0-indexed
            end = symbol.end_line
            symbol_content = "\n".join(lines[start:end])

            # If the symbol is too large, sub-chunk it
            if len(symbol_content) > self.max_chunk_size:
                sub_chunks = self._chunk_by_window(
                    symbol_content,
                    file_path,
                    language,
                    repo_full_name,
                    base_line=symbol.start_line,
                )
                for sc in sub_chunks:
                    sc.symbol_name = symbol.name
                    sc.symbol_kind = symbol.kind.value
                chunks.extend(sub_chunks)
            else:
                chunk_id = self._make_id(repo_full_name, file_path, symbol.start_line, symbol.end_line)
                chunks.append(
                    CodeChunk(
                        id=chunk_id,
                        content=symbol_content,
                        file_path=file_path,
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                        language=language,
                        symbol_name=symbol.name,
                        symbol_kind=symbol.kind.value,
                        repo_full_name=repo_full_name,
                    )
                )

        return chunks

    def _chunk_by_window(
        self,
        content: str,
        file_path: str,
        language: str | None,
        repo_full_name: str,
        base_line: int = 1,
    ) -> list[CodeChunk]:
        """Chunk code using a sliding window approach.

        Args:
            content: Source code.
            file_path: File path.
            language: Programming language.
            repo_full_name: Repository name.
            base_line: Line number offset for the start of this content.

        Returns:
            List of window-based chunks.
        """
        lines = content.split("\n")
        chunks: list[CodeChunk] = []

        # Calculate lines per chunk (approximate by avg line length)
        avg_line_len = max(1, len(content) // max(1, len(lines)))
        lines_per_chunk = max(10, self.max_chunk_size // avg_line_len)

        i = 0
        while i < len(lines):
            end_idx = min(i + lines_per_chunk, len(lines))
            chunk_content = "\n".join(lines[i:end_idx])

            start_line = base_line + i
            end_line = base_line + end_idx - 1

            chunk_id = self._make_id(repo_full_name, file_path, start_line, end_line)
            chunks.append(
                CodeChunk(
                    id=chunk_id,
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    repo_full_name=repo_full_name,
                )
            )

            # Advance with overlap
            i = end_idx - self.overlap_lines
            if i <= (end_idx - lines_per_chunk):
                break  # Prevent infinite loop for tiny chunks

        return chunks

    @staticmethod
    def _make_id(repo: str, path: str, start: int, end: int) -> str:
        """Generate a deterministic chunk ID.

        Args:
            repo: Repository full name.
            path: File path.
            start: Start line.
            end: End line.

        Returns:
            SHA-256 based ID string.
        """
        raw = f"{repo}:{path}:{start}-{end}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class RepositoryIndexer:
    """Indexes a repository's code into the vector store for RAG retrieval.

    Orchestrates the full indexing pipeline: fetching files, chunking,
    embedding, and storing in ChromaDB. Supports incremental updates.

    Attributes:
        embedding_service: Service for generating embeddings.
        vector_store: ChromaDB client for storage.
        chunker: Code chunking strategy.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreClient,
        chunker: CodeChunker | None = None,
    ) -> None:
        """Initialize the repository indexer.

        Args:
            embedding_service: Configured embedding service.
            vector_store: ChromaDB client.
            chunker: Optional custom chunker. Defaults to standard CodeChunker.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.chunker = chunker or CodeChunker()

    async def index_files(
        self,
        repo_full_name: str,
        files: dict[str, str],
        languages: dict[str, str | None] | None = None,
        batch_size: int = 50,
    ) -> int:
        """Index a set of repository files into the vector store.

        Args:
            repo_full_name: Full repository name (owner/repo).
            files: Mapping of file_path -> file_content.
            languages: Optional mapping of file_path -> language.
            batch_size: Number of chunks to embed and store at once.

        Returns:
            Total number of chunks indexed.
        """
        logger.info(
            "indexing repository files",
            repo=repo_full_name,
            file_count=len(files),
        )

        # Chunk all files
        all_chunks: list[CodeChunk] = []
        for file_path, content in files.items():
            language = (languages or {}).get(file_path)
            chunks = self.chunker.chunk_file(
                content=content,
                file_path=file_path,
                language=language,
                repo_full_name=repo_full_name,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("no chunks generated", repo=repo_full_name)
            return 0

        # Process in batches
        total_indexed = 0
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]

            # Generate embeddings
            texts = [chunk.content for chunk in batch]
            embeddings = await self.embedding_service.embed_many(texts)

            # Store in vector DB
            self.vector_store.add_documents(
                collection_name=f"{VectorStoreClient.COLLECTION_REPO_CODE}_{repo_full_name.replace('/', '_')}",
                ids=[chunk.id for chunk in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[chunk.metadata for chunk in batch],
            )

            total_indexed += len(batch)

        logger.info(
            "indexing complete",
            repo=repo_full_name,
            chunks_indexed=total_indexed,
        )

        return total_indexed

    async def delete_index(self, repo_full_name: str) -> None:
        """Delete the entire index for a repository.

        Args:
            repo_full_name: Full repository name.
        """
        collection_name = f"{VectorStoreClient.COLLECTION_REPO_CODE}_{repo_full_name.replace('/', '_')}"
        self.vector_store.delete_collection(collection_name)
        logger.info("repository index deleted", repo=repo_full_name)
