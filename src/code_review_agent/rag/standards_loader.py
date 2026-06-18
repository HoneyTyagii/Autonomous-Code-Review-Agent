"""Coding standards ingestion and vector store loading.

Parses coding standards documents (Markdown, YAML, plain text) into
structured rules and loads them into the vector store for retrieval
during code reviews.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from code_review_agent.rag.embeddings import EmbeddingService
from code_review_agent.rag.vector_store import VectorStoreClient
from code_review_agent.logging import get_logger

logger = get_logger("standards_loader")


@dataclass
class CodingRule:
    """A single coding standard rule.

    Attributes:
        id: Unique rule identifier (e.g., "PY-001", "SEC-003").
        title: Short descriptive title.
        description: Full rule description and rationale.
        category: Rule category (security, style, performance, etc.).
        severity: Default severity (critical, high, medium, low).
        language: Language the rule applies to (None for universal).
        examples: Optional code examples showing good/bad patterns.
        tags: Additional classification tags.
    """

    id: str
    title: str
    description: str
    category: str = "general"
    severity: str = "medium"
    language: str | None = None
    examples: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Combine all rule text for embedding."""
        parts = [f"[{self.id}] {self.title}", self.description]
        if self.examples:
            parts.append(f"Examples:\n{self.examples}")
        return "\n\n".join(parts)

    @property
    def metadata(self) -> dict[str, Any]:
        """Convert to metadata dict for vector store."""
        meta: dict[str, Any] = {
            "rule_id": self.id,
            "title": self.title,
            "category": self.category,
            "severity": self.severity,
        }
        if self.language:
            meta["language"] = self.language
        if self.tags:
            meta["tags"] = ",".join(self.tags)
        return meta


class MarkdownStandardsParser:
    """Parses Markdown-formatted coding standards documents.

    Expects a structure where each heading defines a rule, with the
    body providing the description and optional code examples.

    Supported formats:
    - H2 headings (##) as rule boundaries
    - Optional metadata in frontmatter or inline markers
    - Code blocks as examples
    """

    # Pattern for extracting rule ID from heading: "## PY-001: Rule Title"
    RULE_HEADING_RE = re.compile(
        r"^##\s+(?:(?P<id>[A-Z]+-\d+)\s*:\s*)?(?P<title>.+)$"
    )

    # Pattern for metadata markers: <!-- category: security -->
    METADATA_RE = re.compile(
        r"<!--\s*(?P<key>\w+)\s*:\s*(?P<value>[^>]+?)\s*-->"
    )

    def parse(self, content: str, default_category: str = "general") -> list[CodingRule]:
        """Parse a Markdown standards document into rules.

        Args:
            content: The Markdown document text.
            default_category: Category to assign when not specified.

        Returns:
            List of parsed CodingRule objects.
        """
        rules: list[CodingRule] = []
        sections = self._split_by_headings(content)

        rule_counter = 1
        for heading, body in sections:
            match = self.RULE_HEADING_RE.match(heading)
            if not match:
                continue

            rule_id = match.group("id") or f"RULE-{rule_counter:03d}"
            title = match.group("title").strip()

            # Extract metadata from body
            metadata = self._extract_metadata(body)
            category = metadata.get("category", default_category)
            severity = metadata.get("severity", "medium")
            language = metadata.get("language")

            # Extract examples (code blocks)
            examples = self._extract_code_blocks(body)

            # Clean description (remove metadata comments and code blocks)
            description = self._clean_description(body)

            rules.append(
                CodingRule(
                    id=rule_id,
                    title=title,
                    description=description,
                    category=category,
                    severity=severity,
                    language=language,
                    examples=examples if examples else None,
                    tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                )
            )
            rule_counter += 1

        return rules

    def _split_by_headings(self, content: str) -> list[tuple[str, str]]:
        """Split document into (heading, body) pairs at ## boundaries.

        Args:
            content: Full Markdown text.

        Returns:
            List of (heading_line, body_text) tuples.
        """
        sections: list[tuple[str, str]] = []
        current_heading = ""
        current_body_lines: list[str] = []

        for line in content.split("\n"):
            if line.startswith("## "):
                if current_heading:
                    sections.append((current_heading, "\n".join(current_body_lines)))
                current_heading = line
                current_body_lines = []
            else:
                current_body_lines.append(line)

        # Don't forget the last section
        if current_heading:
            sections.append((current_heading, "\n".join(current_body_lines)))

        return sections

    def _extract_metadata(self, body: str) -> dict[str, str]:
        """Extract metadata from HTML comments in the body.

        Args:
            body: Section body text.

        Returns:
            Metadata key-value pairs.
        """
        metadata: dict[str, str] = {}
        for match in self.METADATA_RE.finditer(body):
            metadata[match.group("key")] = match.group("value").strip()
        return metadata

    def _extract_code_blocks(self, body: str) -> str:
        """Extract code blocks from the body.

        Args:
            body: Section body text.

        Returns:
            Combined code block content.
        """
        blocks: list[str] = []
        in_block = False
        current_block: list[str] = []

        for line in body.split("\n"):
            if line.startswith("```"):
                if in_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                in_block = not in_block
            elif in_block:
                current_block.append(line)

        return "\n---\n".join(blocks)

    def _clean_description(self, body: str) -> str:
        """Remove metadata comments and code blocks from body.

        Args:
            body: Raw section body.

        Returns:
            Cleaned description text.
        """
        # Remove metadata comments
        cleaned = self.METADATA_RE.sub("", body)

        # Remove code blocks
        lines: list[str] = []
        in_block = False
        for line in cleaned.split("\n"):
            if line.startswith("```"):
                in_block = not in_block
                continue
            if not in_block:
                lines.append(line)

        return "\n".join(lines).strip()


class StandardsLoader:
    """Loads coding standards into the vector store for retrieval.

    Ingests standards documents from files or strings, parses them
    into individual rules, embeds them, and stores them in ChromaDB
    scoped to specific repositories.

    Attributes:
        embedding_service: Service for generating embeddings.
        vector_store: ChromaDB client.
        parser: Markdown standards parser.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreClient,
    ) -> None:
        """Initialize the standards loader.

        Args:
            embedding_service: Configured embedding service.
            vector_store: ChromaDB client.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.parser = MarkdownStandardsParser()

    async def load_from_file(
        self,
        file_path: Path | str,
        repo_full_name: str,
        default_category: str = "general",
    ) -> int:
        """Load standards from a Markdown file.

        Args:
            file_path: Path to the standards document.
            repo_full_name: Repository these standards apply to.
            default_category: Default category for rules without one.

        Returns:
            Number of rules loaded.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning("standards file not found", path=str(path))
            return 0

        content = path.read_text(encoding="utf-8")
        return await self.load_from_text(content, repo_full_name, default_category)

    async def load_from_text(
        self,
        content: str,
        repo_full_name: str,
        default_category: str = "general",
    ) -> int:
        """Load standards from a text string.

        Args:
            content: The standards document text.
            repo_full_name: Repository these standards apply to.
            default_category: Default category for rules without one.

        Returns:
            Number of rules loaded.
        """
        rules = self.parser.parse(content, default_category)

        if not rules:
            logger.warning("no rules parsed from standards document")
            return 0

        return await self._store_rules(rules, repo_full_name)

    async def load_default_standards(self, repo_full_name: str) -> int:
        """Load built-in default coding standards.

        Provides a baseline set of universal best practices when no
        custom standards document is configured.

        Args:
            repo_full_name: Repository to scope standards to.

        Returns:
            Number of rules loaded.
        """
        default_rules = self._get_default_rules()
        return await self._store_rules(default_rules, repo_full_name)

    async def _store_rules(
        self, rules: list[CodingRule], repo_full_name: str
    ) -> int:
        """Embed and store rules in the vector store.

        Args:
            rules: Parsed coding rules to store.
            repo_full_name: Repository scope.

        Returns:
            Number of rules stored.
        """
        texts = [rule.full_text for rule in rules]
        embeddings = await self.embedding_service.embed_many(texts)

        ids = [
            hashlib.sha256(f"{repo_full_name}:{rule.id}".encode()).hexdigest()[:16]
            for rule in rules
        ]

        metadatas = []
        for rule in rules:
            meta = rule.metadata
            meta["repo"] = repo_full_name
            metadatas.append(meta)

        self.vector_store.add_documents(
            collection_name=VectorStoreClient.COLLECTION_STANDARDS,
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "standards loaded",
            repo=repo_full_name,
            rules_count=len(rules),
        )
        return len(rules)

    @staticmethod
    def _get_default_rules() -> list[CodingRule]:
        """Get built-in default coding standards."""
        return [
            CodingRule(
                id="SEC-001",
                title="No hardcoded secrets or credentials",
                description="Never hardcode API keys, passwords, tokens, or other secrets in source code. Use environment variables or secret management services.",
                category="security",
                severity="critical",
            ),
            CodingRule(
                id="SEC-002",
                title="Validate and sanitize all external input",
                description="All data from external sources (user input, API responses, file contents) must be validated before use. Use parameterized queries for database operations.",
                category="security",
                severity="critical",
            ),
            CodingRule(
                id="ERR-001",
                title="Handle errors explicitly",
                description="Never silently swallow exceptions. Log errors with context, handle them appropriately, or propagate them. Avoid bare except clauses.",
                category="best_practice",
                severity="high",
            ),
            CodingRule(
                id="DOC-001",
                title="Document public interfaces",
                description="All public functions, classes, and modules should have docstrings explaining purpose, parameters, return values, and exceptions.",
                category="documentation",
                severity="medium",
            ),
            CodingRule(
                id="PERF-001",
                title="Avoid N+1 queries and unnecessary loops",
                description="Database queries inside loops indicate an N+1 problem. Use batch fetching, joins, or eager loading instead.",
                category="performance",
                severity="high",
            ),
            CodingRule(
                id="MAINT-001",
                title="Keep functions focused and small",
                description="Functions should do one thing. If a function exceeds ~50 lines or has deeply nested logic, consider breaking it into smaller, well-named functions.",
                category="maintainability",
                severity="medium",
            ),
            CodingRule(
                id="TEST-001",
                title="New features should include tests",
                description="New functionality should be accompanied by unit tests covering the happy path and key edge cases. Bug fixes should include a regression test.",
                category="testing",
                severity="medium",
            ),
            CodingRule(
                id="STYLE-001",
                title="Use consistent naming conventions",
                description="Follow the language's naming conventions: snake_case for Python, camelCase for JS/TS. Be descriptive and avoid single-letter names except in small loops.",
                category="style",
                severity="low",
            ),
        ]
