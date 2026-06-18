"""Tree-sitter AST parsing for multi-language code understanding.

Provides structural analysis of source code using Tree-sitter grammars.
Extracts functions, classes, imports, and other symbols to give the
review engine deeper understanding of code changes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import tree_sitter
from tree_sitter import Language, Parser, Node

from code_review_agent.logging import get_logger

logger = get_logger("ast_parser")


class SymbolKind(str, Enum):
    """Types of code symbols that can be extracted."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    IMPORT = "import"
    VARIABLE = "variable"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    DECORATOR = "decorator"
    CONSTANT = "constant"


@dataclass
class CodeSymbol:
    """A code symbol extracted from AST analysis.

    Attributes:
        name: The symbol's identifier name.
        kind: The type of symbol (function, class, etc.).
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (1-indexed).
        docstring: Associated documentation string, if any.
        parameters: Function/method parameters.
        return_type: Return type annotation, if present.
        decorators: List of decorator names applied to this symbol.
        parent: Name of the parent class (for methods).
        is_exported: Whether the symbol is exported/public.
    """

    name: str
    kind: SymbolKind
    start_line: int
    end_line: int
    docstring: str | None = None
    parameters: list[str] = field(default_factory=list)
    return_type: str | None = None
    decorators: list[str] = field(default_factory=list)
    parent: str | None = None
    is_exported: bool = False


@dataclass
class ASTAnalysis:
    """Result of AST analysis on a source file.

    Attributes:
        language: The programming language analyzed.
        symbols: All extracted code symbols.
        imports: List of imported modules/packages.
        complexity_hints: Indicators of code complexity.
        errors: Any parse errors encountered.
    """

    language: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    complexity_hints: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def functions(self) -> list[CodeSymbol]:
        """Get all function symbols."""
        return [s for s in self.symbols if s.kind == SymbolKind.FUNCTION]

    @property
    def classes(self) -> list[CodeSymbol]:
        """Get all class symbols."""
        return [s for s in self.symbols if s.kind == SymbolKind.CLASS]

    @property
    def methods(self) -> list[CodeSymbol]:
        """Get all method symbols."""
        return [s for s in self.symbols if s.kind == SymbolKind.METHOD]

    def get_symbol_at_line(self, line: int) -> CodeSymbol | None:
        """Find the symbol that contains a given line number.

        Args:
            line: The line number to look up (1-indexed).

        Returns:
            The innermost symbol containing that line, or None.
        """
        candidates = [
            s for s in self.symbols
            if s.start_line <= line <= s.end_line
        ]
        # Return the most specific (innermost) symbol
        if candidates:
            return min(candidates, key=lambda s: s.end_line - s.start_line)
        return None

    def get_context_for_lines(self, start: int, end: int) -> list[CodeSymbol]:
        """Get all symbols that overlap with a line range.

        Args:
            start: Start line number (1-indexed).
            end: End line number (1-indexed).

        Returns:
            List of symbols overlapping the range.
        """
        return [
            s for s in self.symbols
            if s.start_line <= end and s.end_line >= start
        ]


class TreeSitterParser:
    """Multi-language AST parser using Tree-sitter.

    Lazily initializes language grammars and provides a unified interface
    for parsing Python, JavaScript, and TypeScript source code.

    Attributes:
        _languages: Cached language grammar objects.
        _parsers: Cached parser instances per language.
    """

    # Supported languages and their tree-sitter package references
    SUPPORTED_LANGUAGES = {
        "python": "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
    }

    def __init__(self) -> None:
        """Initialize the Tree-sitter parser with empty caches."""
        self._languages: dict[str, Language] = {}
        self._parsers: dict[str, Parser] = {}

    def _get_parser(self, language: str) -> Parser | None:
        """Get or create a parser for the specified language.

        Args:
            language: The programming language name.

        Returns:
            Configured Parser instance, or None if language unsupported.
        """
        if language in self._parsers:
            return self._parsers[language]

        if language not in self.SUPPORTED_LANGUAGES:
            return None

        try:
            lang = self._load_language(language)
            if lang is None:
                return None

            parser = Parser()
            parser.language = lang
            self._parsers[language] = parser
            return parser

        except Exception as e:
            logger.error("failed to initialize parser", language=language, error=str(e))
            return None

    def _load_language(self, language: str) -> Language | None:
        """Load a Tree-sitter language grammar.

        Args:
            language: The language name to load.

        Returns:
            The Language object, or None on failure.
        """
        if language in self._languages:
            return self._languages[language]

        try:
            if language == "python":
                import tree_sitter_python
                lang = Language(tree_sitter_python.language())
            elif language == "javascript":
                import tree_sitter_javascript
                lang = Language(tree_sitter_javascript.language())
            elif language == "typescript":
                import tree_sitter_typescript
                lang = Language(tree_sitter_typescript.language_typescript())
            else:
                return None

            self._languages[language] = lang
            logger.debug("language loaded", language=language)
            return lang

        except ImportError as e:
            logger.warning(
                "tree-sitter language package not installed",
                language=language,
                error=str(e),
            )
            return None

    def parse(self, source_code: str, language: str) -> ASTAnalysis:
        """Parse source code and extract structural information.

        Args:
            source_code: The source code string to parse.
            language: The programming language of the source.

        Returns:
            ASTAnalysis with extracted symbols and metadata.
        """
        analysis = ASTAnalysis(language=language)

        parser = self._get_parser(language)
        if parser is None:
            analysis.errors.append(f"Unsupported language: {language}")
            return analysis

        try:
            tree = parser.parse(source_code.encode("utf-8"))
        except Exception as e:
            analysis.errors.append(f"Parse error: {str(e)}")
            return analysis

        root_node = tree.root_node

        # Check for parse errors
        if root_node.has_error:
            analysis.errors.append("Source contains syntax errors")

        # Extract symbols based on language
        if language == "python":
            self._extract_python_symbols(root_node, source_code, analysis)
        elif language in ("javascript", "typescript"):
            self._extract_js_ts_symbols(root_node, source_code, analysis)

        logger.debug(
            "ast parsed",
            language=language,
            symbols=len(analysis.symbols),
            imports=len(analysis.imports),
            errors=len(analysis.errors),
        )

        return analysis

    def _extract_python_symbols(
        self, root: Node, source: str, analysis: ASTAnalysis
    ) -> None:
        """Extract symbols from Python AST.

        Args:
            root: The root AST node.
            source: Original source code for text extraction.
            analysis: The analysis object to populate.
        """
        self._walk_python_node(root, source, analysis, parent_class=None)

    def _walk_python_node(
        self,
        node: Node,
        source: str,
        analysis: ASTAnalysis,
        parent_class: str | None,
    ) -> None:
        """Recursively walk Python AST nodes extracting symbols.

        Args:
            node: Current AST node.
            source: Original source code.
            analysis: Analysis object to populate.
            parent_class: Name of enclosing class, if any.
        """
        for child in node.children:
            if child.type == "import_statement" or child.type == "import_from_statement":
                import_text = self._node_text(child, source)
                analysis.imports.append(import_text)

            elif child.type == "function_definition":
                symbol = self._extract_python_function(child, source, parent_class)
                if symbol:
                    analysis.symbols.append(symbol)

            elif child.type == "class_definition":
                class_symbol = self._extract_python_class(child, source)
                if class_symbol:
                    analysis.symbols.append(class_symbol)
                    # Recurse into class body for methods
                    body = self._find_child_by_type(child, "block")
                    if body:
                        self._walk_python_node(body, source, analysis, class_symbol.name)

            elif child.type == "decorated_definition":
                # Handle decorated functions/classes
                for inner in child.children:
                    if inner.type == "function_definition":
                        symbol = self._extract_python_function(inner, source, parent_class)
                        if symbol:
                            # Extract decorators
                            symbol.decorators = self._extract_python_decorators(child, source)
                            analysis.symbols.append(symbol)
                    elif inner.type == "class_definition":
                        class_symbol = self._extract_python_class(inner, source)
                        if class_symbol:
                            class_symbol.decorators = self._extract_python_decorators(
                                child, source
                            )
                            analysis.symbols.append(class_symbol)
                            body = self._find_child_by_type(inner, "block")
                            if body:
                                self._walk_python_node(
                                    body, source, analysis, class_symbol.name
                                )

    def _extract_python_function(
        self, node: Node, source: str, parent_class: str | None
    ) -> CodeSymbol | None:
        """Extract a Python function/method symbol.

        Args:
            node: The function_definition AST node.
            source: Original source code.
            parent_class: Enclosing class name, if any.

        Returns:
            CodeSymbol for the function, or None if extraction fails.
        """
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None

        name = self._node_text(name_node, source)
        kind = SymbolKind.METHOD if parent_class else SymbolKind.FUNCTION

        # Extract parameters
        params_node = self._find_child_by_type(node, "parameters")
        parameters = []
        if params_node:
            for param in params_node.children:
                if param.type in ("identifier", "typed_parameter", "default_parameter"):
                    parameters.append(self._node_text(param, source))

        # Extract return type
        return_type = None
        ret_node = self._find_child_by_type(node, "type")
        if ret_node:
            return_type = self._node_text(ret_node, source)

        # Extract docstring
        docstring = self._extract_python_docstring(node, source)

        return CodeSymbol(
            name=name,
            kind=kind,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
            parent=parent_class,
            is_exported=not name.startswith("_"),
        )

    def _extract_python_class(self, node: Node, source: str) -> CodeSymbol | None:
        """Extract a Python class symbol.

        Args:
            node: The class_definition AST node.
            source: Original source code.

        Returns:
            CodeSymbol for the class, or None if extraction fails.
        """
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None

        name = self._node_text(name_node, source)
        docstring = self._extract_python_docstring(node, source)

        return CodeSymbol(
            name=name,
            kind=SymbolKind.CLASS,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=docstring,
            is_exported=not name.startswith("_"),
        )

    def _extract_python_decorators(self, node: Node, source: str) -> list[str]:
        """Extract decorator names from a decorated definition.

        Args:
            node: The decorated_definition AST node.
            source: Original source code.

        Returns:
            List of decorator name strings.
        """
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                dec_text = self._node_text(child, source)
                # Strip the @ prefix
                decorators.append(dec_text.lstrip("@").split("(")[0])
        return decorators

    def _extract_python_docstring(self, node: Node, source: str) -> str | None:
        """Extract docstring from a function or class definition.

        Args:
            node: The function/class definition AST node.
            source: Original source code.

        Returns:
            The docstring text, or None if no docstring found.
        """
        body = self._find_child_by_type(node, "block")
        if not body or not body.children:
            return None

        first_stmt = body.children[0]
        if first_stmt.type == "expression_statement":
            expr = first_stmt.children[0] if first_stmt.children else None
            if expr and expr.type == "string":
                text = self._node_text(expr, source)
                # Strip triple quotes
                return text.strip('"""').strip("'''").strip()

        return None

    def _extract_js_ts_symbols(
        self, root: Node, source: str, analysis: ASTAnalysis
    ) -> None:
        """Extract symbols from JavaScript/TypeScript AST.

        Args:
            root: The root AST node.
            source: Original source code.
            analysis: The analysis object to populate.
        """
        self._walk_js_ts_node(root, source, analysis, parent_class=None)

    def _walk_js_ts_node(
        self,
        node: Node,
        source: str,
        analysis: ASTAnalysis,
        parent_class: str | None,
    ) -> None:
        """Recursively walk JS/TS AST nodes extracting symbols.

        Args:
            node: Current AST node.
            source: Original source code.
            analysis: Analysis object to populate.
            parent_class: Name of enclosing class, if any.
        """
        for child in node.children:
            if child.type in ("import_statement", "import_declaration"):
                import_text = self._node_text(child, source)
                analysis.imports.append(import_text)

            elif child.type in ("function_declaration", "function"):
                symbol = self._extract_js_function(child, source, parent_class)
                if symbol:
                    analysis.symbols.append(symbol)

            elif child.type == "class_declaration":
                class_symbol = self._extract_js_class(child, source)
                if class_symbol:
                    analysis.symbols.append(class_symbol)
                    # Recurse into class body
                    body = self._find_child_by_type(child, "class_body")
                    if body:
                        self._walk_js_ts_node(body, source, analysis, class_symbol.name)

            elif child.type == "method_definition":
                symbol = self._extract_js_method(child, source, parent_class)
                if symbol:
                    analysis.symbols.append(symbol)

            elif child.type in ("lexical_declaration", "variable_declaration"):
                # Check for arrow functions: const foo = () => {}
                symbols = self._extract_js_variable_declarations(
                    child, source, parent_class
                )
                analysis.symbols.extend(symbols)

            elif child.type == "export_statement":
                # Recurse into exported declarations
                self._walk_js_ts_node(child, source, analysis, parent_class)

            elif child.type == "interface_declaration":
                symbol = self._extract_ts_interface(child, source)
                if symbol:
                    analysis.symbols.append(symbol)

            elif child.type == "type_alias_declaration":
                symbol = self._extract_ts_type_alias(child, source)
                if symbol:
                    analysis.symbols.append(symbol)

    def _extract_js_function(
        self, node: Node, source: str, parent_class: str | None
    ) -> CodeSymbol | None:
        """Extract a JS/TS function symbol.

        Args:
            node: The function_declaration AST node.
            source: Original source code.
            parent_class: Enclosing class name, if any.

        Returns:
            CodeSymbol for the function, or None.
        """
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None

        name = self._node_text(name_node, source)
        params_node = self._find_child_by_type(node, "formal_parameters")
        parameters = self._extract_js_params(params_node, source) if params_node else []

        return CodeSymbol(
            name=name,
            kind=SymbolKind.METHOD if parent_class else SymbolKind.FUNCTION,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            parameters=parameters,
            parent=parent_class,
            is_exported=self._is_js_exported(node),
        )

    def _extract_js_class(self, node: Node, source: str) -> CodeSymbol | None:
        """Extract a JS/TS class symbol.

        Args:
            node: The class_declaration AST node.
            source: Original source code.

        Returns:
            CodeSymbol for the class, or None.
        """
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            # Could be a type_identifier in TS
            name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return None

        name = self._node_text(name_node, source)

        return CodeSymbol(
            name=name,
            kind=SymbolKind.CLASS,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            is_exported=self._is_js_exported(node),
        )

    def _extract_js_method(
        self, node: Node, source: str, parent_class: str | None
    ) -> CodeSymbol | None:
        """Extract a JS/TS class method symbol.

        Args:
            node: The method_definition AST node.
            source: Original source code.
            parent_class: Enclosing class name.

        Returns:
            CodeSymbol for the method, or None.
        """
        name_node = self._find_child_by_type(node, "property_identifier")
        if not name_node:
            return None

        name = self._node_text(name_node, source)
        params_node = self._find_child_by_type(node, "formal_parameters")
        parameters = self._extract_js_params(params_node, source) if params_node else []

        return CodeSymbol(
            name=name,
            kind=SymbolKind.METHOD,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            parameters=parameters,
            parent=parent_class,
            is_exported=False,
        )

    def _extract_js_variable_declarations(
        self, node: Node, source: str, parent_class: str | None
    ) -> list[CodeSymbol]:
        """Extract arrow function declarations from variable statements.

        Identifies patterns like: const myFunc = (args) => { ... }

        Args:
            node: The lexical/variable declaration AST node.
            source: Original source code.
            parent_class: Enclosing class name, if any.

        Returns:
            List of extracted function symbols.
        """
        symbols = []

        for child in node.children:
            if child.type == "variable_declarator":
                name_node = self._find_child_by_type(child, "identifier")
                if not name_node:
                    continue

                # Check if the value is an arrow function
                value_node = None
                for inner in child.children:
                    if inner.type == "arrow_function":
                        value_node = inner
                        break

                if value_node:
                    name = self._node_text(name_node, source)
                    params_node = self._find_child_by_type(
                        value_node, "formal_parameters"
                    )
                    parameters = (
                        self._extract_js_params(params_node, source)
                        if params_node
                        else []
                    )

                    symbols.append(
                        CodeSymbol(
                            name=name,
                            kind=SymbolKind.FUNCTION,
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            parameters=parameters,
                            parent=parent_class,
                            is_exported=self._is_js_exported(node),
                        )
                    )

        return symbols

    def _extract_ts_interface(self, node: Node, source: str) -> CodeSymbol | None:
        """Extract a TypeScript interface symbol.

        Args:
            node: The interface_declaration AST node.
            source: Original source code.

        Returns:
            CodeSymbol for the interface, or None.
        """
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return None

        return CodeSymbol(
            name=self._node_text(name_node, source),
            kind=SymbolKind.INTERFACE,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            is_exported=self._is_js_exported(node),
        )

    def _extract_ts_type_alias(self, node: Node, source: str) -> CodeSymbol | None:
        """Extract a TypeScript type alias symbol.

        Args:
            node: The type_alias_declaration AST node.
            source: Original source code.

        Returns:
            CodeSymbol for the type alias, or None.
        """
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return None

        return CodeSymbol(
            name=self._node_text(name_node, source),
            kind=SymbolKind.TYPE_ALIAS,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            is_exported=self._is_js_exported(node),
        )

    def _extract_js_params(self, node: Node, source: str) -> list[str]:
        """Extract parameter names from a formal_parameters node.

        Args:
            node: The formal_parameters AST node.
            source: Original source code.

        Returns:
            List of parameter strings.
        """
        params = []
        for child in node.children:
            if child.type in (
                "identifier",
                "required_parameter",
                "optional_parameter",
                "rest_pattern",
            ):
                params.append(self._node_text(child, source))
        return params

    def _is_js_exported(self, node: Node) -> bool:
        """Check if a JS/TS node is exported.

        Args:
            node: The AST node to check.

        Returns:
            True if the node's parent is an export statement.
        """
        parent = node.parent
        return parent is not None and parent.type == "export_statement"

    @staticmethod
    def _find_child_by_type(node: Node, type_name: str) -> Node | None:
        """Find the first child node matching a type.

        Args:
            node: Parent node to search.
            type_name: The node type to find.

        Returns:
            The first matching child, or None.
        """
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    @staticmethod
    def _node_text(node: Node, source: str) -> str:
        """Get the source text for a node.

        Args:
            node: The AST node.
            source: The full source code string.

        Returns:
            The text content of the node.
        """
        return source[node.start_byte:node.end_byte]
