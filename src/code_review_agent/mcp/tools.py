"""MCP tool definitions and execution handlers.

Defines the tools exposed via the MCP protocol and routes tool
calls to the appropriate internal handlers.
"""

from typing import Any

from code_review_agent.logging import get_logger

logger = get_logger("mcp_tools")

# Tool definitions following the MCP tool schema
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "review_diff",
        "description": (
            "Review a code diff and provide feedback. Accepts a unified diff "
            "string and returns structured review comments with severity, "
            "category, and suggested fixes."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["diff", "language"],
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Unified diff content to review.",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language of the code.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional additional context about the changes.",
                },
            },
        },
    },
    {
        "name": "check_security",
        "description": (
            "Run security analysis on code. Detects vulnerabilities, "
            "hardcoded secrets, injection flaws, and unsafe patterns."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["code", "language"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Source code to scan for security issues.",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language of the code.",
                },
            },
        },
    },
    {
        "name": "query_standards",
        "description": (
            "Query coding standards relevant to a piece of code. Returns "
            "applicable rules and best practices from the configured standards."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Code or description to find relevant standards for.",
                },
                "repo": {
                    "type": "string",
                    "description": "Optional repository name to scope standards.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum standards to return.",
                    "default": 5,
                },
            },
        },
    },
    {
        "name": "search_past_reviews",
        "description": (
            "Search past review comments for similar code patterns. "
            "Helps understand how similar code was reviewed before."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["code_snippet"],
            "properties": {
                "code_snippet": {
                    "type": "string",
                    "description": "Code snippet to find similar past reviews for.",
                },
                "repo": {
                    "type": "string",
                    "description": "Optional repository name to scope search.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return.",
                    "default": 5,
                },
            },
        },
    },
    {
        "name": "generate_patch",
        "description": (
            "Generate a fix patch for a code issue. Given the original code "
            "and a description of the problem, produces a unified diff patch."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["code", "issue_description", "file_path"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The original code containing the issue.",
                },
                "issue_description": {
                    "type": "string",
                    "description": "Description of the issue to fix.",
                },
                "file_path": {
                    "type": "string",
                    "description": "File path for the patch header.",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language of the code.",
                },
            },
        },
    },
    {
        "name": "predict_bugs",
        "description": (
            "Analyze a diff for bug risk using heuristic signals. "
            "Returns risk score, identified signals, and recommendations."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["diff"],
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Unified diff to analyze for bug risk.",
                },
            },
        },
    },
]


async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute an MCP tool by name.

    Routes tool calls to the appropriate handler function and
    returns results in MCP content format.

    Args:
        name: Tool name to execute.
        arguments: Tool arguments dict.

    Returns:
        MCP tool result with content array.
    """
    handlers = {
        "review_diff": _handle_review_diff,
        "check_security": _handle_check_security,
        "query_standards": _handle_query_standards,
        "search_past_reviews": _handle_search_past_reviews,
        "generate_patch": _handle_generate_patch,
        "predict_bugs": _handle_predict_bugs,
    }

    handler = handlers.get(name)
    if not handler:
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
            "isError": True,
        }

    try:
        result = await handler(arguments)
        return {"content": [{"type": "text", "text": result}]}
    except Exception as e:
        logger.error("tool execution failed", tool=name, error=str(e))
        return {
            "content": [{"type": "text", "text": f"Error: {str(e)}"}],
            "isError": True,
        }


async def _handle_review_diff(arguments: dict[str, Any]) -> str:
    """Handle review_diff tool call.

    Reviews a code diff using the LLM review engine.

    Args:
        arguments: Tool arguments with diff and language.

    Returns:
        JSON string with review results.
    """
    import json
    from code_review_agent.llm.factory import create_llm_provider
    from code_review_agent.llm.base import Message, MessageRole
    from code_review_agent.core.prompts import SYSTEM_PROMPT

    diff = arguments["diff"]
    language = arguments["language"]
    context = arguments.get("context", "")

    llm = create_llm_provider()

    user_content = f"Language: {language}\n"
    if context:
        user_content += f"Context: {context}\n"
    user_content += f"\nDiff to review:\n```diff\n{diff}\n```\n"
    user_content += "\nProvide your review as a JSON object with: summary, decision, and comments array."

    response = await llm.generate(
        messages=[
            Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            Message(role=MessageRole.USER, content=user_content),
        ],
        temperature=0.1,
    )

    return response.content


async def _handle_check_security(arguments: dict[str, Any]) -> str:
    """Handle check_security tool call.

    Runs regex-based secret detection on the provided code.

    Args:
        arguments: Tool arguments with code and language.

    Returns:
        JSON string with security findings.
    """
    import json
    from code_review_agent.analysis.security_scanner import SecurityScanner, SecurityScanResult
    from code_review_agent.sandbox.docker_sandbox import DockerSandbox

    code = arguments["code"]
    language = arguments["language"]

    # Run the in-process secret scanner (no Docker required)
    source_files = {"code." + _lang_to_ext(language): code}

    try:
        sandbox = DockerSandbox()
        scanner = SecurityScanner(sandbox=sandbox)
    except Exception:
        # Docker not available, run secrets-only scan
        scanner = SecurityScanner.__new__(SecurityScanner)
        scanner.enabled_tools = ["secrets"]
        scanner.sandbox = None

    findings = scanner._scan_secrets(source_files)

    results = [
        {
            "line": f.line_number,
            "severity": f.severity,
            "title": f.title,
            "description": f.description,
            "remediation": f.remediation,
        }
        for f in findings
    ]

    return json.dumps({"findings": results, "count": len(results)}, indent=2)


async def _handle_query_standards(arguments: dict[str, Any]) -> str:
    """Handle query_standards tool call.

    Searches coding standards in the vector store.

    Args:
        arguments: Tool arguments with query.

    Returns:
        Matching standards as formatted text.
    """
    import json
    from code_review_agent.rag.embeddings import EmbeddingService
    from code_review_agent.rag.vector_store import VectorStoreClient

    query = arguments["query"]
    repo = arguments.get("repo", "")
    limit = arguments.get("limit", 5)

    try:
        embedding_service = EmbeddingService()
        vector_store = VectorStoreClient()

        query_embedding = await embedding_service.embed(query)
        where = {"repo": repo} if repo else None

        results = vector_store.query(
            collection_name=VectorStoreClient.COLLECTION_STANDARDS,
            query_embedding=query_embedding,
            n_results=limit,
            where=where,
        )

        standards = [
            {
                "rule_id": r.metadata.get("rule_id", ""),
                "title": r.metadata.get("title", ""),
                "category": r.metadata.get("category", ""),
                "content": r.content[:300],
                "relevance": round(r.score, 3),
            }
            for r in results
        ]

        return json.dumps({"standards": standards}, indent=2)

    except Exception as e:
        # Fall back to default standards
        from code_review_agent.rag.standards_loader import StandardsLoader
        rules = StandardsLoader._get_default_rules()
        standards = [
            {"rule_id": r.id, "title": r.title, "category": r.category, "content": r.description}
            for r in rules[:limit]
        ]
        return json.dumps({"standards": standards, "source": "defaults"}, indent=2)


async def _handle_search_past_reviews(arguments: dict[str, Any]) -> str:
    """Handle search_past_reviews tool call.

    Searches past review comments by semantic similarity.

    Args:
        arguments: Tool arguments with code_snippet.

    Returns:
        Similar past reviews as JSON.
    """
    import json
    from code_review_agent.rag.embeddings import EmbeddingService
    from code_review_agent.rag.vector_store import VectorStoreClient

    code_snippet = arguments["code_snippet"]
    repo = arguments.get("repo", "")
    limit = arguments.get("limit", 5)

    try:
        embedding_service = EmbeddingService()
        vector_store = VectorStoreClient()

        query_embedding = await embedding_service.embed(code_snippet)
        where = {"repo": repo} if repo else None

        results = vector_store.query(
            collection_name=VectorStoreClient.COLLECTION_REVIEWS,
            query_embedding=query_embedding,
            n_results=limit,
            where=where,
        )

        reviews = [
            {
                "comment": r.content[:300],
                "severity": r.metadata.get("severity", ""),
                "category": r.metadata.get("category", ""),
                "file_path": r.metadata.get("file_path", ""),
                "relevance": round(r.score, 3),
            }
            for r in results
        ]

        return json.dumps({"past_reviews": reviews}, indent=2)

    except Exception:
        return json.dumps({"past_reviews": [], "note": "No review history available."})


async def _handle_generate_patch(arguments: dict[str, Any]) -> str:
    """Handle generate_patch tool call.

    Generates a fix patch using the LLM.

    Args:
        arguments: Tool arguments with code, issue description, and file path.

    Returns:
        Unified diff patch string.
    """
    from code_review_agent.llm.factory import create_llm_provider
    from code_review_agent.llm.base import Message, MessageRole
    from code_review_agent.core.patch_generator import PATCH_SYSTEM_PROMPT

    code = arguments["code"]
    issue = arguments["issue_description"]
    file_path = arguments["file_path"]

    llm = create_llm_provider()

    user_content = (
        f"File: {file_path}\n"
        f"Issue: {issue}\n\n"
        f"Original code:\n```\n{code}\n```\n\n"
        f"Generate the corrected code that fixes this issue. Output ONLY the fixed code."
    )

    response = await llm.generate(
        messages=[
            Message(role=MessageRole.SYSTEM, content=PATCH_SYSTEM_PROMPT),
            Message(role=MessageRole.USER, content=user_content),
        ],
        temperature=0.0,
        max_tokens=2048,
    )

    fixed_code = response.content.strip()
    if fixed_code.startswith("```"):
        lines = fixed_code.split("\n")
        fixed_code = "\n".join(lines[1:-1]) if len(lines) > 2 else fixed_code

    # Generate diff
    import difflib
    original_lines = code.splitlines(keepends=True)
    fixed_lines = fixed_code.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines, fixed_lines,
        fromfile=f"a/{file_path}", tofile=f"b/{file_path}",
    )

    return "".join(diff) or "(no changes needed)"


async def _handle_predict_bugs(arguments: dict[str, Any]) -> str:
    """Handle predict_bugs tool call.

    Runs bug prediction heuristics on a diff.

    Args:
        arguments: Tool arguments with diff.

    Returns:
        Bug prediction results as JSON.
    """
    import json
    from code_review_agent.github.diff_parser import parse_patch, parse_github_files
    from code_review_agent.analysis.bug_predictor import BugPredictor

    diff = arguments["diff"]

    # Parse the diff into a PullRequestDiff
    # Create a minimal file data structure
    files_data = [{
        "filename": "file",
        "status": "modified",
        "additions": diff.count("\n+"),
        "deletions": diff.count("\n-"),
        "patch": diff,
    }]
    pr_diff = parse_github_files(files_data)

    # Run prediction
    predictor = BugPredictor()
    prediction = predictor.predict(pr_diff)

    result = {
        "risk_score": round(prediction.overall_risk_score, 3),
        "risk_level": prediction.risk_level.value,
        "signals": [
            {
                "type": s.signal_type,
                "risk_level": s.risk_level.value,
                "description": s.description,
                "file": s.file_path,
                "confidence": round(s.confidence, 2),
            }
            for s in prediction.signals
        ],
        "recommendations": prediction.recommendations,
    }

    return json.dumps(result, indent=2)


def _lang_to_ext(language: str) -> str:
    """Map language name to file extension."""
    ext_map = {
        "python": "py",
        "javascript": "js",
        "typescript": "ts",
        "go": "go",
        "rust": "rs",
        "java": "java",
        "ruby": "rb",
    }
    return ext_map.get(language.lower(), "txt")
