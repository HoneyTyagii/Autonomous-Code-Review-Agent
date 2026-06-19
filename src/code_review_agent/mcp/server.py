"""MCP server implementation exposing review agent tools.

Implements the Model Context Protocol server that exposes code review
capabilities as tools for LLM-powered clients. Provides tools for:
- Reviewing code diffs
- Querying coding standards
- Searching past reviews
- Running security scans
- Generating patches
"""

import json
from typing import Any

from code_review_agent.mcp.tools import TOOL_DEFINITIONS, execute_tool
from code_review_agent.logging import get_logger

logger = get_logger("mcp_server")

# MCP Protocol version
MCP_VERSION = "2024-11-05"


class MCPServer:
    """Model Context Protocol server for the code review agent.

    Handles MCP JSON-RPC messages and dispatches tool calls to
    the appropriate handlers.

    Attributes:
        name: Server display name.
        version: Server version string.
    """

    def __init__(self) -> None:
        """Initialize the MCP server."""
        self.name = "code-review-agent"
        self.version = "0.1.0"
        self._initialized = False

    async def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Handle an incoming MCP JSON-RPC message.

        Routes messages to appropriate handlers based on the method field.

        Args:
            message: The JSON-RPC message dict.

        Returns:
            The JSON-RPC response dict.
        """
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params", {})

        logger.debug("mcp message received", method=method)

        try:
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "initialized":
                self._initialized = True
                return {}  # Notification, no response
            elif method == "tools/list":
                result = await self._handle_tools_list()
            elif method == "tools/call":
                result = await self._handle_tools_call(params)
            elif method == "resources/list":
                result = await self._handle_resources_list()
            elif method == "resources/read":
                result = await self._handle_resources_read(params)
            elif method == "ping":
                result = {}
            else:
                return self._error_response(
                    msg_id, -32601, f"Method not found: {method}"
                )

            return self._success_response(msg_id, result)

        except Exception as e:
            logger.error("mcp handler error", method=method, error=str(e))
            return self._error_response(msg_id, -32603, str(e))

    async def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle the initialize request.

        Args:
            params: Initialize parameters from client.

        Returns:
            Server capabilities and info.
        """
        logger.info(
            "mcp client connecting",
            client=params.get("clientInfo", {}).get("name", "unknown"),
        )

        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
        }

    async def _handle_tools_list(self) -> dict[str, Any]:
        """Handle tools/list request.

        Returns:
            List of available tool definitions.
        """
        return {"tools": TOOL_DEFINITIONS}

    async def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request.

        Args:
            params: Tool call parameters including name and arguments.

        Returns:
            Tool execution result.
        """
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        logger.info("mcp tool call", tool=tool_name)

        result = await execute_tool(tool_name, arguments)
        return result

    async def _handle_resources_list(self) -> dict[str, Any]:
        """Handle resources/list request.

        Returns:
            List of available resources.
        """
        return {
            "resources": [
                {
                    "uri": "review-agent://standards/default",
                    "name": "Default Coding Standards",
                    "description": "Built-in coding standards enforced by the review agent.",
                    "mimeType": "text/markdown",
                },
                {
                    "uri": "review-agent://config/current",
                    "name": "Current Configuration",
                    "description": "Active review agent configuration (sanitized).",
                    "mimeType": "application/json",
                },
            ]
        }

    async def _handle_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle resources/read request.

        Args:
            params: Resource read parameters including URI.

        Returns:
            Resource content.
        """
        uri = params.get("uri", "")

        if uri == "review-agent://standards/default":
            from code_review_agent.rag.standards_loader import StandardsLoader
            rules = StandardsLoader._get_default_rules()
            content = "\n\n".join(
                f"## {r.id}: {r.title}\n{r.description}"
                for r in rules
            )
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": content,
                }]
            }

        elif uri == "review-agent://config/current":
            from code_review_agent.config import get_settings
            settings = get_settings()
            safe_config = {
                "llm_provider": settings.llm_provider.value,
                "llm_model": settings.openai_model if settings.llm_provider.value == "openai" else settings.anthropic_model,
                "embedding_provider": settings.embedding_provider.value,
                "security_scan_enabled": settings.enable_security_scan,
                "security_tools": settings.security_tools_list,
            }
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(safe_config, indent=2),
                }]
            }

        return {"contents": []}

    @staticmethod
    def _success_response(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        """Build a JSON-RPC success response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result,
        }

    @staticmethod
    def _error_response(msg_id: Any, code: int, message: str) -> dict[str, Any]:
        """Build a JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }


def create_mcp_server() -> MCPServer:
    """Create and return a configured MCP server instance.

    Returns:
        Ready-to-use MCPServer.
    """
    return MCPServer()
