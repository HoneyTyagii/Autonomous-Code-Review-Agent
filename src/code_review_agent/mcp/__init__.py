"""Model Context Protocol (MCP) server for extensible tool access.

Exposes the code review agent's capabilities as MCP tools that can
be consumed by LLM-powered IDEs and other MCP clients.
"""

from code_review_agent.mcp.server import create_mcp_server

__all__ = ["create_mcp_server"]
