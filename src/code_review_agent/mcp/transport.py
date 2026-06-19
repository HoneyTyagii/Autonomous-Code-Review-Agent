"""MCP transport layer for stdio communication.

Implements the stdio-based JSON-RPC transport that MCP clients
(like IDEs) use to communicate with the server process.
"""

import json
import sys
from typing import Any

from code_review_agent.mcp.server import MCPServer, create_mcp_server
from code_review_agent.logging import get_logger, setup_logging

logger = get_logger("mcp_transport")


async def run_stdio_server() -> None:
    """Run the MCP server using stdio transport.

    Reads JSON-RPC messages from stdin and writes responses to stdout.
    This is the standard transport for MCP servers launched as
    subprocess by IDE clients.
    """
    # Log to stderr to keep stdout clean for protocol messages
    setup_logging(log_level="WARNING", json_output=True)

    server = create_mcp_server()

    logger.info("mcp stdio server starting")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _write_error(None, -32700, "Parse error")
            continue

        response = await server.handle_message(message)

        if response:  # Notifications don't get responses
            _write_response(response)

    logger.info("mcp stdio server stopped")


def _write_response(response: dict[str, Any]) -> None:
    """Write a JSON-RPC response to stdout.

    Args:
        response: Response dict to serialize and write.
    """
    output = json.dumps(response)
    sys.stdout.write(output + "\n")
    sys.stdout.flush()


def _write_error(msg_id: Any, code: int, message: str) -> None:
    """Write a JSON-RPC error to stdout.

    Args:
        msg_id: Request ID (or None).
        code: Error code.
        message: Error message.
    """
    error_response = {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": code, "message": message},
    }
    _write_response(error_response)


def main() -> None:
    """Entry point for running the MCP server as a standalone process.

    Can be invoked with:
        python -m code_review_agent.mcp.transport
    """
    import asyncio
    asyncio.run(run_stdio_server())


if __name__ == "__main__":
    main()
