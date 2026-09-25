"""Entry point for the Acme HR MCP server."""

import src.resources  # noqa: F401 — registers resources on the mcp instance
import src.tools  # noqa: F401 — registers tools on the mcp instance
from src._app import mcp
from src.config import settings

if __name__ == "__main__":
    mcp.run(
        "streamable-http",
        host=settings.MCP_HOST,
        port=settings.MCP_PORT,
        stateless_http=settings.stateless_http,
        json_response=settings.json_response,
    )
