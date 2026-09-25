"""The Acme HR MCP server instance: tools and resources register on `mcp`."""

import logging
import typing as t
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from pydantic import AnyHttpUrl

from src.auth.jwt_handler import AuthTokenHandler
from src.config import settings

logger = logging.getLogger(__name__)

# Memory: what the tools save while the server runs.
store: dict[str, dict[str, t.Any]] = {}


@asynccontextmanager
async def lifespan(app: MCPServer) -> AsyncIterator[None]:
    logger.info("Starting Acme HR MCP server...")
    store.clear()
    try:
        yield
    finally:
        store.clear()
        logger.info("Stopping Acme HR MCP server...")


mcp = MCPServer(
    settings.MCP_NAME,
    instructions="Acme AI's HR system: vacation, parking, global policies for the signed-in employee.",
    token_verifier=AuthTokenHandler(),
    lifespan=lifespan,
    auth=AuthSettings(
        issuer_url=AnyHttpUrl(settings.MCP_URL),
        resource_server_url=AnyHttpUrl(settings.MCP_URL),
        validate_token_resource=False,
        required_scopes=["user"],
    ),
)
