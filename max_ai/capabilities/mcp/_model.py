"""Configuration objects for MCP server connections."""

from __future__ import annotations

import typing as t

from pydantic import BaseModel, Field, TypeAdapter, field_validator

from ...errors.mcp import MCPServerConfigError
from ...types.tools import ToolApprovalMode

TransportProtocol = t.Literal["stdio", "sse", "streamable-http"]
HTTPTransportProtocol = t.Literal["sse", "streamable-http"]
class MCPServerConfig(BaseModel):
    """Base configuration for one MCP server."""

    server_id: str
    transport_protocol: TransportProtocol = "streamable-http"
    env: dict[str, str] = Field(default_factory=dict)
    approval_mode: ToolApprovalMode = ToolApprovalMode.ASK_APPROVED
    resource_approval_mode: ToolApprovalMode | None = None
    tool_approval_modes: dict[str, ToolApprovalMode] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=240.0, gt=0)

    @field_validator("server_id")
    @classmethod
    def _server_id_required(cls, value: str) -> str:
        if not value:
            raise MCPServerConfigError("server_id cannot be empty.")
        return value

    @property
    def effective_resource_approval_mode(self) -> ToolApprovalMode:
        return self.resource_approval_mode or self.approval_mode


class StdioMCPServerConfig(MCPServerConfig):
    """Configuration for an MCP server launched over stdio."""

    transport_protocol: t.Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)

    @field_validator("command")
    @classmethod
    def _command_required(cls, value: str) -> str:
        if not value:
            raise MCPServerConfigError("command cannot be empty.")
        return value


class HTTPServerConfig(MCPServerConfig):
    """Configuration for an MCP server exposed over SSE or streamable HTTP."""

    transport_protocol: HTTPTransportProtocol = "streamable-http"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    token: str | None = None

    @field_validator("url")
    @classmethod
    def _url_required(cls, value: str) -> str:
        if not value:
            raise MCPServerConfigError("url cannot be empty.")
        return value

    @property
    def request_headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


_server_list_adapter = TypeAdapter(list[StdioMCPServerConfig | HTTPServerConfig])


def serialize_mcp_servers(
    configs: t.Sequence[StdioMCPServerConfig | HTTPServerConfig],
) -> str:
    """Serialize a mixed server list as JSON, including its transport types."""
    return _server_list_adapter.dump_json(list(configs)).decode("utf-8")


def deserialize_mcp_servers(payload: str | bytes) -> list[MCPServerConfig]:
    """Restore concrete server configurations from JSON."""
    return _server_list_adapter.validate_json(payload)
