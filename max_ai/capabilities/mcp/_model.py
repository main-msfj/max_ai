"""Configuration objects for MCP server connections."""

from __future__ import annotations

import os
import typing as t

from pydantic import BaseModel, Field, TypeAdapter, field_validator

from ...errors.mcp import MCPServerConfigError
from ...types.tools import ToolApprovalMode

TransportProtocol = t.Literal["stdio", "sse", "streamable-http"]
HTTPTransportProtocol = t.Literal["sse", "streamable-http"]
def _read_env(server_id: str, name: str) -> str:
    """Perform the internal ``read env`` operation.

Parameters
----------
server_id : str
    Value supplied for ``server_id``.
name : str
    Value supplied for ``name``."""
    value = os.getenv(name)
    if not value:
        raise MCPServerConfigError(f"MCP server {server_id!r} needs env var {name}.")
    return value


class MCPServerConfig(BaseModel):
    """Base configuration for one MCP server.

    Secrets never live in a serialized config: ``*_env`` fields name the
    env vars that hold them, resolved when the connection opens.
    """

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
        """Perform the internal ``server id required`` operation for ``MCPServerConfig``.

Parameters
----------
value : str
    Value supplied for ``value``."""
        if not value:
            raise MCPServerConfigError("server_id cannot be empty.")
        return value

    @property
    def effective_resource_approval_mode(self) -> ToolApprovalMode:
        """Perform the ``effective resource approval mode`` operation for ``MCPServerConfig``."""
        return self.resource_approval_mode or self.approval_mode


class StdioMCPServerConfig(MCPServerConfig):
    """Configuration for an MCP server launched over stdio."""

    transport_protocol: t.Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)
    env_from: dict[str, str] = Field(
        default_factory=dict,
        description="Child process env var → host env var holding its (secret) value.",
    )

    @property
    def process_env(self) -> dict[str, str]:
        """The child's env with ``env_from`` secrets resolved (call when connecting)."""
        env = dict(self.env)
        for child, name in self.env_from.items():
            env[child] = _read_env(self.server_id, name)
        return env

    @field_validator("command")
    @classmethod
    def _command_required(cls, value: str) -> str:
        """Perform the internal ``command required`` operation for ``StdioMCPServerConfig``.

Parameters
----------
value : str
    Value supplied for ``value``."""
        if not value:
            raise MCPServerConfigError("command cannot be empty.")
        return value


class HTTPServerConfig(MCPServerConfig):
    """Configuration for an MCP server exposed over SSE or streamable HTTP."""

    transport_protocol: HTTPTransportProtocol = "streamable-http"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    headers_env: dict[str, str] = Field(
        default_factory=dict, description="Header name → env var holding its (secret) value.",
    )
    token_env: str | None = Field(default=None, description="Env var holding the bearer token.")
    # In-code only: never dumped, and serialize_mcp_servers refuses it.
    token: str | None = Field(default=None, exclude=True, repr=False)

    @field_validator("url")
    @classmethod
    def _url_required(cls, value: str) -> str:
        """Perform the internal ``url required`` operation for ``HTTPServerConfig``.

Parameters
----------
value : str
    Value supplied for ``value``."""
        if not value:
            raise MCPServerConfigError("url cannot be empty.")
        return value

    @property
    def request_headers(self) -> dict[str, str]:
        """Headers with secrets resolved (call when connecting)."""
        headers = dict(self.headers)
        for header, name in self.headers_env.items():
            headers[header] = _read_env(self.server_id, name)
        token = self.token or (_read_env(self.server_id, self.token_env) if self.token_env else None)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


_server_list_adapter = TypeAdapter(list[StdioMCPServerConfig | HTTPServerConfig])


def ensure_serializable(config: MCPServerConfig) -> None:
    """Refuse configs whose secret would be lost or leaked when stored."""
    if isinstance(config, HTTPServerConfig) and config.token:
        raise MCPServerConfigError(
            f"MCP server {config.server_id!r} has a literal token: use token_env "
            "so the stored config names the env var instead of the secret."
        )


def serialize_mcp_servers(
    configs: t.Sequence[StdioMCPServerConfig | HTTPServerConfig],
) -> str:
    """Serialize a mixed server list as JSON, including its transport types."""
    for config in configs:
        ensure_serializable(config)
    return _server_list_adapter.dump_json(list(configs)).decode("utf-8")


def deserialize_mcp_servers(payload: str | bytes) -> list[MCPServerConfig]:
    """Restore concrete server configurations from JSON."""
    return _server_list_adapter.validate_json(payload)
