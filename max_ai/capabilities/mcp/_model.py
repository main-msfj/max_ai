"""Configuration objects for MCP server connections."""

from __future__ import annotations

import os
import typing as t

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

from ...errors.mcp import MCPServerConfigError
from ...types.tools import ToolApprovalMode

TransportProtocol = t.Literal["stdio", "sse", "streamable-http"]
HTTPTransportProtocol = t.Literal["sse", "streamable-http"]
class MCPSecretBase(BaseModel):
    """What an MCP server needs to know who we are. ``keys`` maps the name of
    an environment variable to its value. Only the names are stored; loading
    a stored config looks each name up in the environment again."""

    keys: dict[str, str | None] = Field(min_length=1, repr=False)

    @field_validator("keys", mode="before")
    @classmethod
    def _from_names(cls, value: t.Any) -> t.Any:
        """A stored config brings only the names: their values are read here."""
        if isinstance(value, (list, tuple)):
            names = t.cast("t.Sequence[str]", value)
            return {name: os.getenv(name) for name in names}
        return value

    @field_serializer("keys")
    def _only_names(self, keys: dict[str, str | None]) -> list[str]:
        return list(keys)

    def __repr_args__(self) -> t.Iterable[tuple[str | None, t.Any]]:
        yield from super().__repr_args__()
        yield "keys", list(self.keys)

    def values(self, server_id: str) -> dict[str, str]:
        """The values, when connecting; an empty one is named in the error."""
        missing = [name for name, value in self.keys.items() if not value]
        if missing:
            raise MCPServerConfigError(
                f"MCP server {server_id!r} needs the env var(s): {', '.join(missing)}."
            )
        return {name: value for name, value in self.keys.items() if value}


class StdioSecret(MCPSecretBase):
    """For a server we start as a process: it gets ``keys`` as environment
    variables, with these same names, and nothing else of our environment."""


class HTTPSecret(MCPSecretBase):
    """For a server reached by URL: the one value of ``keys`` goes in a
    header of every request."""

    type: t.Literal["bearer", "api_key"]  # bearer: Authorization; api_key: ``header``
    header: str | None = None  # where an api_key goes, e.g. "x-api-key"

    @model_validator(mode="after")
    def _one_way(self) -> t.Self:
        if len(self.keys) != 1:
            raise MCPServerConfigError("An HTTP secret has exactly one key.")
        if (self.type == "api_key") != bool(self.header):
            raise MCPServerConfigError('type="api_key" needs a header; type="bearer" takes none.')
        return self

    def headers(self, server_id: str) -> dict[str, str]:
        """The header that carries the secret."""
        (value,) = self.values(server_id).values()
        if self.type == "bearer":
            return {"Authorization": f"Bearer {value}"}
        return {t.cast("str", self.header): value}


class MCPServerConfig(BaseModel):
    """Base configuration for one MCP server.

    Secrets never live in a serialized config: ``secret`` stores the names
    of the env vars that hold them, read when the connection opens.
    """

    # An unknown name is an error, never ignored. There is no ``env`` or
    # ``headers``: what a server gets from us goes in ``secret``.
    model_config = ConfigDict(extra="forbid")

    server_id: str
    transport_protocol: TransportProtocol = "streamable-http"
    approval_mode: ToolApprovalMode = ToolApprovalMode.ASK_APPROVED
    resource_approval_mode: ToolApprovalMode | None = None
    tool_approval_modes: dict[str, ToolApprovalMode] = Field(default_factory=dict)
    # Connecting (handshake + tool discovery) may take at most this long.
    connect_timeout_seconds: float = Field(default=30.0, gt=0)
    # After a failed connection, runs skip this server for this long
    # instead of each paying connect_timeout_seconds again.
    retry_after_seconds: float = Field(default=60.0, ge=0)
    # Trusted: believe what the server says about its own tools (read-only,
    # destructive). Otherwise only this config decides what needs approval.
    trusted: bool = False
    # Required: a run fails if this server is unavailable. Otherwise the run
    # goes on and the server's tools answer that it is unavailable.
    required: bool = False

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
    secret: StdioSecret | None = None  # None: the process gets no credentials

    @property
    def process_env(self) -> dict[str, str]:
        """What the child sees of our environment: the secret's values and
        nothing else (call when connecting)."""
        return self.secret.values(self.server_id) if self.secret else {}

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
    secret: HTTPSecret | None = None  # None: the server is used without authentication

    @field_validator("url")
    @classmethod
    def _url_required(cls, value: str) -> str:
        if not value:
            raise MCPServerConfigError("url cannot be empty.")
        return value

    @property
    def request_headers(self) -> dict[str, str]:
        """The header with the secret's value, the only one we add (call
        when connecting)."""
        return self.secret.headers(self.server_id) if self.secret else {}


_server_list_adapter = TypeAdapter(list[StdioMCPServerConfig | HTTPServerConfig])


def serialize_mcp_servers(
    configs: t.Sequence[StdioMCPServerConfig | HTTPServerConfig],
) -> str:
    """Serialize a mixed server list as JSON, including its transport types."""
    return _server_list_adapter.dump_json(list(configs)).decode("utf-8")


def deserialize_mcp_servers(payload: str | bytes) -> list[MCPServerConfig]:
    """Restore concrete server configurations from JSON."""
    return _server_list_adapter.validate_json(payload)
