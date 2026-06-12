"""Configuration objects for MCP server connections."""

from __future__ import annotations

import typing as t

from ..errors.mcp import MCPServerConfigError
from ..types.tools import ToolApprovalMode

TransportProtocol = t.Literal["stdio", "sse", "streamable-http"]
HTTPTransportProtocol = t.Literal["sse", "streamable-http"]
ToolApprovalOverrides = dict[str, ToolApprovalMode | str]


def _normalize_approval_overrides(
    overrides: ToolApprovalOverrides | None,
) -> dict[str, ToolApprovalMode]:
    return {name: ToolApprovalMode(mode) for name, mode in (overrides or {}).items()}


class MCPServerConfig:
    """Base configuration for one MCP server."""

    def __init__(
        self,
        server_id: str,
        transport_protocol: TransportProtocol = "streamable-http",
        env: dict[str, str] | None = None,
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
        resource_approval_mode: ToolApprovalMode | str | None = None,
        tool_approval_modes: ToolApprovalOverrides | None = None,
        timeout_seconds: float = 240.0,
    ) -> None:
        if not server_id:
            raise MCPServerConfigError("server_id cannot be empty.")
        self.server_id = server_id
        self.transport_protocol = transport_protocol
        self.env = env or {}
        self.approval_mode = ToolApprovalMode(approval_mode)
        self.resource_approval_mode = ToolApprovalMode(
            resource_approval_mode or approval_mode
        )
        self.tool_approval_modes = _normalize_approval_overrides(tool_approval_modes)
        self.timeout_seconds = float(timeout_seconds)


class StdioMCPServerConfig(MCPServerConfig):
    """Configuration for an MCP server launched over stdio."""

    def __init__(
        self,
        server_id: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
        resource_approval_mode: ToolApprovalMode | str | None = None,
        tool_approval_modes: ToolApprovalOverrides | None = None,
        timeout_seconds: float = 240.0,
    ) -> None:
        if not command:
            raise MCPServerConfigError("command cannot be empty.")
        super().__init__(
            server_id=server_id,
            transport_protocol="stdio",
            env=env,
            approval_mode=approval_mode,
            resource_approval_mode=resource_approval_mode,
            tool_approval_modes=tool_approval_modes,
            timeout_seconds=timeout_seconds,
        )
        self.command = command
        self.args = list(args or [])


class HTTPServerConfig(MCPServerConfig):
    """Configuration for an MCP server exposed over SSE or streamable HTTP."""

    def __init__(
        self,
        server_id: str,
        url: str,
        transport_protocol: HTTPTransportProtocol = "streamable-http",
        headers: dict[str, str] | None = None,
        token: str | None = None,
        env: dict[str, str] | None = None,
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
        resource_approval_mode: ToolApprovalMode | str | None = None,
        tool_approval_modes: ToolApprovalOverrides | None = None,
        timeout_seconds: float = 240.0,
    ) -> None:
        if not url:
            raise MCPServerConfigError("url cannot be empty.")
        super().__init__(
            server_id=server_id,
            transport_protocol=transport_protocol,
            env=env,
            approval_mode=approval_mode,
            resource_approval_mode=resource_approval_mode,
            tool_approval_modes=tool_approval_modes,
            timeout_seconds=timeout_seconds,
        )
        self.url = url
        self.headers = dict(headers or {})
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
