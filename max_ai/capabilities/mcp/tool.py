"""CoreTool adapters for MCP tools and resources."""

from __future__ import annotations

import asyncio
import logging
import re
import typing as t
from concurrent.futures import CancelledError as FuturesCancelledError

from mcp import MCPError
from mcp.types import CallToolResult, ImageContent, TextContent

from ...base.tools import CoreTool, ToolContext
from ...core.termination import CancellationToken
from ...errors.mcp import (
    MCPToolContentError,
    MCPToolError,
    MCPToolExecutionError,
)
from ...loggers import ScopedLogger
from ...types.tool_call import ToolCallRecord, ToolResult
from ...types.tools import ToolApprovalMode

if t.TYPE_CHECKING:
    from .client_manager import MCPClientManager

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="mcp.tool")


def _sanitize_name(raw: str) -> str:
    """Sanitize a string into a provider-safe tool name."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    name = re.sub(r"_+", "_", name).strip("_")
    if name and name[0].isdigit():
        name = f"mcp_{name}"
    return name or "mcp_tool"


class MCPTool(CoreTool):
    """Adapter that exposes one remote MCP tool as a local CoreTool."""

    def __init__(
        self,
        mcp_tool_name: str,
        mcp_tool_description: str,
        mcp_tool_schema: dict[str, t.Any],
        client_manager: "MCPClientManager",
        server_id: str,
        version: str = "1.0.0",
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
        timeout_seconds: float = 240.0,
        max_retries: int = 3,
        read_only: bool = False,
    ) -> None:
        """Initialize ``MCPTool``.

Parameters
----------
mcp_tool_name : str
    Value supplied for ``mcp_tool_name``.
mcp_tool_description : str
    Value supplied for ``mcp_tool_description``.
mcp_tool_schema : dict[str, t.Any]
    Value supplied for ``mcp_tool_schema``.
client_manager : 'MCPClientManager'
    Value supplied for ``client_manager``.
server_id : str
    Value supplied for ``server_id``.
version : str
    Value supplied for ``version``.
approval_mode : ToolApprovalMode | str
    Value supplied for ``approval_mode``.
timeout_seconds : float
    Value supplied for ``timeout_seconds``.
max_retries : int
    Value supplied for ``max_retries``.
read_only : bool
    Value supplied for ``read_only``."""
        self.mcp_tool_name = mcp_tool_name
        self.client_manager = client_manager
        self.server_id = server_id
        self._parameter_schema = mcp_tool_schema
        super().__init__(
            name=_sanitize_name(f"{server_id}_{mcp_tool_name}"),
            description=mcp_tool_description,
            version=version,
            approval_mode=approval_mode,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            read_only=read_only,
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        """Perform the ``parameters`` operation for ``MCPTool``."""
        return self._parameter_schema

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        """Execute the requested operation for ``MCPTool``.

Parameters
----------
tool_request : ToolCallRecord
    Value supplied for ``tool_request``.
tool_context : ToolContext | None
    Value supplied for ``tool_context``.
cancellation_token : CancellationToken | None
    Value supplied for ``cancellation_token``."""
        if cancellation_token and cancellation_token.is_cancelled():
            return ToolResult.cancelled_before_start(tool_request.id)

        params = self.validate_parameters(tool_request)
        if not params.is_tool_valid:
            return ToolResult.invalid_parameters(
                tool_request.id,
                params.msg_error or "Failed to validate MCP tool parameters.",
            )

        logger_for_call = log.child(
            server_id=self.server_id,
            tool_name=self.mcp_tool_name,
            tool_call_id=tool_request.id,
        )
        try:
            result: CallToolResult = await self.client_manager.call_tool(
                self.server_id,
                self.mcp_tool_name,
                tool_request.parameters,
                self.timeout_seconds,
            )

            if result.is_error:
                raise MCPToolExecutionError(f"Tool returned error: {result.content}")

            output = self._extract_tool_result(result)
            if isinstance(output, dict):
                if "results" in output:
                    output = output["results"]
                elif "result" in output:
                    output = output["result"]

            return ToolResult.success_result(
                tool_request.id,
                output,
                metadata={
                    "server_id": self.server_id,
                    "mcp_tool_name": self.mcp_tool_name,
                    "tool_type": "mcp",
                },
            )

        except (asyncio.CancelledError, FuturesCancelledError):
            return ToolResult.cancelled_during_execution(tool_request.id)
        except MCPError as exc:
            msg = f"MCP error: {exc}"
            logger_for_call.warning(msg)
            return ToolResult.execution_error(tool_request.id, msg)
        except MCPToolError as exc:
            msg = str(exc)
            logger_for_call.warning(msg)
            return ToolResult.execution_error(tool_request.id, msg)
        except Exception as exc:
            msg = f"MCP tool execution failed: {exc!r}"
            logger_for_call.error(msg, exc=exc)
            return ToolResult.execution_error(tool_request.id, msg)

    def _extract_tool_result(self, result: CallToolResult) -> t.Any:
        """Perform the internal ``extract tool result`` operation for ``MCPTool``.

Parameters
----------
result : CallToolResult
    Value supplied for ``result``."""
        if result.structured_content is not None:
            return result.structured_content

        outputs: dict[str, t.Any] = {"text": [], "images": []}
        for content in result.content:
            if isinstance(content, TextContent):
                outputs["text"].append(content.text)
            elif isinstance(content, ImageContent):
                outputs["images"].append(
                    {"data": content.data, "mime_type": content.mime_type}
                )

        if outputs["text"]:
            outputs["text"] = "\n".join(outputs["text"])
        else:
            outputs.pop("text")

        if not outputs["images"]:
            outputs.pop("images")

        if not outputs:
            raise MCPToolContentError("Tool returned empty content")

        if set(outputs) == {"text"}:
            return t.cast(str, outputs["text"])
        return outputs


class MCPResourceTool(CoreTool):
    """Synthetic CoreTool for reading resources from one MCP server."""

    def __init__(
        self,
        client_manager: "MCPClientManager",
        server_id: str,
        available_resources: list[t.Any] | None = None,
        resource_templates: list[t.Any] | None = None,
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
        timeout_seconds: float = 240.0,
    ) -> None:
        """Initialize ``MCPResourceTool``.

Parameters
----------
client_manager : 'MCPClientManager'
    Value supplied for ``client_manager``.
server_id : str
    Value supplied for ``server_id``.
available_resources : list[t.Any] | None
    Value supplied for ``available_resources``.
resource_templates : list[t.Any] | None
    Value supplied for ``resource_templates``.
approval_mode : ToolApprovalMode | str
    Value supplied for ``approval_mode``.
timeout_seconds : float
    Value supplied for ``timeout_seconds``."""
        self.client_manager = client_manager
        self.server_id = server_id
        self.available_resources = list(available_resources or [])
        self.resource_templates = list(resource_templates or [])
        super().__init__(
            name=_sanitize_name(f"{server_id}_read_resource"),
            description=self._build_description(),
            approval_mode=approval_mode,
            timeout_seconds=timeout_seconds,
            read_only=True,  # reading a resource changes nothing
        )
        self._parameter_schema: dict[str, t.Any] = {
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "URI of the MCP resource to read.",
                }
            },
            "required": ["uri"],
            "additionalProperties": False,
        }

    @property
    def parameters(self) -> dict[str, t.Any]:
        """Perform the ``parameters`` operation for ``MCPResourceTool``."""
        return self._parameter_schema

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        """Execute the requested operation for ``MCPResourceTool``.

Parameters
----------
tool_request : ToolCallRecord
    Value supplied for ``tool_request``.
tool_context : ToolContext | None
    Value supplied for ``tool_context``.
cancellation_token : CancellationToken | None
    Value supplied for ``cancellation_token``."""
        if cancellation_token and cancellation_token.is_cancelled():
            return ToolResult.cancelled_before_start(tool_request.id)

        params = self.validate_parameters(tool_request)
        if not params.is_tool_valid:
            return ToolResult.invalid_parameters(
                tool_request.id,
                params.msg_error or "Failed to validate MCP resource parameters.",
            )

        uri = tool_request.parameters["uri"]
        logger_for_call = log.child(
            server_id=self.server_id,
            resource_uri=uri,
            tool_call_id=tool_request.id,
        )
        try:
            result = await self.client_manager.read_resource(self.server_id, uri)

            texts: list[str] = []
            for content in result.contents:
                if hasattr(content, "text"):
                    texts.append(content.text)
                elif hasattr(content, "blob"):
                    mime_type = getattr(content, "mime_type", None) or "unknown"
                    texts.append(f"[Binary content: {mime_type}]")

            if not texts:
                return ToolResult.execution_error(
                    tool_request.id,
                    f"Resource {uri!r} returned no content.",
                )

            return ToolResult.success_result(
                tool_request.id,
                "\n".join(texts),
                metadata={
                    "server_id": self.server_id,
                    "resource_uri": uri,
                    "tool_type": "mcp",
                },
            )

        except (asyncio.CancelledError, FuturesCancelledError):
            return ToolResult.cancelled_during_execution(tool_request.id)
        except MCPError as exc:
            msg = f"MCP error reading resource: {exc}"
            logger_for_call.warning(msg)
            return ToolResult.execution_error(tool_request.id, msg)
        except Exception as exc:
            msg = f"Failed to read MCP resource {uri!r}: {exc!r}"
            logger_for_call.error(msg, exc=exc)
            return ToolResult.execution_error(tool_request.id, msg)

    def _build_description(self) -> str:
        """Perform the internal ``build description`` operation for ``MCPResourceTool``."""
        parts = [f"Read a resource from the '{self.server_id}' MCP server by URI."]
        if self.available_resources:
            lines = []
            for resource in self.available_resources:
                desc = getattr(resource, "description", None)
                suffix = f" ({desc})" if desc else ""
                lines.append(f"- {resource.name}: {resource.uri}{suffix}")
            parts.append("Available resources:\n" + "\n".join(lines))
        if self.resource_templates:
            lines = []
            for template in self.resource_templates:
                desc = getattr(template, "description", None)
                suffix = f" ({desc})" if desc else ""
                lines.append(f"- {template.uri_template}{suffix}")
            parts.append("Resource templates:\n" + "\n".join(lines))
        return "\n".join(parts)
