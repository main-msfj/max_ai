"""CoreTool adapters for MCP tools and resources."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import typing as t
from concurrent.futures import CancelledError as FuturesCancelledError

from mcp import MCPError
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from ...base.tools import CoreTool, ToolContext
from ...core.event_type import MCPProgressEvent
from ...core.termination import CancellationToken
from ...errors.mcp import (
    MCPConnectionLost,
    MCPToolContentError,
    MCPToolError,
    MCPToolExecutionError,
)
from ...loggers import ScopedLogger
from ...types.tool_call import ToolCallRecord, ToolFile, ToolResult
from ...types.tools import ToolApprovalMode, ToolKind
from .task import MCPTask, MCPTaskRunning

if t.TYPE_CHECKING:
    from .client_manager import MCPClientManager

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="mcp.tool")


TASK_RUNNING = "The tool is still working on the server. Its result will arrive when it is ready."


def _sanitize_name(raw: str) -> str:
    """Sanitize a string into a provider-safe tool name."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    name = re.sub(r"_+", "_", name).strip("_")
    if name and name[0].isdigit():
        name = f"mcp_{name}"
    return name or "mcp_tool"


class MCPTool(CoreTool):
    """Adapter that exposes one remote MCP tool as a local CoreTool."""

    kind = ToolKind.MCP  # the server names it

    def __init__(
        self,
        mcp_tool_name: str,
        mcp_tool_description: str,
        mcp_tool_schema: dict[str, t.Any],
        client_manager: "MCPClientManager",
        server_id: str,
        version: str = "1.0.0",
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
        read_only: bool = False,
        deferred: bool = True,
    ) -> None:
        self.mcp_tool_name = mcp_tool_name
        self.client_manager = client_manager
        self.server_id = server_id
        self._parameter_schema = mcp_tool_schema
        super().__init__(
            name=_sanitize_name(f"{server_id}_{mcp_tool_name}"),
            description=mcp_tool_description,
            version=version,
            approval_mode=approval_mode,
            read_only=read_only,
            deferred=deferred,
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        return self._parameter_schema

    @property
    def timeout(self) -> float:
        """Seconds the dispatcher lets a call run: the manager's own limit."""
        return self.client_manager.max_timeout

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
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
            def on_progress(progress: float, total: float | None, message: str | None) -> None:
                if tool_context is not None and tool_context.emit_event is not None:
                    tool_context.emit_event(MCPProgressEvent(
                        source=self.name, tool_call_id=tool_request.id,
                        server_id=self.server_id, tool_name=self.mcp_tool_name,
                        progress=progress, total=total, message=message,
                    ))

            result: CallToolResult = await self.client_manager.call_tool(
                self.server_id,
                self.mcp_tool_name,
                tool_request.parameters,
                read_only=self.read_only,
                on_progress=on_progress,
            )

            if result.is_error:
                raise MCPToolExecutionError(f"Tool returned error: {result.content}")

            output, files = self._extract_tool_result(result)
            # Servers wrap a plain value as {"result": value}: unwrap only that,
            # never a dictionary that says more.
            if isinstance(output, dict) and len(output) == 1:
                output = output.get("results", output.get("result", output))

            return ToolResult.success_result(
                tool_request.id,
                output,
                metadata={
                    "server_id": self.server_id,
                    "mcp_tool_name": self.mcp_tool_name,
                    "tool_type": "mcp",
                },
            ).model_copy(update={"files": files})

        except (asyncio.CancelledError, FuturesCancelledError):
            return ToolResult.cancelled_during_execution(tool_request.id)
        except MCPTaskRunning as running:
            if tool_context is not None:
                tool_context.run_context.mcp_tasks[running.task_id] = MCPTask(
                    task_id=running.task_id, server_id=self.server_id, tool_name=self.name,
                    tool_call_id=tool_request.id, expires_at=running.expires_at,
                ).model_dump()
            return ToolResult.success_result(tool_request.id, TASK_RUNNING)
        except MCPConnectionLost as exc:
            msg = (f"{exc} The connection to {self.server_id} dropped during the call: it may "
                   "or may not have run. Check before calling it again.")
            logger_for_call.warning(msg)
            return ToolResult.execution_error(tool_request.id, msg)
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

    def _extract_tool_result(self, result: CallToolResult) -> tuple[t.Any, list[ToolFile]]:
        """What the model reads, and apart the files the server sent."""
        texts: list[str] = []
        files: list[ToolFile] = []
        for content in result.content:
            if isinstance(content, TextContent):
                texts.append(content.text)
            elif isinstance(content, (ImageContent, AudioContent)):
                files.append(
                    ToolFile(data=base64.b64decode(content.data), mime_type=content.mime_type)
                )
            elif isinstance(content, EmbeddedResource):
                _read_resource(content.resource, texts, files)
            elif isinstance(content, ResourceLink):
                texts.append(f"Resource {content.name}: {content.uri}")
            else:
                texts.append(f"[content this client cannot read: {content.type}]")
        output: t.Any = "\n".join(texts)
        if result.structured_content is not None:
            output = result.structured_content
        if output in ("", None) and not files:
            raise MCPToolContentError("Tool returned empty content")
        return output, files


def _read_resource(resource: t.Any, texts: list[str], files: list[ToolFile]) -> None:
    """A resource's content: its text to ``texts``, anything else to ``files``."""
    if isinstance(resource, TextResourceContents):
        texts.append(resource.text)
    elif isinstance(resource, BlobResourceContents):
        mime_type = resource.mime_type or "application/octet-stream"
        files.append(ToolFile(data=base64.b64decode(resource.blob), mime_type=mime_type))


class MCPResourceTool(CoreTool):
    """Synthetic CoreTool for reading resources from one MCP server."""

    kind = ToolKind.MCP

    def __init__(
        self,
        client_manager: "MCPClientManager",
        server_id: str,
        available_resources: list[t.Any] | None = None,
        resource_templates: list[t.Any] | None = None,
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
        deferred: bool = True,
    ) -> None:
        self.client_manager = client_manager
        self.server_id = server_id
        self.available_resources = list(available_resources or [])
        self.resource_templates = list(resource_templates or [])
        super().__init__(
            name=_sanitize_name(f"{server_id}_read_resource"),
            description=self._build_description(),
            approval_mode=approval_mode,
            read_only=True,  # reading a resource changes nothing
            deferred=deferred,
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
        return self._parameter_schema

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
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
            files: list[ToolFile] = []
            for content in result.contents:
                _read_resource(content, texts, files)

            if not texts and not files:
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
            ).model_copy(update={"files": files})

        except (asyncio.CancelledError, FuturesCancelledError):
            return ToolResult.cancelled_during_execution(tool_request.id)
        except MCPError as exc:
            msg = f"MCP error reading resource: {exc}"
            logger_for_call.warning(msg)
            return ToolResult.execution_error(tool_request.id, msg)
        except MCPToolError as exc:
            msg = str(exc)
            logger_for_call.warning(msg)
            return ToolResult.execution_error(tool_request.id, msg)
        except Exception as exc:
            msg = f"Failed to read MCP resource {uri!r}: {exc!r}"
            logger_for_call.error(msg, exc=exc)
            return ToolResult.execution_error(tool_request.id, msg)

    def _build_description(self) -> str:
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
