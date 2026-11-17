"""call_tool: the door to deferred tools.

It never runs a tool itself. The dispatcher asks it to ``unwrap`` each
call, and what runs is the real tool with its arguments. ``execute`` only
runs when that was not possible, to tell the model what to fix.
"""

from __future__ import annotations

import json
import typing as t

from ....base.tools import CoreTool, ToolContext
from ....core.tool.registry import ToolRegistry
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import ToolApprovalMode
from ..search_tool import SearchToolsTool


class CallToolTool(CoreTool):
    """Run a deferred tool by name with its arguments."""

    TOOL_NAME: t.ClassVar[str] = "call_tool"
    DESCRIPTION: t.ClassVar[str] = "Run a tool found with search_tools, passing it its arguments."
    LOADED: t.ClassVar[str] = "{name} is already loaded: call it directly, not through call_tool."
    UNKNOWN: t.ClassVar[str] = "No tool named {name!r}. Find it with {search} first."

    def __init__(self, catalog: ToolRegistry) -> None:
        # Its own flags only cover the error answer: the real tool brings its own.
        super().__init__(
            name=self.TOOL_NAME,
            description=self.DESCRIPTION,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
        self._catalog = catalog

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "description": "Name of the tool to run."},
                "arguments": {"type": "object", "description": "The tool's parameters."},
            },
            "required": ["tool", "arguments"],
            "additionalProperties": False,
        }

    def unwrap(self, parameters: dict[str, t.Any]) -> tuple[str, dict[str, t.Any]] | None:
        """The deferred tool to run and its arguments; None if the call is wrong."""
        name = parameters.get("tool")
        arguments = self.arguments(parameters)
        target = self._catalog.get(name) if isinstance(name, str) else None
        if target is None or not target.deferred or arguments is None:
            return None
        return target.name, arguments

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        """Reached only when ``unwrap`` refused the call: say what to fix."""
        name = str(tool_request.parameters.get("tool") or "")
        target = self._catalog.get(name)

        # 1. A loaded tool is called by its own name.
        if target is not None and not target.deferred:
            return ToolResult.invalid_parameters(tool_request.id, self.LOADED.format(name=name))

        # 2. Anything else: there is no such tool.
        message = self.UNKNOWN.format(name=name, search=SearchToolsTool.TOOL_NAME)
        return ToolResult.invalid_parameters(tool_request.id, message)

    @staticmethod
    def arguments(parameters: dict[str, t.Any]) -> dict[str, t.Any] | None:
        """The arguments as a dictionary; some models send them as a JSON string."""
        arguments = parameters.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except ValueError:
                return None
        return t.cast("dict[str, t.Any]", arguments) if isinstance(arguments, dict) else None
