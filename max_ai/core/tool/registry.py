"""Single tool catalog. Provider-specific schema conversion stays in clients."""

from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any

from ...base.tools import CoreTool
from ...types.tools import CoreToolDefinition


class ToolRegistry:
    def __init__(self, tools: Iterable[CoreTool | Callable[..., Any]] = ()) -> None:
        self._tools: dict[str, CoreTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: CoreTool | Callable[..., Any]) -> CoreTool:
        """Accept CoreTools (including MCP adapters) or wrapped functions."""
        if not isinstance(tool, CoreTool):
            if not callable(tool):
                raise TypeError("Expected a CoreTool or callable")
            from ...capabilities.tools.function_as_tool import FunctionAsTool

            tool = FunctionAsTool(tool)
        if not tool.name or tool.name in self._tools:
            raise ValueError(f"Empty or duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> CoreTool | None:
        return self._tools.get(name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def all_tools(self) -> list[CoreTool]:
        return list(self._tools.values())

    def definitions(self) -> list[CoreToolDefinition]:
        return [
            CoreToolDefinition(name=tool.name, description=tool.description,
                               parameters=deepcopy(tool.parameters))
            for tool in self._tools.values()
        ]
