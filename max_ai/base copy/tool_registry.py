"""Single tool catalog. Provider-specific schema conversion stays in clients."""

from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any

from ..types.tools import CoreToolDefinition
from .tools import CoreTool


class ToolRegistry:
    def __init__(self, tools: Iterable[CoreTool | Callable[..., Any]] = ()):
        self._tools: dict[str, CoreTool] = {}
        self._references: dict[str, Any] = {}
        self._host_tools: set[str] = set()
        for tool in tools:
            self.register(tool)

    def register(self, tool: CoreTool | Callable[..., Any], *, reference=None,
                 host: bool = False) -> CoreTool:
        """Accept CoreTools (including MCP adapters) or wrapped functions."""
        if not isinstance(tool, CoreTool):
            if not callable(tool):
                raise TypeError("Expected a CoreTool or callable")
            from ..capabilities.tools.function_as_tool import FunctionAsTool

            tool = FunctionAsTool(tool)
        if not tool.name or tool.name in self._tools:
            raise ValueError(f"Empty or duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool
        if reference is not None:
            self._references[tool.name] = reference
        if host:
            self._host_tools.add(tool.name)
        return tool

    def reference(self, name: str):
        return self._references.get(name)

    def runs_on_host(self, name: str) -> bool:
        """Trusted application choice for MCP/SDK tools, never model input."""
        return name in self._host_tools

    def get(self, name: str) -> CoreTool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[CoreTool]:
        return list(self._tools.values())

    def definitions(self) -> list[CoreToolDefinition]:
        return [
            CoreToolDefinition(name=tool.name, description=tool.description,
                               parameters=deepcopy(tool.parameters))
            for tool in self._tools.values()
        ]
