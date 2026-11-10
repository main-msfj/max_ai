"""Single tool catalog. Provider-specific schema conversion stays in clients."""

from collections.abc import Iterable

from ...base.tools import CoreTool
from ...errors.tools import ToolRegistryError


class ToolRegistry:
    def __init__(self, tools: Iterable[CoreTool] = ()) -> None:
        self._tools: dict[str, CoreTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: CoreTool) -> CoreTool:
        """Add a CoreTool; a function becomes one with @tool or FunctionAsTool."""
        if not isinstance(tool, CoreTool):
            raise ToolRegistryError.not_a_tool(tool)
        if tool.name in self._tools:
            raise ToolRegistryError.duplicate_name(tool.name)
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> CoreTool | None:
        """Any registered tool, loaded or deferred."""
        return self._tools.get(name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def all_tools(self) -> list[CoreTool]:
        return list(self._tools.values())

    def loaded_tools(self) -> list[CoreTool]:
        """Sent to the model on every request."""
        return [tool for tool in self._tools.values() if not tool.deferred]

    def deferred_tools(self) -> list[CoreTool]:
        """Listed in the prompt and found with search_tools."""
        return [tool for tool in self._tools.values() if tool.deferred]
