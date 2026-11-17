"""The tools an agent has. Knows nothing about users, sandboxes or runs."""

import json
import typing as t
from collections.abc import Iterable

from ...base.tools import CoreTool
from ...errors.tools import ToolRegistryError
from ...types.tools import CoreToolDefinition
from ..tokenizer import TokenCounter


class ToolRegistry:
    def __init__(self, tools: Iterable[CoreTool] = ()) -> None:
        self._tools: dict[str, CoreTool] = {}
        self._tokens: dict[str | None, int] = {}  # by tokenizer; emptied when the tools change
        for tool in tools:
            self.register(tool)

    def register(self, tool: t.Any) -> CoreTool:
        """Add a CoreTool; a function becomes one with @tool or FunctionAsTool."""
        if not isinstance(tool, CoreTool):
            raise ToolRegistryError.not_a_tool(tool)
        if tool.name in self._tools:
            raise ToolRegistryError.duplicate_name(tool.name)
        self._tools[tool.name] = tool
        self._tokens.clear()
        return tool

    def get(self, name: str) -> CoreTool | None:
        """Any registered tool, loaded or deferred."""
        return self._tools.get(name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._tokens.clear()

    def all_tools(self) -> list[CoreTool]:
        return list(self._tools.values())

    def loaded_tools(self) -> list[CoreTool]:
        """Sent to the model on every request."""
        return [tool for tool in self._tools.values() if not tool.deferred]

    def deferred_tools(self) -> list[CoreTool]:
        """Listed in the prompt and found with search_tools."""
        return [tool for tool in self._tools.values() if tool.deferred]

    def definitions(self) -> list[CoreToolDefinition]:
        """What the model receives: name, description and parameters of each
        loaded tool. Each client turns it into its provider's format."""
        return [tool.build_tool_definition() for tool in self.loaded_tools()]

    def tokens(self, counter: TokenCounter | None = None) -> int:
        """About how many tokens the loaded tools take in a request. Counted
        once and kept until a tool is added or removed."""
        # 1. The number kept for this tokenizer, if the tools did not change.
        key = counter.tokenizer_base if counter is not None else None
        if key in self._tokens:
            return self._tokens[key]
        # 2. Count the definitions the way they travel: as JSON.
        definitions = [definition.model_dump(exclude_none=True) for definition in self.definitions()]
        text = json.dumps(definitions, ensure_ascii=False, default=str) if definitions else ""
        self._tokens[key] = TokenCounter.estimate(text, counter)
        return self._tokens[key]
