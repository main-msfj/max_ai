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
        self._direct = False  # True: the deferred tools are few and go loaded
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
        return [tool for tool in self._tools.values() if self._direct or not tool.deferred]

    def deferred_tools(self) -> list[CoreTool]:
        """Listed in the prompt and found with search_tools."""
        return [tool for tool in self._tools.values() if tool.deferred and not self._direct]

    def load_deferred(self) -> None:
        """Send the deferred tools on every request too. For when they are
        few: finding them would cost more than carrying them. The tools
        themselves are not changed, only how this registry gives them."""
        self._direct = True
        self._tokens.clear()

    def deferred_tokens(self, counter: TokenCounter | None = None) -> int:
        """About how many tokens the deferred tools would take if loaded."""
        return self._count(self.deferred_tools(), counter)

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
        # 2. Count them and keep the number.
        self._tokens[key] = self._count(self.loaded_tools(), counter)
        return self._tokens[key]

    @staticmethod
    def _count(tools: list[CoreTool], counter: TokenCounter | None) -> int:
        """The tokens of ``tools`` the way they travel: their definitions as JSON."""
        definitions = [tool.build_tool_definition().model_dump(exclude_none=True) for tool in tools]
        text = json.dumps(definitions, ensure_ascii=False, default=str) if definitions else ""
        return TokenCounter.estimate(text, counter)
