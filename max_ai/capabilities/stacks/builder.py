"""Builds the system prompt from the prompt layers and the tool registry."""

from __future__ import annotations

import typing as t

from ...base.layer import CoreLayer
from ...core.tool.registry import ToolRegistry


class PromptBuilder:
    """Joins the layers, in order, into one system prompt.

    Each layer receives only the variables it declares, and one that
    renders nothing is left out. What the tools say (names, descriptions,
    instructions) comes from the registry; the rest from ``build``.
    """

    SEPARATOR: t.ClassVar[str] = "\n\n"
    # Variable a layer may ask for -> the tool it names.
    TOOL_NAMES: t.ClassVar[dict[str, str]] = {
        "search_tool": "search_tools",
        "call_tool": "call_tool",
        "skill_tool": "skill",
        "bash_tool": "bash",
    }

    def __init__(
        self,
        layers: t.Sequence[CoreLayer],
        registry: ToolRegistry | None = None,
        tool_names: t.Mapping[str, str] | None = None,
    ) -> None:
        self.layers = list(layers)
        self.registry = registry
        self.tool_names = {**self.TOOL_NAMES, **(tool_names or {})}

    def tool_variables(self) -> dict[str, t.Any]:
        """What the registry gives the layers."""
        if self.registry is None:
            return {}
        variables: dict[str, t.Any] = dict(self.tool_names)
        if self.registry.get(self.tool_names["bash_tool"]) is None:
            variables["bash_tool"] = None  # no commands: layers must not mention it
        # Only the first line: the whole description comes with search_tools.
        variables["deferred_tools"] = [
            {"name": tool.name, "description": tool.description.strip().split("\n")[0]}
            for tool in self.registry.deferred_tools()
        ]
        variables["tool_instructions"] = [
            {"name": tool.name, "instructions": text.strip()}
            for tool in self.registry.all_tools()
            if (text := getattr(tool, "instructions", None))
        ]
        return variables

    def build(self, variables: t.Mapping[str, t.Any] | None = None) -> str:
        """The system prompt. ``variables`` win over what the registry gives."""
        known = {**self.tool_variables(), **(variables or {})}
        blocks: list[str] = []
        for layer in self.layers:
            wanted = layer.declared_variables()
            rendered = layer.render({k: v for k, v in known.items() if k in wanted}).strip()
            if rendered:
                blocks.append(rendered)
        return self.SEPARATOR.join(blocks)
