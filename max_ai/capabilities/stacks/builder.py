"""Builds the system prompt from the prompt layers and the tool registry."""

from __future__ import annotations

import typing as t

from ...base.layer import CoreLayer
from ...core.tokenizer import TokenCounter
from ...core.tool.registry import ToolRegistry


class PromptBuilder:
    """Joins the layers into one system prompt.

    The builder knows no layer: it gets them and, at most, an order.

    - Without ``order`` the layers stay as they were given.
    - With ``order`` (a list of layer classes) they are sorted by it. A layer
      whose class is not listed goes last; layers of the same class keep the
      order they came in.

    Order matters for the provider's cache: it keeps the prompt up to the
    first character that changes. Put first what never changes and last what
    changes during a conversation (``STANDARD_ORDER`` does). Arranging the
    layers another way is up to whoever passes them.

    Each layer receives only the variables it declares, and one that
    renders nothing is left out. What the tools say (names, descriptions,
    instructions) comes from the registry; the rest from ``build``.
    """

    SEPARATOR: t.ClassVar[str] = "\n\n"
    # Variable a layer may ask for -> the tool it names.
    TOOL_NAMES: t.ClassVar[dict[str, str]] = {
        "search_tool": "search_tools",
        "call_tool": "call_tool",
        "skill_tool": "read_skill",
        "bash_tool": "bash",
        "memory_tool": "save_memory",
    }

    def __init__(
        self,
        layers: t.Sequence[CoreLayer],
        registry: ToolRegistry | None = None,
        order: t.Sequence[type[CoreLayer]] | None = None,
        tool_names: t.Mapping[str, str] | None = None,
    ) -> None:
        self.layers = list(layers)
        if order:
            place = {layer_class: index for index, layer_class in enumerate(order)}
            self.layers.sort(key=lambda layer: place.get(type(layer), len(place)))
        self.registry = registry
        self.tool_names = {**self.TOOL_NAMES, **(tool_names or {})}

    def tool_variables(self) -> dict[str, t.Any]:
        """What the registry gives the layers; empty lists without one."""
        tools = self.registry.all_tools() if self.registry is not None else []
        deferred = self.registry.deferred_tools() if self.registry is not None else []
        variables: dict[str, t.Any] = dict(self.tool_names)
        if not any(tool.name == self.tool_names["bash_tool"] for tool in tools):
            variables["bash_tool"] = None  # no commands: layers must not mention it
        if not any(tool.name == self.tool_names["memory_tool"] for tool in tools):
            variables["memory_tool"] = None  # nothing to save with
        # Only the first line: the whole description comes with search_tools.
        variables["deferred_tools"] = [
            {"name": tool.name, "description": tool.description.strip().split("\n")[0]}
            for tool in deferred
        ]
        variables["tool_instructions"] = [
            {"name": tool.name, "instructions": text.strip()}
            for tool in tools
            if (text := getattr(tool, "instructions", None))
        ]
        return variables

    def build(self, variables: t.Mapping[str, t.Any] | None = None) -> str:
        """The system prompt. ``variables`` win over what the registry gives."""
        known: dict[str, t.Any] = {"skills": [], **self.tool_variables(), **(variables or {})}
        blocks: list[str] = []
        for layer in self.layers:
            wanted = layer.declared_variables()
            rendered = layer.render({k: v for k, v in known.items() if k in wanted}).strip()
            if rendered:
                blocks.append(rendered)
        return self.SEPARATOR.join(blocks)

    def tokens(
        self,
        variables: t.Mapping[str, t.Any] | None = None,
        counter: TokenCounter | None = None,
    ) -> int:
        """About how many tokens the system prompt takes."""
        return TokenCounter.estimate(self.build(variables), counter)
