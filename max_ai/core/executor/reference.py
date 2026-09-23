"""Explicit, JSON-only reconstruction of native tools in remote images."""

import importlib
from typing import Any, Literal

from pydantic import BaseModel, Field

from ...base.tools import CoreTool


class ToolReference(BaseModel):
    """Constructor arguments must reproduce the tool advertised by the host."""
    module: str
    qualname: str
    kind: Literal["class", "function", "factory"] = "class"
    config: dict[str, Any] = Field(default_factory=dict)
    tool_name: str | None = None

    def build(self) -> CoreTool:
        if self.module == "__main__" or "<locals>" in self.qualname:
            raise ValueError("Remote tools must be importable from an installed module")
        value = importlib.import_module(self.module)
        for part in self.qualname.split("."):
            value = getattr(value, part)
        if self.kind == "function":
            if isinstance(value, CoreTool):
                return value
            from ...capabilities.tools.function_as_tool import FunctionAsTool
            return FunctionAsTool(value, **self.config)
        value = value(**self.config)
        if self.kind == "factory":
            for tool in value.tools:
                if tool.name == self.tool_name:
                    return tool
            raise ValueError(f"Factory does not provide {self.tool_name!r}")
        if not isinstance(value, CoreTool):
            raise TypeError("Reference did not construct a CoreTool")
        return value


def reference_for(tool: CoreTool) -> ToolReference:
    """Adapt existing docker_ref definitions; never pickle live tool objects."""
    from ...capabilities.tools.bash import BashTool

    if type(tool) is BashTool:
        return ToolReference(
            module="max_ai.capabilities.tools.bash", qualname="BashTool",
            config=tool.docker_ref().config,
        )
    try:
        old = tool.docker_ref()
    except Exception as error:
        raise ValueError(
            f"Tool {tool.name!r} needs an explicit ToolReference for remote execution"
        ) from error
    return ToolReference(
        module=old.module, qualname=old.qualname, kind=old.kind,
        config=old.options if old.kind == "function" else old.config,
    )
