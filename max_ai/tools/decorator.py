"""
Decorator for turning functions into tools.
"""

import typing as t
from typing import overload

from ..types.tools import ToolApprovalMode
from .function_as_tool import FunctionAsTool


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    approval_mode: ToolApprovalMode = ToolApprovalMode.AUTO_APPROVED,
) -> t.Callable[[t.Callable[..., t.Any]], FunctionAsTool]: ...


@overload
def tool(
    func: t.Callable[..., t.Any],
    *,
    name: str | None = None,
    description: str | None = None,
    approval_mode: ToolApprovalMode = ToolApprovalMode.AUTO_APPROVED,
) -> FunctionAsTool: ...


def tool(
    func: t.Callable[..., t.Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    approval_mode: ToolApprovalMode = ToolApprovalMode.AUTO_APPROVED,
) -> FunctionAsTool | t.Callable[[t.Callable[..., t.Any]], FunctionAsTool]:
    """Wrap a function as a FunctionAsTool.

    Can be used with or without arguments:
        @tool
        def fn(...): ...

        @tool(name="custom", approval=ToolApprovalMode.ASK_APPROVED)
        def fn(...): ...

    Args:
        func: The function to wrap (only when used without parentheses).
        name: Custom tool name. Defaults to the function name.
        description: Custom description. Defaults to the function's docstring,
            or "Execute <func_name>" if no docstring exists.
        approval: When the tool requires user approval. Defaults to AUTO_APPROVED
            — decorated functions are assumed to be local, non-destructive helpers.
            Use ASK_APPROVED explicitly for tools that modify state.
    """

    def decorator(fn: t.Callable[..., t.Any]) -> FunctionAsTool:
        return FunctionAsTool(
            func=fn,
            name=name,
            description=description,
            approval_mode=approval_mode,
        )

    return decorator(func) if func is not None else decorator
