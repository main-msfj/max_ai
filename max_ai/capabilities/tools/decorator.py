"""
Decorator for turning functions into tools.
"""

import typing as t
from typing import overload

from ...types.tools import ToolApprovalMode
from .function_as_tool import FunctionAsTool


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    approval_mode: str | ToolApprovalMode = ToolApprovalMode.AUTO_APPROVED,
    read_only: bool = False,
) -> t.Callable[[t.Callable[..., t.Any]], FunctionAsTool]:
    """Perform the ``tool`` operation.

Parameters
----------
name : str | None
    Value supplied for ``name``.
description : str | None
    Value supplied for ``description``.
approval_mode : str | ToolApprovalMode
    Value supplied for ``approval_mode``.
read_only : bool
    Value supplied for ``read_only``."""
    ...


@overload
def tool(
    func: t.Callable[..., t.Any],
    *,
    name: str | None = None,
    description: str | None = None,
    approval_mode: str | ToolApprovalMode = ToolApprovalMode.AUTO_APPROVED,
    read_only: bool = False,
) -> FunctionAsTool:
    """Perform the ``tool`` operation.

Parameters
----------
func : t.Callable[..., t.Any]
    Value supplied for ``func``.
name : str | None
    Value supplied for ``name``.
description : str | None
    Value supplied for ``description``.
approval_mode : str | ToolApprovalMode
    Value supplied for ``approval_mode``.
read_only : bool
    Value supplied for ``read_only``."""
    ...


def tool(
    func: t.Callable[..., t.Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    approval_mode: str | ToolApprovalMode = ToolApprovalMode.AUTO_APPROVED,
    read_only: bool = False,
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
        read_only: No side effects: may run in parallel with other read-only calls.
    """

    def decorator(fn: t.Callable[..., t.Any]) -> FunctionAsTool:
        """Perform the ``decorator`` operation.

Parameters
----------
fn : t.Callable[..., t.Any]
    Value supplied for ``fn``."""
        return FunctionAsTool(
            func=fn,
            name=name,
            description=description,
            approval_mode=approval_mode,
            read_only=read_only,
        )

    return decorator(func) if func is not None else decorator
