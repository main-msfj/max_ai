"""
Wrap Python functions as agent tools.
"""

import asyncio
import inspect
import logging
import typing as t
from concurrent.futures import CancelledError as FuturesCancelledError

from asyncio import CancelledError as AsyncioCancelledError
from pydantic import ConfigDict, create_model, TypeAdapter, ValidationError


from ..loggers import ScopedLogger
from ..errors.tools import ToolRetry
from ..termination import CancellationToken
from ..base.tools import CoreTool, ToolContext
from ..types.tool_call import ToolCallRecord, ToolResult
from ..types.tools import ToolApprovalMode, CoreToolParameters


# -------- LOGGER -----------------------------------------------------------
logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="FunctionAsTool")


class FunctionAsTool(CoreTool):
    """Wrap a Python function as a tool with schema + validation."""

    _LOG_ACTION: str = "Yielding validation error to LLM/UI."

    def __init__(
        self,
        func: t.Callable[..., t.Any],
        name: str | None = None,
        description: str | None = None,
        version: str = "1.0.0",
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
        timeout_seconds: float = 300,
        max_retries: int = 3,
    ):
        """Create a tool from a Python function.

        Args:
            func: The function to wrap as a tool (sync or async).
            name: Custom name (defaults to function name).
            description: Custom description (defaults to function docstring).
            version: Semver string.
            approval_mode: Whether approval is required before execution.
            timeout_seconds: Max execution time in seconds.
            max_retries: Consumed by the executor on ToolRetry.
        """
        self.func = func
        super().__init__(
            name=name or func.__name__,
            description=description or func.__doc__ or f"Execute {func.__name__}",
            version=version,
            approval_mode=approval_mode,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

        self.signature = inspect.signature(func)
        self.type_hints = t.get_type_hints(func)

        # ToolContext detection: must be the FIRST parameter if used.
        self._first_param_name: str | None = next(iter(self.signature.parameters), None)
        self._needs_context: bool = (
            self._first_param_name is not None
            and self.type_hints.get(self._first_param_name) is ToolContext
        )

        self._dynamic_model, self._parameters_schema = self._build_parameters_schema()
        self._type_adapter: TypeAdapter[t.Any] = TypeAdapter(self._dynamic_model)

    @property
    def parameters(self) -> dict[str, t.Any]:
        return self._parameters_schema

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        """Validate parameters using the dynamic Pydantic model.

        Overrides ``CoreTool.validate_parameters``: since the schema is
        built from the function's type hints, this uses the native
        Pydantic validator rather than round-tripping through
        jsonschema. The JSON schema shown to the LLM is generated
        from the same model, so the two stay in sync.
        """
        try:
            self._type_adapter.validate_python(tool_request.parameters)
            return CoreToolParameters(is_tool_valid=True, msg_error=None)
        except ValidationError as e:
            messages = [
                f"{'.'.join(map(str, err['loc'])) or '<root>'}: {err['msg']}"
                for err in e.errors()
            ]
            return CoreToolParameters(
                is_tool_valid=False,
                msg_error="; ".join(messages),
            )

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        logs = log.child(
            action="Executing FunctionTool",
            tool_name=self.name,
            tool_call_id=tool_request.id,
            parameters=tool_request.parameters,
        )

        # 1. Validate tool parameters.
        tool_params = self.validate_parameters(tool_request)
        if not tool_params.is_tool_valid:
            msg = tool_params.msg_error or "Failed at validating parameters"
            logs.error(msg, action=self._LOG_ACTION)
            return ToolResult.invalid_parameters(tool_request.id, msg)

        # 2. Build kwargs, injecting ToolContext if the function requires it.
        kwargs: dict[str, t.Any] = dict(tool_request.parameters)
        if self._needs_context:
            if tool_context is None:
                msg = (
                    f"Tool: {self.name} requires a ToolContext "
                    "but none was provided by the caller."
                )
                logs.error(msg, action=self._LOG_ACTION)
                return ToolResult.execution_error(tool_request.id, msg)
            assert self._first_param_name is not None
            kwargs[self._first_param_name] = tool_context

        # 3. Dispatch sync vs async.
        try:
            if inspect.iscoroutinefunction(self.func):
                task: t.Any = asyncio.create_task(self.func(**kwargs))
            else:
                # Sync functions run in the default executor. Note:
                # threads are not truly cancellable in Python —
                # link_future marks the future as cancelled but the
                # thread runs to completion.
                loop = asyncio.get_running_loop()
                task = loop.run_in_executor(None, lambda: self.func(**kwargs))

            if cancellation_token:
                cancellation_token.link_future(task)

            result = await asyncio.wait_for(task, timeout=self.timeout_seconds)
            return ToolResult.success_result(
                tool_request.id, result, {"name": self.name}
            )

        except asyncio.TimeoutError:
            logs.error(
                "FunctionTool execution timed out",
                action="Yielding timeout error to LLM/UI.",
            )
            return ToolResult.timeout(
                tool_request.id, timeout_seconds=self.timeout_seconds
            )

        except (AsyncioCancelledError, FuturesCancelledError):
            logs.warning(
                msg="FunctionTool execution cancelled", action=self._LOG_ACTION
            )
            return ToolResult.cancelled_during_execution(tool_request.id)

        except ToolRetry as retry:
            return ToolResult.tool_failure(
                tool_request.id,
                error=retry.message,
                metadata={"name": self.name, "retry_requested": True},
            )

        except Exception as e:
            msg = f"Error executing FunctionTool '{self.name}': {e!r}"
            logs.error(msg, action=self._LOG_ACTION)
            return ToolResult.execution_error(tool_request.id, msg)

    # -------- PRIVATE -----------------------------------------------------------
    def _build_parameters_schema(self) -> tuple[type, dict[str, t.Any]]:
        """Build a dynamic Pydantic model and its JSON schema from the signature."""
        fields: dict[str, t.Any] = {}
        for param_name, param in self.signature.parameters.items():
            if self._needs_context and param_name == self._first_param_name:
                continue
            param_type = self.type_hints.get(param_name, t.Any)
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[param_name] = (param_type, default)

        DynamicModel = create_model(
            f"{self.func.__name__}_params",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        return DynamicModel, DynamicModel.model_json_schema()

    # -------- SERIALIZATION -----------------------------------------------------------
    def dump_component(self) -> t.NoReturn:
        """``FunctionAsTool`` wraps arbitrary callables and cannot be serialized."""
        raise TypeError(
            f"FunctionTool '{self.name}' cannot be serialized: it wraps an "
            "arbitrary Python callable. Create a CoreTool subclass for "
            "serializable tools."
        )
