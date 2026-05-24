"""
Core abstractions for LLM agent tools.

Defines base classes and interfaces that enable agents
to execute external actions (e.g., APIs, file I/O, services).
"""

import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel
from jsonschema import Draft202012Validator

from ..errors.tools import DockerToolReferenceError
from ..termination import CancellationToken
from .component import ComponentBase
from ..types.tool_call import ToolCallRecord, ToolResult
from ..types.tools import (
    ToolApprovalMode,
    CoreToolParameters,
    CoreToolDefinition,
    DockerToolRef,
)


class ToolContext:
    """Runtime context available to tools."""

    __slots__ = ("user_id", "session_id", "run_id", "retry_count", "deps")

    def __init__(
        self,
        run_id: str,
        session_id: str = "",
        user_id: str = "runtime",
        retry_count: int = 0,
        deps: dict[str, t.Any] | None = None,
    ):
        self.run_id = run_id
        self.user_id = user_id
        self.session_id = session_id
        self.retry_count = retry_count
        self.deps = deps or {}


class CoreTool(ComponentBase[BaseModel], ABC):
    """Base class for all agent tools."""

    _JSON_TYPE_MAP: t.ClassVar[dict[str, type | tuple[type, ...]]] = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
        timeout_seconds: float = 300,
        max_retries: int = 3,
    ):
        self.name = name
        self.version = version
        self.description = description
        self.approval_mode = approval_mode
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Lazily-built schema validator
        self._schema_validator: Draft202012Validator | None = None

    @property
    @abstractmethod
    def parameters(self) -> dict[str, t.Any]:
        """JSON schema for tool inputs."""
        ...

    def docker_ref(self) -> DockerToolRef:
        raise DockerToolReferenceError(
            self.name,
            "it does not provide a DockerToolRef",
        )


    # -------- ABSTRACT -----------------------------------------------------------
    @abstractmethod
    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        """Execute the tool
        Args:
            tool_request: Input parameters for this invocation.
            tool_context: Runtime context (run id, session id, deps, ...).
            cancellation_token: Token for cooperative cancellation.

        Returns:
            A ToolResult describing the outcome.
        """
        ...

    # -------- PUBLIC -----------------------------------------------------------
    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        """
        Validate input parameters against this tool's JSON schema.

        Uses the Draft 2020-12 JSON Schema spec — the same spec that LLM
        providers (OpenAI, Anthropic, ...) use to describe tool parameters.
        The compiled validator is cached on first use, because building it
        is non-trivial (schema compilation, $ref resolution, etc.)
        Subclasses may override this method to add custom rules on top of
        the schema validation.
        """
        validator = self._get_validator()

        # iter_errors yields ALL errors, not just the first. Sorting by path
        # gives stable, deterministic messages (useful for tests and for the
        # LLM to correct everything in a single retry).
        errors = sorted(
            validator.iter_errors(tool_request.parameters),
            key=lambda e: list(e.absolute_path),
        )
        if not errors:
            return CoreToolParameters(is_tool_valid=True, msg_error=None)

        message = "; ".join(
            f"{'.'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in errors
        )
        return CoreToolParameters(is_tool_valid=False, msg_error=message)

    def build_tool_definition(self) -> CoreToolDefinition:
        """Return a provider-agnostic descriptor of this tool."""
        return CoreToolDefinition(
            name=f"{self.name}_v{self.version}",
            description=self.description,
            parameters=self.parameters,
        )

    # -------- PRIVATE -----------------------------------------------------------
    @classmethod
    def _check_type(cls, value: t.Any, expected_type: str) -> bool:
        """Match value to JSON schema type."""
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type in ("number", "integer") and isinstance(value, bool):
            return False
        expected = cls._JSON_TYPE_MAP.get(expected_type)
        if expected is None:
            return True
        return isinstance(value, expected)

    def _get_validator(self) -> Draft202012Validator:
        """Lazily build and cache the JSON Schema Validator"""
        if self._schema_validator is None:
            schema = self.parameters
            Draft202012Validator.check_schema(schema)
            self._schema_validator = Draft202012Validator(schema)
        return self._schema_validator

    # -------- DUNDERS -----------------------------------------------------------
    def __str__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} "
            f"name='{self.name}' "
            f"version='{self.version}' "
            f"description='{self.description}'>"
        )


class CoreRuntimeTool(CoreTool, ABC):
    """Marker base for tools that must run in an isolated runtime.

    Normal ``CoreTool`` instances are safe to execute in the agent process
    by default. Runtime tools need a filesystem/runtime environment such as
    Docker, Firecracker, or another sandbox.
    """
