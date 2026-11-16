"""
Core abstractions for LLM agent tools.

Defines base classes and interfaces that enable agents
to execute external actions (e.g., APIs, file I/O, services).
"""

import re
import typing as t
from abc import ABC, abstractmethod

from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel

from ..core.termination import CancellationToken
from ..types.tool_call import ToolCallRecord, ToolResult
from ..types.tools import (
    CoreToolDefinition,
    CoreToolParameters,
    ToolApprovalMode,
    ToolKind,
)
from .component import ComponentBase

if t.TYPE_CHECKING:
    from ..types.run_context import RunContext
    from .sandbox import SandboxCore


class ToolContext:
    """What a tool has at hand while it runs: the conversation, the sandbox
    it may use and how to tell the outside what happens. The dispatcher
    builds it; who it is for comes from ``run_context``."""

    __slots__ = ("run_context", "sandbox", "emit_event", "deps", "cancellation")

    def __init__(
        self,
        run_context: "RunContext",
        sandbox: "SandboxCore | None" = None,
        emit_event: t.Callable[[t.Any], None] | None = None,
        deps: dict[str, t.Any] | None = None,
        cancellation: CancellationToken | None = None,
    ) -> None:
        self.run_context = run_context
        self.sandbox = sandbox  # already bound to the user
        self.emit_event = emit_event
        self.deps = deps or {}  # the developer's own objects
        self.cancellation = cancellation  # for tools that leave something running

    @property
    def cancelled(self) -> bool:
        """The user cancelled: stop at the next clean point."""
        return self.cancellation is not None and self.cancellation.is_cancelled()

    @property
    def user_id(self) -> str:
        return self.run_context.user_id

    @property
    def session_id(self) -> str:
        return self.run_context.session_id or ""

    @property
    def run_id(self) -> str:
        return self.run_context.run_id


class CoreTool(ComponentBase[BaseModel], ABC):
    """Base class for all agent tools.

    A tool checks its own name and description when it is built, long
    before any agent uses it: a tool that cannot be used never starts one.
    """

    component_type = "tool"
    NAME_PATTERN: t.ClassVar[str] = r"[a-z][a-z0-9_]{0,63}"

    kind: t.ClassVar[ToolKind] = ToolKind.NATIVE
    sandbox: t.ClassVar[bool] = False  # True: runs in the sandbox. False: on the host.
    waits_for_user: t.ClassVar[bool] = False  # The result comes from a person.
    instructions: t.ClassVar[str] = ""  # How to use it; goes to the prompt.
    timeout: float | None = None  # Seconds one call may run; None: the dispatcher's.

    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
        read_only: bool = False,
        deferred: bool = False,
    ):
        """
        Parameters
        ----------
        name : str
            What the model calls. snake_case, at most 64 characters, except
            for an MCP tool, whose server names it.
        description : str
            What the tool does, in a few words. How to use it goes in
            ``instructions``.
        approval_mode : ToolApprovalMode | str
            Whether a call asks the user first. The dispatcher's hook may
            decide otherwise.
        read_only : bool
            The tool changes nothing: it only reads. It is a fact about the
            tool, not a permission. Two things use it: the dispatcher runs
            consecutive read-only calls in parallel, and MCP sends a
            read-only call again when the connection drops mid-call.
        deferred : bool
            Not sent to the model: it is listed in the prompt, found with
            ``search_tools`` and run with ``call_tool``.
        """
        self.name = name
        self.version = version
        self.description = description
        self.approval_mode = approval_mode
        self.read_only = read_only
        self.deferred = deferred
        self._schema_validator: Draft202012Validator | None = None  # built on first use
        self._check_definition()

    # -------- DEFINITION --------------------------------------------------------
    def _check_definition(self) -> None:
        """Everything a tool must have to be usable; raises on the first miss."""
        self.required_name()
        self.required_description()

    def required_name(self) -> None:
        """Raise unless the tool has a usable name. A server names its own
        tools, so an MCP tool only needs to have one."""
        if not self.name:
            raise ToolDefinitionError("The tool has no name.")
        if self.kind is not ToolKind.MCP and not re.fullmatch(self.NAME_PATTERN, self.name):
            raise ToolDefinitionError(
                f"Tool name {self.name!r} must be snake_case, at most 64 characters."
            )

    def required_description(self) -> None:
        """Raise unless the tool says what it does."""
        if not self.description or not self.description.strip():
            raise ToolDefinitionError(f"Tool {self.name!r} has no description.")

    @property
    @abstractmethod
    def parameters(self) -> dict[str, t.Any]:
        """JSON schema for tool inputs."""
        ...

    def unwrap(self, parameters: dict[str, t.Any]) -> tuple[str, dict[str, t.Any]] | None:
        """The tool and arguments this call really runs; None runs this tool."""
        return None

    # -------- ABSTRACT -----------------------------------------------------------
    @abstractmethod
    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        """Run the tool and return what happened.

        The dispatcher already stops a cancelled call from outside. Look at
        ``tool_ctx.cancelled`` only when stopping is not enough: a process
        left running, a task on a server, work half done.
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

        # All the errors, sorted by path: the model can fix them in one retry.
        # jsonschema's own type hints leave iter_errors partly untyped.
        found: t.Iterable[ValidationError] = validator.iter_errors(  # pyright: ignore[reportUnknownMemberType]
            tool_request.parameters
        )
        errors = sorted(found, key=lambda e: list(e.absolute_path))
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
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            sandbox=self.sandbox,
        )

    # -------- PRIVATE -----------------------------------------------------------
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


class ToolDefinitionError(ValueError):
    """A tool was built with a name or a description that cannot be used."""
