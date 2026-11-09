import typing as t
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from ..core.ids import short_id
from ..core.primitives import FailureReason, ToolCallStatus


# -------- TOOL RESULT -----------------------------------------------------------
class ToolResult(BaseModel):
    """Tool execution result to be returned to the LLM.

    A ``ToolResult`` is always produced by executing a single
    ``ToolCallRecord``. We store ``tool_call_id`` (not the full record)
    because the result lives **inside** its parent record at
    ``ToolCallRecord.result`` — duplicating identity/tracking fields
    would just be noise.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(...)
    error: str | None = Field(default=None)
    result: t.Any | None = Field(default=None, description="Tool execution output")
    failure_reason: FailureReason | None = Field(default=None)

    # -------- TRACKING -----------------------------------------------------------
    tool_call_id: str = Field(
        ...,
        description="ID of the ToolCallRecord that produced this result.",
    )
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = Field(default=None)
    metadata: dict[str, t.Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> int | None:
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds() * 1000)
        return None

    # -------- FACTORIES -----------------------------------------------------------
    @classmethod
    def success_result(
        cls,
        tool_call_id: str,
        result: t.Any,
        metadata: dict[str, t.Any] | None = None,
    ) -> t.Self:
        """Factory for a successful ``ToolResult``."""
        return cls(
            success=True,
            result=result,
            error=None,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
            completed_at=datetime.now(timezone.utc),
        )

    @classmethod
    def tool_failure(
        cls,
        tool_call_id: str,
        error: str,
        reason: FailureReason = FailureReason.EXECUTION_ERROR,
        metadata: dict[str, t.Any] | None = None,
    ) -> t.Self:
        """Factory for a failed ``ToolResult`` with a categorized reason."""
        return cls(
            success=False,
            result=None,
            error=error,
            failure_reason=reason,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
            completed_at=datetime.now(timezone.utc),
        )

    # -------- CONVENIENCE FACTORIES FOR COMMON FAILURES -----------------------------------------------------------
    @classmethod
    def cancelled_before_start(cls, tool_call_id: str) -> t.Self:
        return cls.tool_failure(
            tool_call_id,
            error="Tool was cancelled before it started.",
            reason=FailureReason.CANCELLED_BEFORE_START,
        )

    @classmethod
    def cancelled_during_execution(cls, tool_call_id: str) -> t.Self:
        return cls.tool_failure(
            tool_call_id,
            error="Tool was cancelled during execution.",
            reason=FailureReason.CANCELLED_DURING_EXECUTION,
        )

    @classmethod
    def invalid_parameters(cls, tool_call_id: str, val_err: str) -> t.Self:
        return cls.tool_failure(
            tool_call_id,
            error=f"Invalid parameters: {val_err}",
            reason=FailureReason.INVALID_PARAMETERS,
        )

    @classmethod
    def execution_error(
        cls, tool_call_id: str, err_msg: str | None = None
    ) -> t.Self:
        return cls.tool_failure(
            tool_call_id,
            error=err_msg or "Sorry, an error occurred while processing the request.",
            reason=FailureReason.EXECUTION_ERROR,
        )

    @classmethod
    def approval_denied(cls, tool_call_id: str, err_msg: str | None = None) -> t.Self:
        """Factory for a tool blocked by a denial — explicit user rejection or
        static permission policy. Distinct from ``execution_error`` so callers
        (the loop, in particular) can single out "won't run" from "broke"."""
        return cls.tool_failure(
            tool_call_id,
            error=err_msg or "Tool call was denied.",
            reason=FailureReason.APPROVAL_DENIED,
        )

    @classmethod
    def timeout(cls, tool_call_id: str, timeout_seconds: float) -> t.Self:
        return cls.tool_failure(
            tool_call_id,
            error=f"Tool execution exceeded timeout of {timeout_seconds} seconds.",
            reason=FailureReason.TIMEOUT,
        )


# -------- TOOL CALL RECORD -----------------------------------------------------------
class ToolCallRecord(BaseModel):
    """Everything we know about one tool call within a run.

    Lifecycle::

        PENDING_APPROVAL ─approve()──►       APPROVED      ─start_execution()─►  EXECUTING ─mark_consumed()─► CONSUMED
                          ╲                                                              ╲
                           reject()──►       REJECTED                                     (stays here if process crashes)
                          ╲
                           auto_approve()──► AUTO_APPROVED ─start_execution()─►  EXECUTING ─mark_consumed()─► CONSUMED
                          ╲
                           await_user_input()──► INPUT_NEEDED ─store_result()─► APPROVED (result stored)

    A tool with ``waits_for_user`` is answered by a person: the record moves
    to ``INPUT_NEEDED`` and the turn pauses. ``store_result()`` keeps what
    they produced and moves the record to ``APPROVED``, so the resumed run
    delivers it. The pause is record state: it survives process death.

    Records left in ``EXECUTING`` after a process crash or connection
    drop are visible via ``ToolState.stale_executions``. The agent
    decides what to do with them on rehydration via
    ``mark_stale_as_failed()`` (default policy) or by resetting
    them back to approved (custom policy).

    Cancellation and timeout are NOT separate statuses — they finish
    via ``mark_consumed()`` with a ``ToolResult`` whose ``success``
    is ``False`` and ``failure_reason`` is ``CANCELLED_*`` or
    ``TIMEOUT``. The status remains ``CONSUMED``.
    """

    # -------- IDENTITY -----------------------------------------------------------
    id: str = Field(default_factory=short_id)
    tool_name: str = Field(...)
    parameters: dict[str, t.Any] = Field(default_factory=dict)

    # -------- CLASSIFICATION -----------------------------------------------------------
    tool_type: t.Literal["mcp", "custom", "skill"] = Field(default="custom")

    # -------- TRACKING -----------------------------------------------------------
    session_id: str | None = Field(default=None)
    requested_by: str = Field(default="assistant")
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # -------- APPROVAL LIFECYCLE -----------------------------------------------------------
    status: ToolCallStatus = Field(default=ToolCallStatus.PENDING_APPROVAL)
    approval_reason: str | None = Field(
        default=None,
        description="Why approval was requested, why the user rejected, or why the user approved.",
    )
    approval_decided_at: datetime | None = Field(default=None)
    was_auto_approved: bool = Field(
        default=False,
        description="True if approval was granted automatically (tool didn't require user action).",
    )

    # -------- ELICITATION (ask-the-user) -----------------------------------------------------------

    # -------- EXECUTION -----------------------------------------------------------
    started_at: datetime | None = Field(
        default=None,
        description="When the tool transitioned to EXECUTING.",
    )
    consumed_at: datetime | None = Field(default=None)
    result: ToolResult | None = Field(
        default=None,
        description="Set after the tool ran, or by a person for a waits_for_user tool.",
    )

    # -------- TAG-ALONG DATA -----------------------------------------------------------
    metadata: dict[str, t.Any] = Field(default_factory=dict)

    # -------- DERIVED PROPERTIES -----------------------------------------------------------
    @property
    def is_pending_approval(self) -> bool:
        return self.status == ToolCallStatus.PENDING_APPROVAL

    @property
    def is_approved(self) -> bool:
        """Approved (manually or auto), not yet started executing."""
        return self.status in (
            ToolCallStatus.APPROVED,
            ToolCallStatus.AUTO_APPROVED,
        )

    @property
    def is_rejected(self) -> bool:
        return self.status == ToolCallStatus.REJECTED

    @property
    def is_awaiting_input(self) -> bool:
        """Paused waiting for the user to answer a question."""
        return self.status == ToolCallStatus.INPUT_NEEDED

    @property
    def is_executing(self) -> bool:
        return self.status == ToolCallStatus.EXECUTING

    @property
    def is_consumed(self) -> bool:
        return self.status == ToolCallStatus.CONSUMED

    @property
    def is_actionable(self) -> bool:
        """Ready to start executing: approved and not yet started."""
        return self.is_approved

    @property
    def is_stale_execution(self) -> bool:
        """Was executing but never reached terminal state. Likely from a crashed run."""
        return self.is_executing

    # -------- TRANSITIONS -----------------------------------------------------------
    def approve(self, reason: str | None = None) -> t.Self:
        """Move ``PENDING_APPROVAL`` → ``APPROVED``. Returns self for chaining."""
        if not self.is_pending_approval:
            raise ValueError(
                f"Tool call {self.id} cannot be approved from status {self.status}."
            )
        self.status = ToolCallStatus.APPROVED
        self.approval_reason = reason
        self.approval_decided_at = datetime.now(timezone.utc)
        return self

    def auto_approve(self) -> t.Self:
        """Move directly to ``AUTO_APPROVED`` (tool didn't require user approval). Returns self."""
        if not self.is_pending_approval:
            raise ValueError(
                f"Tool call {self.id} cannot be auto-approved from status {self.status}."
            )
        self.status = ToolCallStatus.AUTO_APPROVED
        self.was_auto_approved = True
        self.approval_decided_at = datetime.now(timezone.utc)
        return self

    def reject(self, reason: str | None = None) -> t.Self:
        """Move ``PENDING_APPROVAL`` → ``REJECTED``. Returns self."""
        if not self.is_pending_approval:
            raise ValueError(
                f"Tool call {self.id} cannot be rejected from status {self.status}."
            )
        self.status = ToolCallStatus.REJECTED
        self.approval_reason = reason or "Rejected from UI."
        self.approval_decided_at = datetime.now(timezone.utc)
        return self

    def await_user_input(self) -> t.Self:
        """Move ``PENDING_APPROVAL`` → ``INPUT_NEEDED``. Returns self."""
        if not self.is_pending_approval:
            raise ValueError(
                f"Tool call {self.id} cannot await user input from status "
                f"{self.status}."
            )
        self.status = ToolCallStatus.INPUT_NEEDED
        return self

    def store_result(self, result: ToolResult) -> t.Self:
        """Move ``INPUT_NEEDED`` → ``APPROVED``, keeping a person's result.

        Not consumed yet: only unconsumed records are picked up on resume,
        and that is what gives the call its ``ToolMessage``.
        """
        if not self.is_awaiting_input:
            raise ValueError(
                f"Tool call {self.id} cannot store a result from status "
                f"{self.status} (must be INPUT_NEEDED)."
            )
        if result.tool_call_id != self.id:
            raise ValueError(
                f"Result.tool_call_id ({result.tool_call_id}) does not "
                f"match record id ({self.id})."
            )
        self.status = ToolCallStatus.APPROVED
        self.result = result
        self.approval_decided_at = datetime.now(timezone.utc)
        return self

    def start_execution(self) -> t.Self:
        """Move ``APPROVED``/``AUTO_APPROVED`` → ``EXECUTING``. Returns self."""
        if not self.is_actionable:
            raise ValueError(
                f"Tool call {self.id} cannot start execution from status {self.status}."
            )
        self.status = ToolCallStatus.EXECUTING
        self.started_at = datetime.now(timezone.utc)
        return self

    def mark_consumed(self, result: ToolResult) -> t.Self:
        """Move ``EXECUTING`` → ``CONSUMED`` with a result. Returns self.

        The ``result`` may carry ``success=True`` (normal completion)
        or ``success=False`` (timeout, cancellation, validation error,
        runtime exception — captured by the executor's try/except).
        Either way the status moves to ``CONSUMED``.
        """
        if self.is_consumed:
            raise ValueError(f"Tool call {self.id} already consumed.")
        if not self.is_executing:
            raise ValueError(
                f"Tool call {self.id} cannot be consumed from status "
                f"{self.status} (must be EXECUTING)."
            )
        if result.tool_call_id != self.id:
            raise ValueError(
                f"Result.tool_call_id ({result.tool_call_id}) does not "
                f"match record id ({self.id})."
            )
        self.status = ToolCallStatus.CONSUMED
        self.consumed_at = datetime.now(timezone.utc)
        self.result = result
        return self

    def mark_stale_as_failed(self, reason: str | None = None) -> t.Self:
        """Force-transition a stale ``EXECUTING`` record to a failed ``CONSUMED``.

        Used during rehydration when a record is found in ``EXECUTING``
        but the previous run died. Bypasses the normal
        ``EXECUTING``-required check of ``mark_consumed`` because, by
        definition, we don't know how the previous execution actually
        ended — the only safe assumption is that it failed.

        The synthesized ``ToolResult`` has ``success=False`` and
        ``failure_reason=EXECUTION_ERROR``. The LLM sees this in the
        next turn as a normal tool failure and reacts accordingly
        (retry, ask user, give up).

        Returns self for chaining.
        """
        if not self.is_stale_execution:
            raise ValueError(
                f"Tool call {self.id} is not in a stale state (status={self.status})."
            )
        failure = ToolResult.tool_failure(
            tool_call_id=self.id,
            error=reason or "Tool execution did not complete (process or connection lost).",
            reason=FailureReason.EXECUTION_ERROR,
        )
        self.status = ToolCallStatus.CONSUMED
        self.consumed_at = datetime.now(timezone.utc)
        self.result = failure
        return self

    def force_consume(self, result: ToolResult) -> t.Self:
        """Force-transition a non-EXECUTING record to CONSUMED.

        Used by the executor for paths that never reached EXECUTING
        — rejected approvals, validation failures, tool-not-found.
        These records still need a terminal result for downstream
        consumers (persistence, response building) but ``mark_consumed``
        rejects them because it requires EXECUTING.
        """
        if self.is_consumed:
            raise ValueError(f"Tool call {self.id} already consumed.")
        if self.is_executing:
            raise ValueError(
                f"Tool call {self.id} is EXECUTING — use mark_consumed() instead."
            )
        if result.tool_call_id != self.id:
            raise ValueError(
                f"Result.tool_call_id ({result.tool_call_id}) does not "
                f"match record id ({self.id})."
            )
        self.status = ToolCallStatus.CONSUMED
        self.consumed_at = datetime.now(timezone.utc)
        self.result = result
        return self