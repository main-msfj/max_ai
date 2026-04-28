"""
Tool execution state for a single run.

ToolState tracks every tool call the assistant has made within a run,
keyed by ``tool_call_id``. Each entry is a ``ToolCallRecord`` carrying
its own approval state, execution state, and (eventually) result.
"""

from __future__ import annotations

import logging
import typing as t
from pydantic import BaseModel, Field

from ..loggers import ScopedLogger
from ..types.tool_call import ToolCallRecord, ToolResult


logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="ToolState")


# -------- -----------------------------------------------------------
# TOOL STATE
# -------- -----------------------------------------------------------
class ToolState(BaseModel):
    """All tool call records for the current run, keyed by ``record.id``.

    Lifecycle of a record (managed by ``ToolCallRecord`` itself)::

        PENDING_APPROVAL ─approve()──►       APPROVED      ─start_execution()─►  EXECUTING ─mark_consumed()─► CONSUMED
                          ╲                                                              ╲
                           reject()──►       REJECTED                                     (stays in EXECUTING if process crashes)
                          ╲
                           auto_approve()──► AUTO_APPROVED ─start_execution()─►  EXECUTING ─mark_consumed()─► CONSUMED
    """

    records: dict[str, ToolCallRecord] = Field(
        default_factory=dict,
        description="Tool call records keyed by record id.",
    )

    # -------- QUERIES -----------------------------------------------------------
    def get(self, tool_call_id: str) -> ToolCallRecord | None:
        return self.records.get(tool_call_id)

    def has(self, tool_call_id: str) -> bool:
        return tool_call_id in self.records

    @property
    def waiting_for_approval(self) -> bool:
        """True if any record is still pending user approval."""
        return any(r.is_pending_approval for r in self.records.values())

    @property
    def pending_approvals(self) -> list[ToolCallRecord]:
        return [r for r in self.records.values() if r.is_pending_approval]

    @property
    def actionable_calls(self) -> list[ToolCallRecord]:
        """Records approved (manually or auto) and not yet started executing."""
        return [r for r in self.records.values() if r.is_actionable]

    @property
    def rejected_calls(self) -> list[ToolCallRecord]:
        return [r for r in self.records.values() if r.is_rejected]

    @property
    def executing_calls(self) -> list[ToolCallRecord]:
        """Records currently mid-execution within this process."""
        return [r for r in self.records.values() if r.is_executing]

    @property
    def stale_executions(self) -> list[ToolCallRecord]:
        """Records left in ``EXECUTING`` from a previous run.

        On rehydration, the agent should decide what to do with these
        (mark failed, retry, ask user) before continuing the run.
        """
        return [r for r in self.records.values() if r.is_stale_execution]

    @property
    def consumed_calls(self) -> list[ToolCallRecord]:
        return [r for r in self.records.values() if r.is_consumed]

    def get_result(self, tool_call_id: str) -> ToolResult | None:
        record = self.records.get(tool_call_id)
        return record.result if record else None

    # -------- MUTATIONS -----------------------------------------------------------
    def add(self, record: ToolCallRecord) -> ToolCallRecord:
        """Track a new tool call record. Returns the record for chaining.

        Raises:
            ValueError: If a record with the same id is already tracked.
        """
        if record.id in self.records:
            raise ValueError(
                f"Tool call {record.id} is already tracked in this run."
            )
        self.records[record.id] = record
        return record

    def apply_approval(
        self,
        tool_call_id: str,
        approved: bool,
        reason: str | None = None,
    ) -> ToolCallRecord:
        """Apply a user approval decision. Returns the updated record.

        Raises:
            KeyError: If no record exists with that id.
            ValueError: If the record is not in PENDING_APPROVAL state.
        """
        record = self.records.get(tool_call_id)
        if record is None:
            raise KeyError(
                f"Cannot apply approval — no tool call {tool_call_id} tracked."
            )
        if approved:
            record.approve(reason)
        else:
            record.reject(reason)
        log.child(tool_call_id=tool_call_id).info(
            "Approval applied",
            approved=approved,
            reason=record.approval_reason,
        )
        return record

    def consume(self, tool_call_id: str, result: ToolResult) -> ToolCallRecord:
        """Mark an executing record as consumed with its result.

        The record must be in ``EXECUTING`` state — call
        ``record.start_execution()`` before invoking this. Returns
        the updated record.

        Raises:
            KeyError: If no record exists with that id.
            ValueError: If the record is not in EXECUTING state, or
                already CONSUMED, or the result's ``tool_call_id``
                does not match.
        """
        record = self.records.get(tool_call_id)
        if record is None:
            raise KeyError(f"Cannot consume — no tool call {tool_call_id} tracked.")
        record.mark_consumed(result)
        log.child(tool_call_id=tool_call_id).info(
            "Tool call consumed",
            success=result.success,
            duration_ms=result.duration_ms,
        )
        return record

    def fail_stale(
        self, tool_call_id: str, reason: str | None = None
    ) -> ToolCallRecord:
        """Force-transition a stale ``EXECUTING`` record to a failed ``CONSUMED``.

        Used at rehydration time for records that were mid-execution
        when the previous run died. See
        ``ToolCallRecord.mark_stale_as_failed`` for details.

        Raises:
            KeyError: If no record exists with that id.
            ValueError: If the record is not in a stale (EXECUTING) state.
        """
        record = self.records.get(tool_call_id)
        if record is None:
            raise KeyError(
                f"Cannot fail stale — no tool call {tool_call_id} tracked."
            )
        record.mark_stale_as_failed(reason)
        log.child(tool_call_id=tool_call_id).warning(
            "Stale execution force-failed at rehydration",
            reason=reason,
        )
        return record

    # -------- LIFECYCLE -----------------------------------------------------------
    @classmethod
    def load_from(
        cls,
        state: dict[str, t.Any] | None = None,
        approval: tuple[str, bool, str | None] | None = None,
    ) -> t.Self:
        """Rehydrate from serialized state, optionally applying a new approval.

        Args:
            state: Serialized ``ToolState`` dict.
            approval: Optional ``(tool_call_id, approved, reason)`` tuple
                applied right after rehydration. Useful when resuming a
                run after the user just answered a pending approval.
        """
        instance = cls.model_validate(state) if state else cls()
        if approval is not None:
            tool_call_id, approved, reason = approval
            instance.apply_approval(tool_call_id, approved, reason)
        return instance

    def reset(self) -> None:
        """Clear all tracked tool calls."""
        self.records.clear()