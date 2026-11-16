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

from ...loggers import ScopedLogger
from ...types.tool_call import ToolCallRecord, ToolResult

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["ToolState"])


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

    def get(self, tool_call_id: str) -> ToolCallRecord | None:
        return self.records.get(tool_call_id)

    @property
    def waiting_for_approval(self) -> bool:
        """True if any record is still pending user approval."""
        return any(r.is_pending_approval for r in self.records.values())

    @property
    def pending_approvals(self) -> list[ToolCallRecord]:
        return [r for r in self.records.values() if r.is_pending_approval]

    @property
    def waiting_for_input(self) -> bool:
        """True if any record is paused waiting for a user answer."""
        return any(r.is_awaiting_input for r in self.records.values())

    @property
    def pending_user_input(self) -> list[ToolCallRecord]:
        """Records in ``INPUT_NEEDED`` — questions the user hasn't answered."""
        return [r for r in self.records.values() if r.is_awaiting_input]

    @property
    def stale_executions(self) -> list[ToolCallRecord]:
        """Records left in ``EXECUTING`` from a previous run. On rehydration,
        decide what to do with these (``fail_stale``) before continuing."""
        return [r for r in self.records.values() if r.is_stale_execution]

    # -------- CHANGES -----------------------------------------------------------
    # Running a call and closing it is the record's own work (the dispatcher
    # calls it). Here is only what the application does from outside.
    def add(self, record: ToolCallRecord) -> ToolCallRecord:
        """Track a new tool call record. ValueError if its id is already tracked."""
        if record.id in self.records:
            raise ValueError(f"Tool call {record.id} is already tracked in this run.")
        self.records[record.id] = record
        return record

    def apply_approval(
        self, tool_call_id: str, approved: bool, reason: str | None = None
    ) -> ToolCallRecord:
        """Apply a user's approval decision. KeyError if the call is not
        tracked; ValueError if it is not waiting for approval."""
        record = self._record(tool_call_id)
        decide = record.approve if approved else record.reject
        decide(reason)
        log.child(tool_call_id=tool_call_id).info(
            "Approval applied", approved=approved, reason=record.approval_reason
        )
        return record

    def provide_result(self, tool_call_id: str, result: ToolResult) -> ToolCallRecord:
        """Keep the result a person produced; the next resume delivers it.
        KeyError if the call is not tracked; ValueError if it asked nothing."""
        record = self._record(tool_call_id)
        record.store_result(result)
        log.child(tool_call_id=tool_call_id).info("Result provided by user")
        return record

    def fail_stale(self, tool_call_id: str, reason: str | None = None) -> ToolCallRecord:
        """Close as failed a record that was mid-execution when the previous
        run died. KeyError if not tracked; ValueError if it is not stale."""
        record = self._record(tool_call_id)
        record.mark_stale_as_failed(reason)
        log.child(tool_call_id=tool_call_id).warning(
            "Stale execution force-failed at rehydration", reason=reason
        )
        return record

    def _record(self, tool_call_id: str) -> ToolCallRecord:
        record = self.records.get(tool_call_id)
        if record is None:
            raise KeyError(f"No tool call {tool_call_id} tracked in this run.")
        return record

    @classmethod
    def load_from(cls, state: dict[str, t.Any] | None = None) -> t.Self:
        """Rebuild from a serialized ``ToolState``; empty when there is none."""
        return cls.model_validate(state) if state else cls()

    def reset(self) -> None:
        """Clear all tracked tool calls."""
        self.records.clear()
