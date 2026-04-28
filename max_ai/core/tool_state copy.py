"""
Tool execution state for a single run.

ToolState composes approval workflow state with tool execution results,
providing a single mutable container that tracks everything related to
tool calls during an agent run.
"""

from __future__ import annotations

import logging
import typing as t
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from ..loggers import ScopedLogger

if t.TYPE_CHECKING:
    from .payloads import (
        ToolCallRecord,
        ToolResult,
        ToolApprovalRequest,
        ToolApprovalResponse,
    )

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, component="tool_state")


# -------- -----------------------------------------------------------
# APPROVAL WORKFLOW
# -------- -----------------------------------------------------------
class ApprovalState(BaseModel):
    """Manages the approval workflow for tool calls within a single run."""

    approval_requests: list[ToolApprovalRequest] = Field(
        default_factory=list,
        description="Tool calls submitted for user approval.",
    )
    approval_responses: dict[str, ToolApprovalResponse] = Field(
        default_factory=dict,
        description="Approval responses keyed by tool_call_id.",
    )

    # -------- -----------------------------------------------------------
    # QUERIES
    # -------- -----------------------------------------------------------
    @property
    def waiting_for_approval(self) -> bool:
        """True if any registered request does not yet have a response."""
        return any(
            r.tool_call_id not in self.approval_responses
            for r in self.approval_requests
        )

    @property
    def pending_tool_calls(self) -> dict[str, ToolCallRecord]:
        """Map of tool_call_id → ToolCallRecord for all registered approval requests."""
        return {req.tool_call_id: req.tool_request for req in self.approval_requests}

    def get_approval_response(self, tool_call_id: str) -> ToolApprovalResponse | None:
        return self.approval_responses.get(tool_call_id)

    def is_approved_and_unconsumed(self, tool_call_id: str) -> bool:
        """True if the tool call was approved and has not yet been executed."""
        resp = self.approval_responses.get(tool_call_id)
        return resp is not None and resp.approved and not resp.consumed

    def get_rejected_tool_calls(self) -> list[tuple[str, ToolCallRecord]]:
        """Return rejected tool calls. Does NOT mutate state."""
        return [
            (req.tool_call_id, req.tool_request)
            for req in self.approval_requests
            if (resp := self.approval_responses.get(req.tool_call_id))
            and not resp.approved
        ]

    # -------- -----------------------------------------------------------
    # MUTATIONS
    # -------- -----------------------------------------------------------
    def build_approval_request(self, tool_call: ToolCallRecord) -> ToolApprovalRequest:
        """
        Build an approval request WITHOUT registering it.

        Used when event emission may fail: build first, then call
        `register_approval_request` only after the event is successfully
        yielded. Avoids zombie requests if emission fails.
        """
        return ToolApprovalRequest(tool_request=tool_call)

    def register_approval_request(self, request: ToolApprovalRequest) -> None:
        """Register a built approval request into state."""
        self.approval_requests.append(request)

    def add_approval_response(self, response: ToolApprovalResponse) -> None:
        """Store an approval response keyed by tool_call_id."""
        self.approval_responses[response.tool_call_id] = response

    def pop_rejected_tool_calls(self) -> list[tuple[str, ToolCallRecord]]:
        """Return rejected tool calls AND remove them from pending requests."""
        rejected = self.get_rejected_tool_calls()
        rejected_ids = {tool_call_id for tool_call_id, _ in rejected}
        self.approval_requests = [
            r for r in self.approval_requests if r.tool_call_id not in rejected_ids
        ]
        return rejected

    def mark_approval_consumed(self, tool_call_id: str) -> bool:
        """
        Mark an approval response as consumed after successful tool execution.

        Prevents re-execution of the same approved tool_call_id within the run.
        Returns False if approval does not exist or was already consumed.
        """

        approval = self.approval_responses.get(tool_call_id)
        if approval is None:
            log.child(tool_call_id=tool_call_id).warning(
                "Attempted to consume non-existent approval response"
            )
            return False

        if approval.consumed:
            log.child(tool_call_id=tool_call_id).warning(
                "Attempted to consume already consumed approval response"
            )
            return False

        approval.consumed = True
        approval.consumed_at = datetime.now(timezone.utc)
        log.child(tool_call_id=tool_call_id).info(
            "Approval response marked as consumed",
            approved=approval.approved,
            reason=approval.reason,
        )
        return True

    # -------- -----------------------------------------------------------
    # LIFECYCLE
    # -------- -----------------------------------------------------------
    @classmethod
    def load_from(
        cls,
        state: dict[str, t.Any] | None = None,
        response: ToolApprovalResponse | None = None,
    ) -> t.Self:
        """Rehydrate from serialized state, optionally applying a new response."""
        instance = cls.model_validate(state) if state else cls()
        if response:
            instance.add_approval_response(response)
        return instance

    def reset(self) -> None:
        """Clear all approval state."""
        self.approval_responses.clear()
        self.approval_requests.clear()


# -------- -----------------------------------------------------------
# TOOL STATE
# -------- -----------------------------------------------------------
class ToolState(BaseModel):
    """
    Container for all tool-related state within a single run.

    Composes:
    - approval: workflow for tool calls requiring user approval
    - completed_results: results of tools that have finished executing
    """

    approval: ApprovalState = Field(default_factory=ApprovalState)
    completed_results: list[ToolResult] = Field(default_factory=list)

    # -------- -----------------------------------------------------------
    # QUERIES
    # -------- -----------------------------------------------------------
    def get_result(self, tool_call_id: str) -> ToolResult | None:
        """Find a completed result by tool_call_id."""
        return next(
            (
                r
                for r in self.completed_results
                if r.tool_request.tool_call_id == tool_call_id
            ),
            None,
        )

    def has_result(self, tool_call_id: str) -> bool:
        return self.get_result(tool_call_id) is not None

    # -------- -----------------------------------------------------------
    # MUTATIONS
    # -------- -----------------------------------------------------------
    def record_result(self, result: ToolResult) -> None:
        """Add a completed tool result."""
        self.completed_results.append(result)

    # -------- -----------------------------------------------------------
    # LIFECYCLE
    # -------- -----------------------------------------------------------
    @classmethod
    def load_from(
        cls,
        state: dict[str, t.Any] | None = None,
        approval_response: ToolApprovalResponse | None = None,
    ) -> t.Self:
        """Rehydrate ToolState from serialized dict, optionally applying a new approval."""
        if state is None:
            instance = cls()
        else:
            instance = cls.model_validate(state)
        if approval_response:
            instance.approval.add_approval_response(approval_response)
        return instance

    def reset(self) -> None:
        """Clear all tool state."""
        self.approval.reset()
        self.completed_results.clear()
