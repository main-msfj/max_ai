"""Spans of a turn: what ran, when and for how long, built from the events of a run.

The CLI draws them as bars in the terminal and the web as an SVG; both read
the same ``Span`` list from one ``SpanProjector``.
"""

from __future__ import annotations

import typing as t
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ..types.completions import Usage
from .event_type import (
    CompactionEvent,
    CoreEvent,
    FatalErrorEvent,
    ModelCallEvent,
    ModelResponseEvent,
    ModelRetryEvent,
    ModelStreamChunkEvent,
    ReasoningCompleteEvent,
    ReasoningIterationEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
)
from .ids import short_id

SpanKind = t.Literal["turn", "step", "model", "tool", "compaction"]
SpanStatus = t.Literal["running", "ok", "error", "cancelled"]


class Span(BaseModel):
    """One thing that took time: a turn, a step of it, a model call, a tool or a compaction."""

    id: str = Field(default_factory=short_id)  # A tool span uses its tool_call_id
    parent_id: str | None = None  # None only for a turn
    kind: SpanKind
    name: str  # Turn task, "step 2", model name or tool name
    started_at: datetime
    ended_at: datetime | None = None  # None while it runs
    first_token_at: datetime | None = None  # Model only: the first streamed piece
    status: SpanStatus = "running"
    retries: int = 0  # Model only: tries that failed before the answer
    usage: Usage | None = None  # Model and compaction calls
    input: t.Any = None  # Tool parameters, or how many messages the model saw
    output: t.Any = None  # Model text, tool result or error, compaction summary

    @property
    def duration_ms(self) -> int | None:
        if self.ended_at is None:
            return None
        return int((self.ended_at - self.started_at).total_seconds() * 1000)

    @property
    def ttft_ms(self) -> int | None:
        if self.first_token_at is None:
            return None
        return int((self.first_token_at - self.started_at).total_seconds() * 1000)


class SpanProjector:
    """Turns the events of a run into spans: turn → step → model / tool, and compactions.

    The host calls ``start_turn`` when the user sends a message, ``feed`` with
    every event, and ``end_turn`` if the turn is cut without a closing event (esc).
    """

    def __init__(self) -> None:
        self.spans: dict[str, Span] = {}  # Every span by id, in the order it opened
        self.turn: Span | None = None  # The turn running now
        self.step: Span | None = None  # The step running now
        self.model: Span | None = None  # The model call waiting for its answer
        self.compaction: Span | None = None  # The compaction running now

    def start_turn(self, task: str | None, at: datetime | None = None) -> Span:
        """Open a turn; a turn still open is closed as cancelled."""
        if self.turn is not None:
            self.end_turn("cancelled", at)
        self.turn = self._open("turn", task or "turn", None, at or datetime.now(timezone.utc))
        return self.turn

    def end_turn(self, status: SpanStatus, at: datetime | None = None) -> None:
        """Close the turn and whatever still runs inside it."""
        if self.turn is None:
            return
        at = at or datetime.now(timezone.utc)

        # The open step ends with the turn; a model or tool still running was cut.
        for span in self.children(self.turn.id, deep=True):
            if span.status != "running":
                continue
            cut = span.kind != "step" and status == "ok"
            self._close(span, at, "cancelled" if cut else status)
        self._close(self.turn, at, status)
        self.turn = self.step = self.model = self.compaction = None

    def feed(self, event: CoreEvent) -> Span | None:
        """Apply one event; return the span it opened or changed, if any."""
        at = event.timestamp

        # A run that started without start_turn still gets its turn.
        if self.turn is None and not isinstance(event, ReasoningCompleteEvent):
            self.start_turn(None, at)
        turn = self.turn

        if isinstance(event, ReasoningIterationEvent):
            # A new step closes the one before.
            if self.step is not None:
                self._close(self.step, at, "ok")
            self.step = self._open("step", f"step {event.iteration}", self._parent(), at)
            return self.step

        if isinstance(event, ModelCallEvent):
            self.model = self._open("model", event.model, self._parent(), at)
            self.model.input = len(event.input_messages)
            return self.model

        if isinstance(event, ModelStreamChunkEvent):
            # Only the first piece matters: it marks the end of the wait.
            if self.model is not None and self.model.first_token_at is None:
                self.model.first_token_at = at
                return self.model
            return None

        if isinstance(event, ModelRetryEvent):
            if self.model is not None:
                self.model.retries += 1
            return self.model

        if isinstance(event, ModelResponseEvent):
            span, self.model = self.model, None
            if span is None:
                return None
            span.usage, span.output = event.usage, event.response
            return self._close(span, at, "ok")

        if isinstance(event, ToolCallEvent):
            span = self._open("tool", event.tool_name, self._parent(), at, event.tool_call_id)
            span.input = event.parameters
            return span

        if isinstance(event, ToolCallResponseEvent):
            span = self.spans.get(event.tool_call_id)
            if span is None:
                return None
            result = event.tool_result
            if result is None:
                return self._close(span, at, "error")
            # The result knows when the tool really ran (without the approval wait).
            span.started_at = result.started_at
            span.output = result.result if result.success else result.error
            return self._close(span, result.completed_at or at, "ok" if result.success else "error")

        if isinstance(event, CompactionEvent):
            if event.phase == "start":
                self.compaction = self._open("compaction", event.strategy, self._parent(), at)
                self.compaction.input = event.tokens_before
                return self.compaction
            span, self.compaction = self.compaction, None
            if span is None:
                return None
            span.usage, span.output = event.usage, event.summary
            return self._close(span, at, "ok")

        if isinstance(event, FatalErrorEvent):
            if turn is not None:
                turn.output = event.error_message
            self.end_turn("error", at)
            return turn

        if isinstance(event, ReasoningCompleteEvent):
            status = self.FINISH_STATUS.get(event.finish_reason, "ok")
            self.end_turn(status, at)
            return turn

        return None

    def children(self, span_id: str, deep: bool = False) -> list[Span]:
        """The spans opened inside ``span_id``; with ``deep``, theirs too."""
        found = [s for s in self.spans.values() if s.parent_id == span_id]
        if deep:
            found += [d for s in list(found) for d in self.children(s.id, deep=True)]
        return found

    def turns(self) -> list[Span]:
        return [s for s in self.spans.values() if s.kind == "turn"]

    FINISH_STATUS: t.ClassVar[dict[str, SpanStatus]] = {"error": "error", "cancelled": "cancelled"}

    def _parent(self) -> str | None:
        """Where a new span goes: the step if there is one, else the turn."""
        if self.step is not None:
            return self.step.id
        return self.turn.id if self.turn is not None else None

    def _open(
        self, kind: SpanKind, name: str, parent_id: str | None, at: datetime,
        span_id: str | None = None,
    ) -> Span:
        span = Span(kind=kind, name=name, parent_id=parent_id, started_at=at)
        if span_id is not None:
            span.id = span_id
        self.spans[span.id] = span
        return span

    @staticmethod
    def _close(span: Span, at: datetime, status: SpanStatus) -> Span:
        span.ended_at, span.status = at, status
        return span
