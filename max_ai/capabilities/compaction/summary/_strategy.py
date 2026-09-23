"""Summarize what leaves the window; keep the recent messages raw.

    window = [system prompt + summary] + [recent messages] + [answer] + [margin]

The summary lives in ``ctx.compaction.state["summary"]`` (it travels with
the session) and is rebuilt incrementally: each compaction merges the
previous summary with the messages leaving the window.
"""

from __future__ import annotations

from pydantic import JsonValue

from ....base.clients import CoreChatCompletionClient
from ....base.compaction import CoreCompaction
from ....core.compaction import (
    CompactionOutput,
    CompactionResult,
    MessageGroup,
    split_recent_messages,
    turn_starts,
)
from ....core.messages import CoreMessage, UserMessage
from ....types.run_context import RunContext
from ....types.stacks import PromptCtx
from ._model import SummaryCompactionConfig

SUMMARY_TASK = (
    "Create an updated summary of this conversation.\n"
    "Merge the previous summary with the newer messages below. The newer "
    "messages win over the previous summary when they conflict. Keep the "
    "objectives, pending work, completed work, decisions and every fact "
    "needed to continue. Return only the requested structured output.\n\n"
    "Previous summary:\n{previous}\n\n"
    "Newer messages to merge:\n{transcript}"
)

_SECTIONS = (
    ("objective", "Objectives"),
    ("pending", "Pending"),
    ("successfully_done", "Done"),
    ("decisions", "Decisions"),
    ("important_context", "Important context"),
)


class SummaryCompaction(CoreCompaction):
    """Keep ``[summary] + [recent messages]`` once the window fills up.

    Triggers on the token threshold. Older blocks are folded into the
    summary with one LLM call (more if they don't fit one request); the
    recent blocks that fit the budget stay raw. ``summary_max_tokens`` is
    reserved from that budget because the summary itself goes back into
    the system prompt.
    """

    component_schema = SummaryCompactionConfig
    component_provider_override = "maxai.compaction.SummaryCompaction"

    def __init__(
        self, *, summary_max_tokens: int = 2000, message_cap_tokens: int = 2000, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.config = SummaryCompactionConfig(
            **self.config.model_dump(),
            summary_max_tokens=summary_max_tokens,
            message_cap_tokens=message_cap_tokens,
        )

    async def _compact(
        self,
        blocks: list[MessageGroup],
        *,
        state: dict[str, JsonValue],
        budget_tokens: int,
        client: CoreChatCompletionClient,
    ) -> CompactionResult:
        cut = self._split_index(blocks, max(1, budget_tokens - self.config.summary_max_tokens))
        old = [m for block in blocks[:cut] for m in block.messages]
        recent = [m for block in blocks[cut:] for m in block.messages]
        if not old:
            return CompactionResult(messages=recent, state=state)

        summary = await self._summarize(old, self._summary(state), client)
        state["summary"] = summary.model_dump(exclude_defaults=True)
        return CompactionResult(changed=True, messages=recent, old_messages=old, state=state)

    def _split_index(self, blocks: list[MessageGroup], budget: int) -> int:
        """First block kept raw: the newest blocks that fit ``budget``,
        moved forward to a turn start so the window never opens with an
        answer whose question was summarized."""
        _, recent = split_recent_messages(
            blocks, max_tokens=budget, min_keep_groups=self.config.min_keep_groups,
        )
        cut, kept = len(blocks), 0
        while cut > 0 and kept < len(recent):
            cut -= 1
            kept += len(blocks[cut].messages)
        later_turns = [start for start in turn_starts(blocks) if start >= cut]
        if later_turns and len(blocks) - later_turns[0] >= self.config.min_keep_groups:
            cut = later_turns[0]
        return cut

    def render(self, state: dict[str, JsonValue]) -> str | None:
        summary = self._summary(state)
        if summary is None:
            return None
        lines = ["<conversation_summary>",
                 "Summary of the earlier conversation. The live messages win if they conflict."]
        if summary.summary:
            lines.append(summary.summary)
        for field, title in _SECTIONS:
            items = getattr(summary, field)
            if items:
                lines.append(f"{title}:")
                lines.extend(f"- {item}" for item in items)
        lines.append("</conversation_summary>")
        return "\n".join(lines)

    # -------- SUMMARIZATION -----------------------------------------------------------
    @staticmethod
    def _summary(state: dict[str, JsonValue]) -> CompactionOutput | None:
        raw = state.get("summary")
        return CompactionOutput.model_validate(raw) if isinstance(raw, dict) else None

    async def _summarize(
        self,
        messages: list[CoreMessage],
        previous: CompactionOutput | None,
        client: CoreChatCompletionClient,
    ) -> CompactionOutput:
        """Fold ``messages`` into ``previous``, chunking when they don't fit
        one request: chunk N's summary is chunk N+1's previous summary."""
        rows = [self._row(message) for message in messages]
        chunks: list[list[str]] = [[]]
        used, limit = 0, self._input_budget(client)
        for row in rows:
            size = self.token_counter.count_text(row)
            if chunks[-1] and used + size > limit:
                chunks.append([])
                used = 0
            chunks[-1].append(row)
            used += size

        summary = previous
        for chunk in chunks:
            summary = await self._summarize_chunk("\n".join(chunk), summary, client)
        return summary or CompactionOutput()

    def _input_budget(self, client: CoreChatCompletionClient) -> int:
        """Transcript tokens per summary request: half the summarizer's window
        minus the summary output and the task scaffold (8k if unknown)."""
        window = getattr(getattr(client, "config", None), "max_context_window", 0) or 0
        if window <= 0:
            return 8_000
        return max(1_000, window // 2 - self.config.summary_max_tokens - 1_000)

    async def _summarize_chunk(
        self,
        transcript: str,
        previous: CompactionOutput | None,
        client: CoreChatCompletionClient,
    ) -> CompactionOutput:
        task = SUMMARY_TASK.format(
            previous=previous.model_dump_json(exclude_defaults=True) if previous else "None yet.",
            transcript=transcript,
        )
        result = await client.run(
            ctx=RunContext(messages=[UserMessage(source="compaction", content=task)]),
            prompts=PromptCtx.model_construct(
                stack=None, variables={}, rendered_layers={}, layer_usage={}, prompt_tokens=0,
            ),
            tools=None,
            output_format=CompactionOutput,
            stream=False,
            max_tokens=self.config.summary_max_tokens,
        )
        structured = result.message.structured_output
        if isinstance(structured, CompactionOutput):
            return structured
        if structured is not None:
            return CompactionOutput.model_validate(structured.model_dump())
        # Models without structured output still return usable prose.
        return CompactionOutput(summary=result.message.text().strip())

    def _row(self, message: CoreMessage) -> str:
        text = message.text()
        calls = getattr(message, "tool_calls", None)
        if calls:
            text += " " + "; ".join(f"{c.tool_name}({c.parameters})" for c in calls)
        tokens = self.token_counter.encode(text)
        cap = self.config.message_cap_tokens
        if len(tokens) > cap:
            text = self.token_counter.decode(tokens[:cap]) + " …[truncated]"
        return f"[{message.role}/{message.source}] {text.strip()}"
