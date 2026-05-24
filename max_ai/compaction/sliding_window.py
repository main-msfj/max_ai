from __future__ import annotations

import typing as t

from pydantic import Field

from ..base.compaction import (
    CoreCompaction,
    TokenCounter,
    group_atomic_messages,
    split_recent_messages,
)
from ..config import setting
from ..core.compaction import CompactionOutput, CompactionResult
from ..core.messages import UserMessage
from ..types.run_context import RunContext
from ..types.stacks import PromptCtx

if t.TYPE_CHECKING:
    from ..base.clients import CoreChatCompletionClient


class SlidingWindowCompaction(CoreCompaction):
    """Keep recent live messages and summarize older context when needed.

    Older messages are merged with the previous compaction summary using the
    configured chat completion client. Newer raw messages win over stale
    summary details when there is a conflict.
    """

    token_counter: TokenCounter = Field(default_factory=TokenCounter)

    async def compact(
        self,
        *,
        ctx: "RunContext",
        prompts: "PromptCtx",
        max_context_tokens: int,
        client: "CoreChatCompletionClient",
    ) -> CompactionResult:
        previous_summary = self._load_previous_summary(ctx)

        live_message_threshold_tokens = int(
            max_context_tokens * setting.compaction_live_message_threshold
        )
        live_message_budget_tokens = setting.compaction_live_message_budget_tokens
        total_tokens = self.token_counter.count_messages(ctx.messages)

        if total_tokens <= live_message_threshold_tokens:
            if previous_summary is not None:
                self._inject_summary(prompts, previous_summary)
            return CompactionResult(
                changed=False,
                recent_messages=list(ctx.messages),
                recent_token_count=total_tokens,
                total_token_count=total_tokens,
            )

        groups = group_atomic_messages(ctx.messages, self.token_counter)
        old_messages, recent_messages = split_recent_messages(
            groups,
            max_tokens=live_message_budget_tokens,
        )

        summary = None
        if old_messages:
            summary_output = await self._summarize_old_messages(
                client=client,
                previous_summary=previous_summary,
                old_messages=old_messages,
            )
            summary = summary_output.model_dump_json(exclude_none=True)
            ctx.runtime_state.shared_state["compaction_summary"] = summary_output.model_dump(
                exclude_none=True
            )
            self._inject_summary(prompts, summary_output)
            ctx.messages = recent_messages

        return CompactionResult(
            changed=bool(old_messages),
            old_messages=old_messages,
            recent_messages=recent_messages,
            old_token_count=self.token_counter.count_messages(old_messages),
            recent_token_count=self.token_counter.count_messages(recent_messages),
            total_token_count=total_tokens,
            summary=summary,
        )

    def _load_previous_summary(self, ctx: RunContext) -> CompactionOutput | None:
        raw = ctx.runtime_state.shared_state.get("compaction_summary")
        if not raw:
            return None
        if isinstance(raw, CompactionOutput):
            return raw
        if isinstance(raw, dict):
            return CompactionOutput.model_validate(raw)
        if isinstance(raw, str):
            try:
                return CompactionOutput.model_validate_json(raw)
            except ValueError:
                return CompactionOutput(summary=raw)
        return None

    def _inject_summary(self, prompts: PromptCtx, summary: CompactionOutput) -> None:
        block = self._summary_prompt_block(summary)
        if not block:
            return

        try:
            from ..stacks.context_layer import ContextLayer

            existing = prompts.rendered_layers.get(ContextLayer, "")
            prompts.rendered_layers[ContextLayer] = (
                f"{existing.rstrip()}\n\n{block}" if existing else block
            )
        except Exception:
            # Prompt injection should never break the run. If the concrete
            # context layer is unavailable, the summary still remains in
            # runtime_state for the next compaction pass.
            return

    def _summary_prompt_block(self, summary: CompactionOutput) -> str:
        payload = summary.model_dump_json(exclude_none=True)
        if not payload or payload == "{}":
            return ""
        return (
            "<COMPACTION_SUMMARY>\n"
            "This is compressed prior conversation context. Use it for continuity, "
            "but the current live messages always win if they conflict.\n"
            f"{payload}\n"
            "</COMPACTION_SUMMARY>"
        )

    async def _summarize_old_messages(
        self,
        *,
        client: "CoreChatCompletionClient",
        previous_summary: CompactionOutput | None,
        old_messages: list[t.Any],
    ) -> CompactionOutput:
        task = self._summary_task(previous_summary, old_messages)
        summary_ctx = RunContext(
            messages=[UserMessage(source="compaction", content=task)]
        )
        summary_prompts = PromptCtx.model_construct(
            stack=None,
            variables={},
            rendered_layers={},
            layer_usage={},
            prompt_tokens=0,
        )
        result = await client.run(
            ctx=summary_ctx,
            prompts=summary_prompts,
            tools=None,
            output_format=CompactionOutput,
            stream=False,
            max_tokens=setting.compaction_summary_budget_tokens,
        )
        structured = result.message.structured_output
        if isinstance(structured, CompactionOutput):
            return structured
        if structured is not None:
            return CompactionOutput.model_validate(structured.model_dump())
        if result.message.text().strip():
            return CompactionOutput(summary=result.message.text().strip())
        return CompactionOutput()

    def _summary_task(
        self,
        previous_summary: CompactionOutput | None,
        old_messages: list[t.Any],
    ) -> str:
        previous = (
            previous_summary.model_dump_json(exclude_none=True)
            if previous_summary is not None
            else "No previous summary."
        )
        transcript = self._messages_transcript(old_messages)
        return (
            "Create an updated compact conversation summary.\n"
            "Combine the previous summary with the messages below. The messages "
            "below are newer, so they win over the previous summary if there is "
            "any conflict. Preserve objectives, pending work, completed work, "
            "decisions, and important context. Return only the requested "
            "structured output.\n\n"
            f"Previous summary:\n{previous}\n\n"
            f"Newer messages to merge:\n{transcript}"
        )

    def _messages_transcript(self, messages: list[t.Any]) -> str:
        rows: list[str] = []
        for message in messages:
            role = getattr(message, "role", "message")
            source = getattr(message, "source", "unknown")
            text = message.text() if callable(getattr(message, "text", None)) else str(message)
            if not text and getattr(message, "tool_calls", None):
                text = f"tool_calls={message.tool_calls}"
            rows.append(f"[{role}/{source}] {text}")
        return "\n".join(rows)
