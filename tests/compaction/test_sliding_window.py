from __future__ import annotations

import pytest

from max_ai.compaction import SlidingWindowCompaction
from max_ai.base.compaction import TokenCounter
from max_ai.config import setting
from max_ai.core.compaction import CompactionOutput
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx


def user(content: str, *, tokens: int = 1) -> UserMessage:
    return UserMessage(source="user", content=content, token_count=tokens)


def assistant(content: str, *, tokens: int = 1) -> AssistantMessage:
    return AssistantMessage(source="agent", content=content, token_count=tokens)


class SummaryClient:
    def __init__(self) -> None:
        self.calls = []

    async def run(self, *, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        self.calls.append({
            "ctx": ctx,
            "prompts": prompts,
            "tools": tools,
            "output_format": output_format,
            "stream": stream,
            "kwargs": kwargs,
        })
        output = CompactionOutput(
            summary="Merged summary",
            objective=["finish compaction"],
            pending=["wire prompt injection"],
            successfully_done=["kept recent messages"],
        )
        return ChatCompletionResult(
            message=AssistantMessage(
                source="summary-client",
                content=output.model_dump_json(),
                structured_output=output,
            ),
            usage=Usage(llm_calls=1, attempts_to_call_api=1),
            model="summary-client",
            finish_reason="stop",
        )

def make_compaction() -> SlidingWindowCompaction:
    return SlidingWindowCompaction(token_counter=TokenCounter())


def set_compaction_settings(monkeypatch, *, threshold=0.4, live_budget=5):
    monkeypatch.setattr(setting, "compaction_prompt_budget_tokens", 0)
    monkeypatch.setattr(setting, "compaction_min_output_tokens", 0)
    monkeypatch.setattr(setting, "compaction_safety_margin_ratio", 0)
    monkeypatch.setattr(setting, "compaction_live_message_threshold", threshold)
    monkeypatch.setattr(setting, "compaction_live_message_keep_ratio", live_budget / 100)


def make_prompts() -> PromptCtx:
    return PromptCtx.model_construct(
        stack=None,
        variables={},
        rendered_layers={},
        layer_usage={},
        prompt_tokens=0,
    )


@pytest.mark.asyncio
async def test_sliding_window_noops_below_live_threshold(monkeypatch):
    ctx = RunContext(
        messages=[
            user("hello", tokens=2),
            assistant("hi", tokens=2),
        ]
    )
    set_compaction_settings(monkeypatch, threshold=0.1, live_budget=2)
    compaction = make_compaction()

    result = await compaction.compact(
        ctx=ctx,
        prompts=make_prompts(),
        max_context_tokens=100,
        client=SummaryClient(),
    )

    assert result.changed is False
    assert result.old_messages == []
    assert result.recent_messages == ctx.messages
    assert result.recent_token_count == 4


@pytest.mark.asyncio
async def test_sliding_window_mutates_context_to_recent_messages(monkeypatch):
    ctx = RunContext(
        messages=[
            user("old 1", tokens=4),
            assistant("old 2", tokens=4),
            user("recent 1", tokens=3),
            assistant("recent 2", tokens=2),
        ]
    )
    set_compaction_settings(monkeypatch, threshold=0.01, live_budget=5)
    compaction = make_compaction()
    ctx.runtime_state.shared_state["compaction_summary"] = {
        "summary": "Previous stale summary",
        "pending": ["old pending item"],
    }

    client = SummaryClient()
    prompts = make_prompts()

    result = await compaction.compact(
        ctx=ctx,
        prompts=prompts,
        max_context_tokens=100,
        client=client,
    )

    assert result.changed is True
    assert [message.text() for message in result.old_messages] == ["old 1", "old 2"]
    assert [message.text() for message in ctx.messages] == [
        "recent 1",
        "recent 2",
    ]
    assert client.calls[0]["output_format"] is CompactionOutput
    assert client.calls[0]["kwargs"] == {"max_tokens": setting.compaction_summary_budget_tokens}
    assert result.summary is not None
    assert ctx.runtime_state.shared_state["compaction_summary"]["summary"] == "Merged summary"
    rendered_prompt = "\n".join(prompts.rendered_layers.values())
    assert rendered_prompt.count("<COMPACTION_SUMMARY>") == 1
    assert "Merged summary" in rendered_prompt
    assert "Previous stale summary" not in rendered_prompt
    summary_task = client.calls[0]["ctx"].messages[0].text()
    assert "Previous stale summary" in summary_task
    assert "old 1" in summary_task


@pytest.mark.asyncio
async def test_sliding_window_does_not_split_tool_group(monkeypatch):
    tool_call = ToolCall(id="call_1", tool_name="lookup", parameters={})
    assistant_call = AssistantMessage(
        source="agent",
        content="",
        tool_calls=[tool_call],
        token_count=3,
    )
    tool_result = ToolMessage.success_message(
        tool_call_id="call_1",
        tool_name="lookup",
        content="result",
        source="tool",
    ).model_copy(update={"token_count": 3})
    ctx = RunContext(
        messages=[
            user("old", tokens=2),
            assistant_call,
            tool_result,
        ]
    )
    set_compaction_settings(monkeypatch, threshold=0.01, live_budget=4)
    compaction = make_compaction()

    result = await compaction.compact(
        ctx=ctx,
        prompts=make_prompts(),
        max_context_tokens=100,
        client=SummaryClient(),
    )

    assert [message.text() for message in result.old_messages] == ["old"]
    assert ctx.messages == [assistant_call, tool_result]
    assert result.recent_token_count == 6
