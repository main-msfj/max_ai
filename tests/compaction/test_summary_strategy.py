"""SummaryCompaction: [summary that always lives] + recent messages."""

from __future__ import annotations

from types import SimpleNamespace

from max_ai.capabilities.compaction import SummaryCompaction
from max_ai.core.compaction import CompactionOutput
from max_ai.core.messages import AssistantMessage, UserMessage
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx

WINDOW = 10_000  # capacity 7_500 → threshold 6_000, keep budget 3_000


class Summarizer:
    """Fake LLM: records each request, answers with the next output."""

    def __init__(self, *outputs, window: int = 0, structured: bool = True):
        self.outputs = list(outputs)
        self.calls: list[dict] = []
        self.generation_options = {"max_tokens": 1_000}
        self.config = SimpleNamespace(max_context_window=window)
        self.structured = structured

    async def run(self, *, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        self.calls.append({"task": ctx.messages[0].text(), "output_format": output_format, **kwargs})
        output = self.outputs[min(len(self.calls), len(self.outputs)) - 1]
        return ChatCompletionResult(
            message=AssistantMessage(
                source="summarizer", content=output.summary,
                structured_output=output if self.structured else None,
            ),
            usage=Usage(llm_calls=1, attempts_to_call_api=1),
            model="fake", finish_reason="stop",
        )


def prompts() -> PromptCtx:
    return PromptCtx.model_construct(
        stack=None, variables={}, rendered_layers={}, layer_usage={}, prompt_tokens=1_000,
    )


def chat(turns: int, start: int = 1) -> list:
    return [m for n in range(start, start + turns) for m in (
        UserMessage(source="marvin", content=f"pregunta {n}", token_count=500),
        AssistantMessage(source="a", content=f"respuesta {n}", token_count=500),
    )]


async def compact(strategy, ctx, client):
    return await strategy.compact(ctx=ctx, prompts=prompts(), max_context_tokens=WINDOW, client=client)


FIRST = CompactionOutput(summary="Armando un scraper", objective=["scrapear la página X"])
SECOND = CompactionOutput(summary="Scraper listo", successfully_done=["creado scrape.py"])


async def test_first_compaction_creates_the_summary_and_keeps_recent_raw():
    client = Summarizer(FIRST)
    ctx = RunContext(messages=chat(8))  # 8_000 > 6_000
    before = list(ctx.messages)
    result = await compact(SummaryCompaction(summary_max_tokens=500), ctx, client)

    assert result.changed and not result.pruned_only
    assert result.state["summary"]["summary"] == "Armando un scraper"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["output_format"] is CompactionOutput and call["max_tokens"] == 500
    assert "pregunta 1" in call["task"] and "None yet." in call["task"]
    # budget 3_000 - 500 reserved for the summary → 2 turns stay raw
    assert [m.text() for m in result.messages] == ["pregunta 7", "respuesta 7", "pregunta 8", "respuesta 8"]
    assert ctx.messages == before and ctx.compaction.state == {}


async def test_next_compaction_merges_into_the_previous_summary():
    client = Summarizer(SECOND)
    ctx = RunContext(messages=chat(8, start=9))
    ctx.compaction.state["summary"] = FIRST.model_dump(exclude_defaults=True)
    result = await compact(SummaryCompaction(summary_max_tokens=500), ctx, client)

    assert "Armando un scraper" in client.calls[0]["task"]  # previous summary sent
    assert "pregunta 9" in client.calls[0]["task"]
    assert result.state["summary"]["summary"] == "Scraper listo"


async def test_render_puts_the_summary_in_prompt_form():
    strategy = SummaryCompaction()
    assert strategy.render({}) is None
    block = strategy.render({"summary": FIRST.model_dump(exclude_defaults=True)})
    assert block.startswith("<conversation_summary>") and block.endswith("</conversation_summary>")
    assert "Armando un scraper" in block
    assert "Objectives:\n- scrapear la página X" in block
    assert "Pending:" not in block  # empty sections are left out


async def test_a_big_transcript_is_summarized_in_chunks():
    # Summarizer window 4_000 → 1_000 tokens of transcript per request.
    client = Summarizer(FIRST, SECOND, window=4_000)
    long_text = "palabra " * 700  # ~700 tokens per message
    messages = [m for n in range(1, 9) for m in (
        UserMessage(source="marvin", content=f"{n} {long_text}", token_count=500),
        AssistantMessage(source="a", content="ok", token_count=500),
    )]
    result = await compact(SummaryCompaction(summary_max_tokens=500), RunContext(messages=messages), client)

    assert len(client.calls) > 1
    assert "Armando un scraper" in client.calls[1]["task"]  # chunk 1's summary feeds chunk 2
    assert result.state["summary"]["summary"] == "Scraper listo"


async def test_a_giant_message_is_cut_before_summarizing():
    client = Summarizer(FIRST)
    giant = UserMessage(source="marvin", content="dato " * 5_000, token_count=500)
    ctx = RunContext(messages=[giant, *chat(8)])
    await compact(SummaryCompaction(summary_max_tokens=500, message_cap_tokens=100), ctx, client)
    assert "…[truncated]" in client.calls[0]["task"]
    assert client.calls[0]["task"].count("dato") < 200


async def test_models_without_structured_output_still_work():
    client = Summarizer(FIRST, structured=False)
    result = await compact(SummaryCompaction(summary_max_tokens=500), RunContext(messages=chat(8)), client)
    assert result.state["summary"] == {"summary": "Armando un scraper"}


async def test_under_the_threshold_nothing_is_summarized():
    client = Summarizer(FIRST)
    result = await compact(SummaryCompaction(), RunContext(messages=chat(4)), client)
    assert not result.changed and client.calls == []


async def test_a_long_current_turn_is_cut_between_tool_blocks():
    from max_ai.core.messages import ToolCall, ToolMessage

    client = Summarizer(FIRST)
    msgs = [UserMessage(source="marvin", content="migra el proyecto", token_count=500)]
    for n in range(8):  # one turn, 8 tool blocks of ~1_000 tokens
        msgs += [
            AssistantMessage(source="a", content="", tool_calls=[ToolCall(id=f"c{n}", tool_name="bash")],
                             token_count=500),
            ToolMessage(source="t", content="ok", tool_call_id=f"c{n}", tool_name="bash",
                        success=True, token_count=500),
        ]
    result = await compact(SummaryCompaction(summary_max_tokens=500), RunContext(messages=msgs), client)
    assert result.changed and "migra el proyecto" in client.calls[0]["task"]  # question → summary
    assert result.messages[0].tool_calls  # window opens at a whole tool block


async def test_json_cut_by_the_token_limit_keeps_its_complete_fields():
    class Cut(Summarizer):
        async def run(self, **kwargs):
            result = await super().run(**kwargs)
            cut = '{"summary":"Armando un scraper","decisions":["perro Toby","vive en Lima","formato de 200 pal'
            return result.model_copy(update={"message": AssistantMessage(source="s", content=cut)})

    client = Cut(FIRST, structured=False)
    result = await compact(SummaryCompaction(summary_max_tokens=500), RunContext(messages=chat(8)), client)
    assert result.state["summary"] == {
        "summary": "Armando un scraper", "decisions": ["perro Toby", "vive en Lima"],
    }
    assert "Keep the whole summary under 500 tokens" in client.calls[0]["task"]
