"""The new compaction: cut long results, drop old exchanges, keep a summary, and the loop's part."""

from max_ai.base.component import ComponentBase
from max_ai.base.loop import LoopHook, LoopState
from max_ai.capabilities.compaction import SlidingWindowCompaction, SummaryCompaction
from max_ai.capabilities.loop import ReactLoop
from max_ai.capabilities.stacks.builder import PromptBuilder
from max_ai.capabilities.stacks.custom_layer import CustomLayer
from max_ai.capabilities.tools.decorator import tool as as_tool
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.core.retry import RetryPolicy
from max_ai.core.tool import ToolDispatcher, ToolRegistry
from max_ai.errors.client import ClientError
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.summary import Summary
from max_ai.types.todos import Todo


class Model:
    """A client with a small window that answers with ``replies`` and keeps what it was asked."""

    model = "test-model"
    retry = RetryPolicy(attempts=1)

    def __init__(self, *replies, window: int = 3000) -> None:
        self.config = ModelConfig(max_context_window=window, max_output_tokens=500)
        self.replies, self.asked = list(replies), []

    def format_messages(self, ctx, prompts):
        self.prompts = prompts
        return list(ctx.messages)

    async def run(self, messages, tools=None, output_format=None, **options):
        self.asked.append((messages, output_format))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, Summary):
            reply = AssistantMessage(source="llm", content=reply.model_dump_json(), structured_output=reply)
        elif isinstance(reply, str):
            reply = AssistantMessage(source="llm", content=reply)
        return ChatCompletionResult(message=reply, usage=Usage(llm_calls=1), model="m", finish_reason="stop")


def exchange(n: int, size: int = 400) -> list:
    """A user message, a tool call and its long answer."""
    call = ToolCall(id=f"c{n}", tool_name="read", parameters={"n": n})
    return [
        UserMessage(source="u", content=f"question {n}"),
        AssistantMessage(source="llm", content="", tool_calls=[call]),
        ToolMessage(source="read", tool_call_id=f"c{n}", tool_name="read", success=True, content=f"line {n} " * size),
        AssistantMessage(source="llm", content=f"answer {n}"),
    ]


def conversation(exchanges: int, size: int = 400) -> list:
    return [message for n in range(exchanges) for message in exchange(n, size)]


async def test_a_conversation_that_fits_is_left_alone():
    messages = conversation(1, size=10)
    result = await SlidingWindowCompaction().compact(messages, Model())
    assert not result.changed and result.messages == messages


async def test_cutting_the_old_long_results_can_be_enough():
    messages = conversation(3, size=250)
    window = SlidingWindowCompaction(max_result_chars=200)
    assert window.needed(messages, Model())
    result = await window.compact(messages, Model())
    assert result.changed and result.removed == 0 and len(result.messages) == len(messages)
    assert "[result cut to make room:" in result.messages[2].content
    assert "result cut" not in result.messages[-2].content  # the last exchanges are not touched
    assert result.tokens_after < result.tokens_before


async def test_old_exchanges_leave_whole_and_no_message_is_written():
    messages = conversation(8)
    result = await SlidingWindowCompaction().compact(messages, Model())
    rest = result.messages
    assert result.removed > 0 and result.summary is None
    assert all(any(m is original for original in messages) for m in rest)  # only messages that were there
    # No tool answer is left without its call, and the newest exchange is whole.
    calls = {c.id for m in rest if m.role == "assistant" for c in m.tool_calls}
    assert all(m.tool_call_id in calls for m in rest if m.role == "tool")
    assert rest[-1].content == "answer 7"
    assert result.context.messages == result.tokens_after < result.context.capacity


async def test_a_summary_is_kept_and_the_next_one_is_joined_with_it():
    first = Summary(goal="build fizz", done=["wrote fizz.py"], last_request="run the tests")
    model = Model(first)
    compaction = SummaryCompaction(instructions="Keep the file paths.")
    result = await compaction.compact(conversation(8), model)
    asked, wanted = model.asked[0]
    assert wanted is Summary and "Keep the file paths." in asked[0].content and "question 0" in asked[0].content
    assert result.summary == first and result.usage.llm_calls == 1
    assert "Goal:\nbuild fizz" in first.text() and "- wrote fizz.py" in first.text()
    # A second compaction shows the model the summary kept.
    second = Summary(goal="build fizz", done=["wrote fizz.py", "tests pass"])
    model = Model(second)
    again = await compaction.compact([*result.messages, *conversation(8)], model, summary=result.summary)
    assert "This is the summary kept so far" in model.asked[0][0][0].content and again.summary == second


async def test_a_summary_that_fails_drops_the_messages_and_keeps_what_was_there():
    kept = Summary(goal="build fizz")
    for reply in (ClientError.authentication_failed("x"), "this is not json"):
        result = await SummaryCompaction().compact(conversation(8), Model(reply), summary=kept)
        assert result.removed > 0 and result.summary == kept


async def test_another_model_can_write_the_summary():
    cheap, agent = Model(Summary(goal="g")), Model()
    await SummaryCompaction(client=cheap).compact(conversation(8), agent)
    assert len(cheap.asked) == 1 and agent.asked == []


async def test_the_loop_compacts_before_the_model_and_leaves_ctx_in_order():
    seen = []

    class Watches(LoopHook):
        async def after_compact(self, ctx, state):
            seen.append(len(ctx.messages))

    ctx = RunContext(user_id="u", session_id="s", messages=conversation(8), cwd="/w/repo")
    ctx.todos = [Todo(content="run tests", status="in_progress"), Todo(content="write code", status="done")]
    ctx.files_seen["/w/a.py"] = "hash"
    model = Model(Summary(goal="build fizz"), "all done", "really done")
    loop, state = ReactLoop(compaction=SummaryCompaction()), LoopState()
    events = [e async for e in loop.run(ctx, "p", model, ToolDispatcher(ToolRegistry([])), hook=Watches(), state=state)]
    told = [e for e in events if e.event_type == "compaction"]
    assert len(told) == 1 and told[0].tokens_after < told[0].tokens_before and "build fizz" in told[0].summary
    system = loop.system(ctx, "p")
    assert system.startswith("p\n\nEarlier messages") and "Goal:\nbuild fizz" in system
    assert "Current folder: /w/repo" in system and "- [in_progress] run tests" in system
    assert not any("build fizz" in str(m.content) for m in ctx.messages)  # the summary is not a message
    assert ctx.summary == Summary(goal="build fizz") and ctx.files_seen == {} and seen and state.compactions == 1
    assert state.usage.llm_calls == 3  # the summary call is counted with the two of the turn
    assert ctx.messages[-1].content == "really done"


async def test_a_model_that_refuses_the_conversation_as_too_long_gets_it_compacted_by_force():
    ctx = RunContext(user_id="u", session_id="s", messages=conversation(4, size=20))
    model = Model(ClientError.token_limit_exceeded(), "ok", window=128_000)
    state = LoopState()
    events = [e async for e in ReactLoop().run(ctx, "p", model, ToolDispatcher(ToolRegistry([])), state=state)]
    assert [e.event_type for e in events].count("compaction") == 1 and state.finish_reason == "stop"
    assert ctx.position is not None and ctx.messages[-1].content == "ok"


async def test_the_summary_is_told_its_room_a_part_of_the_window_of_the_agent():
    cheap, agent = Model(Summary(goal="g"), window=100_000), Model(window=3000)
    compaction = SummaryCompaction(client=cheap, max_share=0.2)
    await compaction.compact(conversation(8), agent)
    assert "must stay under 600 tokens" in cheap.asked[0][0][0].content
    again = ComponentBase.deserialize(SummaryCompaction(max_share=0.2).serialize().model_dump())
    assert again.max_share == 0.2


def test_a_loop_is_stored_with_its_compaction_and_built_again():
    loop = ReactLoop(max_iterations=7, compaction=SummaryCompaction(instructions="paths", keep=0.5))
    again = ComponentBase.deserialize(loop.serialize().model_dump())
    assert isinstance(again.compaction, SummaryCompaction)
    assert (again.compaction.instructions, again.compaction.keep, again.max_iterations) == ("paths", 0.5, 7)
    assert isinstance(ComponentBase.deserialize(ReactLoop().serialize().model_dump()).compaction, SlidingWindowCompaction)


async def test_the_window_is_shared_out_part_by_part_and_kept_in_ctx():
    context = SlidingWindowCompaction().measure(conversation(1, size=10), Model(window=10_000), 1000, 500)
    assert (context.window, context.prompt, context.tools, context.output) == (10_000, 1000, 500, 500)
    assert context.margin == 1600 and context.capacity == 6400  # 80% of the 8000 left
    total = context.prompt + context.tools + context.output + context.margin + context.capacity
    assert total == context.window and context.free == context.capacity - context.messages
    assert ModelConfig(max_context_window=200_000).max_output_tokens == 20_000  # never zero
    # The loop leaves it in ctx on every call, compaction or not.
    ctx = RunContext(user_id="u", session_id="s", messages=conversation(1, size=10))
    async for _ in ReactLoop().run(ctx, "You are brief.", Model("ok", window=10_000), ToolDispatcher(ToolRegistry([]))):
        pass
    assert ctx.context.window == 10_000 and ctx.context.prompt > 0 and ctx.context.messages > 0


def test_each_part_counts_itself_and_the_window_is_known_before_the_first_message():
    def weather(city: str) -> str:
        """The weather in a city."""
        return city

    def news(topic: str) -> str:
        """The news about a topic."""
        return topic

    # The registry counts its loaded tools and counts again when they change.
    registry = ToolRegistry([as_tool(weather)])
    counter = SlidingWindowCompaction().counter
    one = registry.tokens(counter)
    assert one > 0 and registry.tokens() > 0 and ToolRegistry([]).tokens() == 0
    registry.register(as_tool(news))
    assert registry.tokens(counter) > one
    registry.unregister("news")
    assert registry.tokens(counter) == one
    # The prompt builder counts what it builds; without a counter, 4 characters a token.
    builder = PromptBuilder([CustomLayer("Answer in one line. " * 20)], registry)
    assert builder.tokens() == len(builder.build()) // 4
    assert builder.tokens(counter=counter) == counter.count_text(builder.build())
    # A session that just opened: prompt and tools are known, the conversation is empty.
    ctx = RunContext(user_id="u", session_id="s")
    context = ReactLoop().measure(ctx, builder.build(), Model(window=10_000), ToolDispatcher(registry))
    assert ctx.context is context and context.messages == 0
    assert (context.prompt, context.tools) == (builder.tokens(counter=counter), one)


# -------- the messages that leave: someone may look at them while they are summarized ----
async def test_who_asked_gets_the_messages_that_leave():
    got = []

    async def look(old):
        got.extend(old)

    messages = conversation(8)
    result = await SummaryCompaction().compact(messages, Model(Summary(goal="g")), leaving=look)
    # The oldest ones, in order (their long results already cut).
    assert result.removed == len(got) > 0
    assert [m.role for m in got] == [m.role for m in messages[: len(got)]]
    assert got[0].content == "question 0"
    # A conversation that fits has nothing leaving: nobody is called.
    got.clear()
    await SummaryCompaction().compact(conversation(1, size=5), Model(), leaving=look)
    assert got == []


async def test_a_failure_looking_at_them_does_not_stop_the_summary_nor_the_other_way():
    async def breaks(old):
        raise RuntimeError("the memory is down")

    kept = await SummaryCompaction().compact(
        conversation(8), Model(Summary(goal="build fizz")), leaving=breaks
    )
    assert kept.summary == Summary(goal="build fizz") and kept.removed > 0

    looked = []

    async def look(old):
        looked.append(len(old))

    failed = await SummaryCompaction().compact(
        conversation(8), Model(ClientError.authentication_failed("x")), leaving=look
    )
    assert failed.removed > 0 and looked == [failed.removed]


async def test_at_compaction_what_the_user_said_goes_to_the_memory(tmp_path, monkeypatch):
    from pathlib import Path

    from max_ai.capabilities.memory import Memory
    from max_ai.capabilities.sandbox.local import LocalSandbox
    from max_ai.config import setting
    from max_ai.core.harness.hooks import HarnessLoopHook

    sandbox = LocalSandbox(root=str(tmp_path)).bind("u")
    await sandbox.prepare()
    fact = (
        '{"facts": [{"name": "uses-uv", "type": "user", '
        '"description": "Installs packages with uv, never pip", "content": "Uses uv."}]}'
    )

    async def run(keeper, memory=None):
        ctx = RunContext(user_id="u", session_id="s", messages=conversation(8))
        hook = HarnessLoopHook(memory=memory or Memory(), sandbox=sandbox, keeper=keeper)
        model = Model(Summary(goal="build fizz"), "all done")
        loop = ReactLoop(compaction=SummaryCompaction())
        async for _ in loop.run(ctx, "p", model, ToolDispatcher(ToolRegistry([])), hook=hook, state=LoopState()):
            pass
        return ctx

    # 1. The cheap model is asked once, with the user's messages only.
    keeper = Model(fact)
    ctx = await run(keeper)
    asked = keeper.asked[0][0][0].content
    assert "question 0" in asked and "line 0" not in asked and "answer 0" not in asked
    saved = Path(str(sandbox.layout.memory), "uses-uv.md")
    assert "Installs packages with uv, never pip" in saved.read_text()
    # 2. The model sees the new fact at once: the index is read again after compacting.
    assert "uses-uv.md" in (ctx.memory_index or "") and ctx.summary == Summary(goal="build fizz")

    # 3. A keeper that fails changes nothing for the turn.
    broken = await run(Model(RuntimeError("down")))
    assert broken.summary == Summary(goal="build fizz") and broken.messages[-1].content == "all done"

    # 4. Read-only, or turned off: the model is not even asked.
    quiet = Model(fact)
    await run(quiet, Memory(read_only=True))
    monkeypatch.setattr(setting, "harness_memory_at_compaction", False)
    await run(quiet)
    assert quiet.asked == []
