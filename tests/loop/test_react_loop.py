"""ReactLoop: model, tools, repeat; what ends a turn and what is left in the conversation."""

import asyncio

import pytest
from pydantic import BaseModel

from max_ai.base.component import ComponentBase
from max_ai.base.loop import LoopHook, LoopState, Note, Refuse, Stop
from max_ai.capabilities.loop import ReactLoop
from max_ai.capabilities.tools.decorator import tool as as_tool
from max_ai.core.messages import AssistantMessage, ToolCall, UserMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.core.retry import RetryPolicy
from max_ai.core.tool import ToolDispatcher, ToolHook, ToolRegistry
from max_ai.errors.client import ClientError
from max_ai.types.completions import ChatCompletionChunk, ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.todos import Todo

WEATHER: list[str] = []


async def weather(city: str) -> str:
    """Current forecast for a city."""
    WEATHER.append(city)
    return f"sunny in {city}"


async def slow(seconds: float) -> str:
    """Wait."""
    await asyncio.sleep(seconds)
    return "done"


class Model:
    """A client that answers with ``replies`` in order and keeps what it was sent."""

    model = "test-model"
    retry = RetryPolicy(attempts=3)
    config = ModelConfig()

    def __init__(self, *replies: AssistantMessage) -> None:
        self.replies, self.sent = list(replies), []
        self.reasons: dict[int, str] = {}  # the finish reason of a call, by its number

    def format_messages(self, ctx, prompts):
        return list(ctx.messages)

    def _next(self, messages, tools):
        self.sent.append(([m.role for m in messages], [t.name for t in tools or []]))
        if not self.replies:
            return says("")
        reply = self.replies[min(len(self.sent) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def run(self, messages, tools=None, output_format=None, *, retry=None, on_retry=None, **options):
        message = self._next(messages, tools)
        if on_retry and len(self.sent) == 1:
            on_retry(1, TimeoutError("slow"), 0.5)
        reason = self.reasons.get(len(self.sent)) or ("tool_calls" if message.tool_calls else "stop")
        usage = Usage(llm_calls=1, tokens_input=10, tokens_output=2)
        return ChatCompletionResult(message=message, usage=usage, model=self.model, finish_reason=reason)

    async def stream(self, messages, tools=None, output_format=None, *, retry=None, on_retry=None, **options):
        message = self._next(messages, tools)
        for word in (message.content or "").split():
            yield ChatCompletionChunk(content=word, is_complete=False)
        yield ChatCompletionChunk(
            content="", is_complete=True, message=message, finish_reason="stop",
            usage=Usage(llm_calls=1, tokens_output=2),
        )


def says(text: str) -> AssistantMessage:
    return AssistantMessage(source="llm", content=text)


def asks(*calls: tuple[str, str, dict]) -> AssistantMessage:
    wanted = [ToolCall(id=i, tool_name=name, parameters=p) for i, name, p in calls]
    return AssistantMessage(source="llm", content="", tool_calls=wanted)


def setup(hook: ToolHook | None = None, approval: str = "auto_approval"):
    WEATHER.clear()
    ctx = RunContext(user_id="u", session_id="s1", messages=[UserMessage(source="u", content="hi")])
    return ctx, ToolDispatcher(ToolRegistry([as_tool(weather, approval_mode=approval), as_tool(slow)]), hook=hook)


async def run(loop, ctx, model, tools, **options):
    return [event async for event in loop.run(ctx, "You are brief.", model, tools, **options)]


async def test_the_model_answers_and_the_turn_ends():
    ctx, tools = setup()
    state = LoopState()
    events = await run(ReactLoop(), ctx, Model(says("hello")), tools, state=state)
    assert [m.role for m in ctx.messages] == ["user", "assistant"]
    assert (state.iteration, state.finish_reason, state.message.content) == (1, "stop", "hello")
    assert [e.event_type for e in events] == [
        "reasoning_iteration", "model_call", "model_retry", "model_response", "reasoning_complete",
    ]


async def test_tools_run_and_their_answers_go_back_to_the_model():
    ctx, tools = setup()
    model = Model(asks(("c1", "weather", {"city": "Lima"}), ("c2", "weather", {"city": "Cusco"})), says("sunny"))
    state = LoopState()
    await run(ReactLoop(), ctx, model, tools, state=state)
    assert WEATHER == ["Lima", "Cusco"]
    assert [m.role for m in ctx.messages] == ["user", "assistant", "tool", "tool", "assistant"]
    assert model.sent[1] == (["user", "assistant", "tool", "tool"], ["weather", "slow"])
    assert (state.iteration, state.usage.llm_calls, state.usage.tokens_input) == (2, 2, 20)


async def test_a_stream_gives_the_text_in_pieces_and_the_whole_message():
    ctx, tools = setup()
    events = await run(ReactLoop(), ctx, Model(says("good morning")), tools, stream=True)
    assert [e.chunk for e in events if e.event_type == "model_stream_chunk"] == ["good", "morning"]
    assert ctx.messages[-1].content == "good morning"


async def test_the_turn_stops_at_its_limit_of_model_calls():
    ctx, tools = setup()
    again = [asks((f"c{n}", "weather", {"city": "Lima"})) for n in range(5)]
    state = LoopState()
    await run(ReactLoop(max_iterations=2), ctx, Model(*again), tools, state=state)
    assert (state.iteration, state.finish_reason) == (2, "max_iterations")
    assert ctx.messages[-1].role == "tool"  # every call has its answer
    # Given in the call, the limit wins over the loop's own.
    ctx, tools = setup()
    state = LoopState()
    await run(ReactLoop(max_iterations=2), ctx, Model(*again), tools, state=state, max_iterations=1)
    assert state.iteration == 1


async def test_a_tool_that_waits_for_a_person_ends_the_turn_and_the_next_one_resumes():
    ctx, tools = setup(approval="ask_for_approval")  # nobody approves: the call waits
    model = Model(asks(("c1", "weather", {"city": "Lima"})), says("sunny"))
    state = LoopState()
    await run(ReactLoop(), ctx, model, tools, state=state)
    assert state.finish_reason == "waiting_for_user" and WEATHER == []
    assert [m.role for m in ctx.messages] == ["user", "assistant"]
    # The person approves; the next run starts by running the call.
    ctx.tool_state.get("c1").approve()
    state = LoopState()
    await run(ReactLoop(), ctx, model, tools, state=state)
    assert WEATHER == ["Lima"] and state.finish_reason == "stop"
    assert [m.role for m in ctx.messages] == ["user", "assistant", "tool", "assistant"]


async def test_the_harness_is_called_before_each_model_call_and_after_each_round_of_tools():
    seen = []

    class Watches(LoopHook):
        async def before_model(self, ctx, state):
            seen.append(("model", len(ctx.messages)))

        async def after_tools(self, ctx, state):
            seen.append(("tools", len(ctx.messages)))

    ctx, tools = setup()
    model = Model(asks(("c1", "weather", {"city": "Lima"})), says("sunny"))
    await run(ReactLoop(), ctx, model, tools, hook=Watches())
    assert seen == [("model", 1), ("tools", 3), ("model", 3)]


async def test_a_repeated_tool_call_id_gets_a_new_one():
    ctx, tools = setup()
    model = Model(asks(("same", "weather", {"city": "Lima"})), asks(("same", "weather", {"city": "Cusco"})), says("ok"))
    await run(ReactLoop(), ctx, model, tools)
    ids = [m.tool_call_id for m in ctx.messages if m.role == "tool"]
    assert WEATHER == ["Lima", "Cusco"] and len(set(ids)) == 2 and ids[0] == "same"


async def test_a_cancelled_turn_leaves_every_tool_call_with_its_answer():
    ctx, tools = setup()
    model = Model(asks(("c1", "slow", {"seconds": 30})), says("never"))
    state = LoopState()
    turn = asyncio.create_task(run(ReactLoop(), ctx, model, tools, state=state))
    await asyncio.sleep(0.2)
    turn.cancel()
    await asyncio.gather(turn, return_exceptions=True)
    assert turn.cancelled() and state.finish_reason == "cancelled"
    assert [m.role for m in ctx.messages] == ["user", "assistant", "tool"]
    assert not ctx.messages[-1].success and ReactLoop.open_records(ctx) == []


def test_it_is_stored_with_its_limit_and_built_again():
    again = ComponentBase.deserialize(ReactLoop(max_iterations=7).serialize().model_dump())
    assert isinstance(again, ReactLoop) and again.max_iterations == 7


# -------- THE RULES --------------------------------------------------------------
def notes(ctx) -> list[str]:
    return [m.content for m in ctx.messages if m.role == "user" and str(m.content).startswith("[harness]")]


def kinds(events) -> list[str]:
    return [e.kind for e in events if e.event_type == "loop_notice"]


async def test_an_empty_answer_is_asked_again_once_and_then_the_turn_ends():
    ctx, tools = setup()
    state = LoopState()
    events = await run(ReactLoop(), ctx, Model(says(""), says("here")), tools, state=state)
    assert kinds(events) == ["empty_answer"] and state.finish_reason == "stop"
    assert "arrived empty" in notes(ctx)[0] and ctx.messages[-1].content == "here"
    ctx, tools = setup()
    state = LoopState()
    await run(ReactLoop(), ctx, Model(says(" "), says("")), tools, state=state)
    assert (state.finish_reason, state.iteration) == ("empty_answer", 2)


async def test_an_answer_the_provider_blocked_is_not_asked_again():
    ctx, tools = setup()
    model, state = Model(says("")), LoopState()
    model.reasons[1] = "content_filter"
    events = await run(ReactLoop(), ctx, model, tools, state=state)
    assert (state.finish_reason, state.iteration, kinds(events)) == ("content_filter", 1, ["content_filter"])


async def test_the_hook_can_answer_for_the_agent_when_the_turn_cannot_go_on():
    class Answers(LoopHook):
        async def after_model(self, ctx, state):
            if state.finish_reason == "content_filter":
                return Stop(reason="content_filter", answer="This content has been filtered.")
            return await super().after_model(ctx, state)

        async def on_error(self, ctx, state, error):
            return Stop(reason="error", answer="The model is not available right now.")

    ctx, tools = setup()
    model, state = Model(says("")), LoopState()
    model.reasons[1] = "content_filter"
    await run(ReactLoop(), ctx, model, tools, hook=Answers(), state=state)
    assert ctx.messages[-1].content == "This content has been filtered." and ctx.messages[-1].role == "assistant"
    ctx, tools = setup()
    state = LoopState()
    events = await run(ReactLoop(), ctx, Model(ClientError.authentication_failed("bad key")), tools, hook=Answers(), state=state)
    assert state.finish_reason == "error" and ctx.messages[-1].content == "The model is not available right now."
    assert events[-1].event_type == "reasoning_complete"  # the error did not go up


async def test_a_hook_that_ends_the_turn_leaves_the_tools_it_did_not_run_with_an_answer():
    class Ends(LoopHook):
        async def after_model(self, ctx, state):
            return Stop(reason="enough")

    ctx, tools = setup()
    state = LoopState()
    await run(ReactLoop(), ctx, Model(asks(("c1", "weather", {"city": "Lima"}))), tools, hook=Ends(), state=state)
    assert WEATHER == [] and state.finish_reason == "enough"
    assert ctx.messages[-1].role == "tool" and "the turn ended" in ctx.messages[-1].content


async def test_the_same_call_stops_running_at_the_third_time_and_ends_the_turn_at_the_fifth():
    ctx, tools = setup()
    same = [asks((f"c{n}", "weather", {"city": "Lima"})) for n in range(6)]
    state = LoopState()
    events = await run(ReactLoop(), ctx, Model(*same), tools, state=state)
    assert WEATHER == ["Lima", "Lima"]  # the third was answered without running
    assert (state.finish_reason, state.iteration) == ("repetition", 5)
    assert kinds(events) == ["repetition"] * 3  # two refusals, then the end
    answers = [m for m in ctx.messages if m.role == "tool"]
    assert len(answers) == 5 and "Not run: this is the same call" in answers[2].content
    # Another call in between starts the count again.
    ctx, tools = setup()
    mixed = [asks((f"c{n}", "weather", {"city": city})) for n, city in enumerate(["Lima", "Lima", "Cusco", "Lima", "Lima"])]
    await run(ReactLoop(), ctx, Model(*mixed, says("ok")), tools)
    assert WEATHER == ["Lima", "Lima", "Cusco", "Lima", "Lima"]


async def test_near_the_limit_the_model_is_told_how_many_calls_it_has_left():
    ctx, tools = setup()
    again = [asks((f"c{n}", "weather", {"city": str(n)})) for n in range(5)]
    events = await run(ReactLoop(max_iterations=5), ctx, Model(*again), tools, hook=LoopHook(warn_before=2))
    assert kinds(events) == ["near_limit", "max_iterations"]
    assert "You have 2 model calls left" in notes(ctx)[0]
    assert [m.role for m in ctx.messages].index("user", 1) == 7  # after the third round


async def test_open_items_of_the_list_are_reminded_once():
    ctx, tools = setup()
    ctx.todos = [Todo(content="write tests", status="in_progress"), Todo(content="done", status="done")]
    state = LoopState()
    events = await run(ReactLoop(), ctx, Model(says("finished"), says("really finished")), tools, state=state)
    assert kinds(events) == ["open_todos"] and "1 items of your list are still open" in notes(ctx)[0]
    assert (state.iteration, state.finish_reason) == (2, "stop")


async def test_the_harness_can_keep_the_turn_going_or_end_it():
    class Steers(LoopHook):
        def __init__(self):
            super().__init__()
            self.said = False

        async def after_model(self, ctx, state):
            if not self.said:
                self.said = True
                return Note("The user added: also in Cusco.")
            return await super().after_model(ctx, state)

    ctx, tools = setup()
    events = await run(ReactLoop(), ctx, Model(says("sunny"), says("both sunny")), tools, hook=Steers())
    assert kinds(events) == ["hook"] and notes(ctx) == ["[harness] The user added: also in Cusco."]

    class Budget(LoopHook):
        async def before_model(self, ctx, state):
            return Stop(reason="budget") if state.iteration >= 1 else None

    ctx, tools = setup()
    state = LoopState()
    model = Model(asks(("c1", "weather", {"city": "Lima"})), says("never"))
    await run(ReactLoop(), ctx, model, tools, hook=Budget(), state=state)
    assert (state.finish_reason, state.iteration) == ("budget", 1) and ctx.messages[-1].role == "tool"


async def test_a_model_error_is_told_and_goes_up_leaving_the_conversation_as_it_was():
    ctx, tools = setup()
    state, events = LoopState(), []
    with pytest.raises(ClientError):
        async for event in ReactLoop().run(ctx, "p", Model(ClientError.authentication_failed("x")), tools, state=state):
            events.append(event)
    assert state.finish_reason == "error" and events[-1].event_type == "fatal_error"
    assert [m.role for m in ctx.messages] == ["user"]  # nothing was added


async def test_tool_calls_cut_at_the_output_limit_are_answered_without_running():
    ctx, tools = setup()
    model, state = Model(asks(("c1", "weather", {"city": "Li"})), says("ok")), LoopState()
    model.reasons[1] = "length"
    events = await run(ReactLoop(), ctx, model, tools, state=state)
    assert WEATHER == [] and kinds(events) == ["arguments_cut"]
    assert "arrived incomplete" in ctx.messages[2].content and state.finish_reason == "stop"


async def test_refuse_answers_the_tools_without_running_them_and_note_lets_them_run():
    class Says(LoopHook):
        def __init__(self, decision):
            super().__init__()
            self.decision = decision

        async def after_model(self, ctx, state):
            decision, self.decision = self.decision, None
            return decision

    # Refuse: the tool does not run and gets the text; the rule is counted.
    ctx, tools = setup()
    model, state = Model(asks(("c1", "weather", {"city": "Lima"})), says("ok")), LoopState()
    events = await run(ReactLoop(), ctx, model, tools, hook=Says(Refuse("Not now.", rule="mine")), state=state)
    assert WEATHER == [] and ctx.messages[2].content == "Not now."
    assert kinds(events) == ["mine"] and state.notices == {"mine": 1}

    # Note: the tool runs and the note is told after its answer.
    ctx, tools = setup()
    model, state = Model(asks(("c1", "weather", {"city": "Lima"})), says("ok")), LoopState()
    events = await run(ReactLoop(), ctx, model, tools, hook=Says(Note("Be quick.", rule="mine")), state=state)
    assert WEATHER == ["Lima"] and [m.role for m in ctx.messages][2:4] == ["tool", "user"]
    assert notes(ctx) == ["[harness] Be quick."] and state.notices == {"mine": 1}


async def test_a_new_message_closes_the_calls_that_waited_for_the_user():
    ctx, tools = setup(approval="ask_for_approval")
    model = Model(asks(("c1", "weather", {"city": "Lima"})), says("ok, no weather then"))
    await run(ReactLoop(), ctx, model, tools)
    ctx.messages.append(UserMessage(source="u", content="forget it, say hi"))
    state = LoopState()
    await run(ReactLoop(), ctx, model, tools, state=state)
    assert WEATHER == [] and state.finish_reason == "stop"
    # The answer sits right after the call, before the new message.
    assert [m.role for m in ctx.messages] == ["user", "assistant", "tool", "user", "assistant"]
    assert "Do not take this as permission" in ctx.messages[2].content


async def test_a_call_lost_when_a_run_died_gets_an_answer():
    ctx, tools = setup()
    ctx.messages.append(asks(("lost", "weather", {"city": "Lima"})))  # saved, never registered
    await run(ReactLoop(), ctx, Model(says("ok")), tools)
    assert WEATHER == [] and [m.role for m in ctx.messages] == ["user", "assistant", "tool", "assistant"]
    assert "interrupted" in ctx.messages[2].content


async def test_a_hook_that_forgets_the_limit_is_stopped_five_calls_later():
    class Forgets(LoopHook):
        async def before_model(self, ctx, state):
            return None

    ctx, tools = setup()
    again = [asks((f"c{n}", "weather", {"city": str(n)})) for n in range(20)]
    state = LoopState()
    await run(ReactLoop(max_iterations=2), ctx, Model(*again), tools, hook=Forgets(), state=state)
    assert (state.iteration, state.finish_reason) == (7, "max_iterations")
    assert ctx.messages[-1].role == "tool"


# -------- WHAT THE USER WRITES DURING THE TURN -----------------------------------
async def test_what_the_user_writes_during_the_tools_reaches_the_model_after_their_answers():
    class UserWrites(LoopHook):
        async def after_tools(self, ctx, state):
            ctx.inbox.append("also in Cusco")

    ctx, tools = setup()
    model = Model(asks(("c1", "weather", {"city": "Lima"})), says("both sunny"))
    await run(ReactLoop(), ctx, model, tools, hook=UserWrites())
    assert [m.role for m in ctx.messages] == ["user", "assistant", "tool", "user", "assistant"]
    assert ctx.messages[3].content == "also in Cusco" and ctx.messages[3].source == "u"
    assert ctx.inbox == [] and model.sent[1][0] == ["user", "assistant", "tool", "user"]


async def test_what_the_user_writes_while_the_model_answers_keeps_the_turn_going():
    ctx, tools = setup()

    class Writes(Model):
        async def run(self, messages, tools=None, output_format=None, **options):
            if not self.sent:
                ctx.inbox.append("wait, in Cusco")
            return await super().run(messages, tools, output_format, **options)

    state = LoopState()
    events = await run(ReactLoop(), ctx, Writes(says("sunny in Lima"), says("sunny in Cusco")), tools, state=state)
    assert [m.role for m in ctx.messages] == ["user", "assistant", "user", "assistant"]
    assert (state.iteration, kinds(events)) == (2, [])  # a user message, not a harness note


async def test_a_question_left_waiting_is_closed_when_the_user_writes_something_else():
    class UserWrites(LoopHook):
        async def after_model(self, ctx, state):
            if state.iteration == 1:
                ctx.inbox.append("never mind, say hi")
            return await super().after_model(ctx, state)

    ctx, tools = setup(approval="ask_for_approval")
    model, state = Model(asks(("c1", "weather", {"city": "Lima"})), says("hi")), LoopState()
    await run(ReactLoop(), ctx, model, tools, hook=UserWrites(), state=state)
    assert WEATHER == [] and state.finish_reason == "stop"
    assert [m.role for m in ctx.messages] == ["user", "assistant", "tool", "user", "assistant"]
    assert "Do not take this as permission" in ctx.messages[2].content


# -------- THE FORMAT OF THE FINAL ANSWER ------------------------------------------
class City(BaseModel):
    name: str
    sunny: bool


async def test_the_final_answer_is_asked_again_in_the_format_without_touching_the_conversation():
    ctx, tools = setup()
    model = Model(asks(("c1", "weather", {"city": "Lima"})), says("It is sunny in Lima."), says('{"name": "Lima", "sunny": true}'))
    state = LoopState()
    await run(ReactLoop(), ctx, model, tools, state=state, output_format=City)
    assert state.message.structured_output == City(name="Lima", sunny=True)
    assert state.message.content == "It is sunny in Lima." and state.finish_reason == "stop"
    assert [m.role for m in ctx.messages] == ["user", "assistant", "tool", "assistant"]
    # The format call has no tools and ends with the harness asking for the format.
    assert model.sent[2] == (["user", "assistant", "tool", "assistant", "user"], [])
    assert state.usage.llm_calls == 3


async def test_an_answer_out_of_format_is_corrected_once_and_then_the_turn_says_so():
    ctx, tools = setup()
    model = Model(says("Sunny."), says("not json"), says('{"name": "Lima", "sunny": false}'))
    state = LoopState()
    events = await run(ReactLoop(), ctx, model, tools, state=state, output_format=City)
    assert kinds(events) == ["invalid_format"] and state.message.structured_output.sunny is False
    assert model.sent[2][0][-2:] == ["assistant", "user"]  # its wrong answer and the correction
    ctx, tools = setup()
    state = LoopState()
    await run(ReactLoop(), ctx, Model(says("Sunny."), says("no"), says("still no")), tools, state=state, output_format=City)
    assert state.finish_reason == "invalid_format" and state.message.structured_output is None
    assert ctx.messages[-1].content == "Sunny."


async def test_what_happened_outside_is_told_to_the_model_before_its_next_call():
    class Tasks(LoopHook):
        async def notes(self, ctx, state):
            return ["The task t1 ended: 42"] if state.iteration == 2 else []

    ctx, tools = setup()
    model = Model(asks(("c1", "weather", {"city": "Lima"})), says("ok"))
    events = await run(ReactLoop(), ctx, model, tools, hook=Tasks())
    assert kinds(events) == ["note"] and notes(ctx) == ["[harness] The task t1 ended: 42"]
    assert model.sent[1][0] == ["user", "assistant", "tool", "user"]
