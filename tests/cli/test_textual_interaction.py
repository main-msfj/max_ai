"""Real key/button interactions with the Textual CLI."""

import asyncio

import pytest
from textual.widgets import Static, TextArea

from max_ai.agents import Agent
from max_ai.base.loop import LoopState
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.ask_user import AskUserTool
from max_ai.cli.app import MaxAIApp
from max_ai.cli.widgets import InfoPanel
from max_ai.core.event_type import (
    ModelResponseEvent,
    ModelRetryEvent,
)
from max_ai.core.harness import messages as harness
from max_ai.core.messages import AssistantMessage, ToolCall, UserMessage
from max_ai.types.completions import Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord
from tests.cli.test_sessions import EchoLLM


async def _resolve(app, ctx):
    """The calls of ``ctx`` wait for the user: the app asks them."""
    app.context = ctx
    return await app._resolve_requests()


def make_app(tmp_path, tools=()):
    agent = Agent("test", "d", "i", EchoLLM(), toolset=list(tools), sandbox=LocalSandbox(str(tmp_path / "work")))
    return MaxAIApp(agent)


@pytest.mark.asyncio
async def test_enter_submits_and_busy_draft_is_preserved(tmp_path):
    app = make_app(tmp_path)
    tasks = []
    app._run_turn = tasks.append
    async with app.run_test() as pilot:
        prompt = app.query_one(TextArea)
        prompt.text = "hello"
        await pilot.press("enter")
        assert tasks == ["hello"]
        prompt.text = "draft"
        await pilot.press("enter")
        assert prompt.text == "draft"
        await pilot.press("shift+enter")
        assert "\n" in prompt.text


@pytest.mark.asyncio
async def test_enter_during_a_model_turn_sends_to_the_inbox(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        app._busy, app._job = True, object()
        prompt = app.query_one(TextArea)
        prompt.text = "also this"
        await pilot.press("enter")
        assert app.context.inbox == ["also this"] and prompt.text == ""
        prompt.text = "/help"
        await pilot.press("enter")
        assert app.context.inbox == ["also this"] and prompt.text == "/help"


@pytest.mark.asyncio
async def test_messages_left_in_the_inbox_start_the_next_turn(tmp_path):
    app = make_app(tmp_path)
    tasks = []

    async def run_once(task):
        tasks.append(task)
        # The first run ends with messages still in the inbox; the second reads them.
        if len(tasks) == 1:
            app.context.inbox += ["one", "two"]
        else:
            app.context.inbox.clear()
        return LoopState(finish_reason="stop")

    app._run_once = run_once
    async with app.run_test():
        await app._run_turn("first").wait()
    assert tasks == ["first", None]


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["yes", "no"])
async def test_approval_enter_while_busy(tmp_path, answer):
    app = make_app(tmp_path)
    ctx = RunContext()
    record = ToolCallRecord(id="a", tool_name="send_email", parameters={})
    ctx.tool_state.add(record)
    response = ctx
    async with app.run_test() as pilot:
        app._busy = True
        worker = asyncio.create_task(_resolve(app, response))
        await pilot.pause()
        app.query_one(TextArea).text = answer
        await pilot.press("enter")
        await asyncio.wait_for(worker, 2)
        assert not record.is_pending_approval


@pytest.mark.asyncio
async def test_always_allow_keeps_the_rule_in_the_sandbox(tmp_path):
    from max_ai.capabilities.tools.bash import BashTool

    app = make_app(tmp_path, [BashTool()])
    ctx = RunContext(user_id="u")
    records = [ToolCallRecord(id=i, tool_name="bash", parameters={"command": c, "description": "d"})
               for i, c in (("a", "ls src"), ("b", "ls tests"), ("c", "rm notes.txt"))]
    for record in records:
        ctx.tool_state.add(record)
    first, second, other = records
    (tmp_path / "work").mkdir()
    await app.agent.sandbox.bind("u").prepare()  # a run would have done it

    async with app.run_test() as pilot:
        app._busy = True
        worker = asyncio.create_task(_resolve(app, ctx))
        await pilot.pause()
        await pilot.press("2")  # Yes, always allow bash(ls:*)
        for _ in range(40):  # the rule is written to setting.json first
            await pilot.pause(0.05)
            if other.is_pending_approval and not second.is_pending_approval:
                break
        # The next ls is covered by the prefix; rm still asks, with no "always".
        assert not first.is_pending_approval and not second.is_pending_approval
        assert not worker.done() and other.is_pending_approval
        await pilot.press("2")  # No, deny
        await asyncio.wait_for(worker, 2)
        assert not other.is_pending_approval and other.status.value != "approved"
    allowed = (await app.agent.sandbox.bind("u").permissions())["allow"]
    assert allowed == ["bash(ls:*)"]


def _ask(text: str, options: list[str]) -> dict:
    return {"questions": [{"question": text, "options": [{"label": o} for o in options]}]}


@pytest.mark.asyncio
async def test_question_button_and_token_events(tmp_path):
    app = make_app(tmp_path)
    ctx = RunContext()
    record = ToolCallRecord(id="q", tool_name=AskUserTool.TOOL_NAME, parameters=_ask("Format?", ["CSV", "JSON"]))
    record.await_user_input()
    ctx.tool_state.add(record)
    response = ctx
    async with app.run_test(size=(120, 45)) as pilot:
        app._busy = True
        worker = asyncio.create_task(_resolve(app, response))
        await pilot.pause()
        await pilot.press("enter")  # question form: enter picks the highlighted option
        await asyncio.wait_for(worker, 2)
        assert record.result.result == {"answers": {"Format?": "CSV"}}
        for _ in range(2):
            await app._write_event(ModelResponseEvent(source="test", response="", usage=Usage(tokens_input=10, tokens_output=3, tokens_cached=2)))
        assert app._tokens_input == 20
        assert app._tokens_output == 6
        usage = str(app.query_one("#usage", Static).render())
        assert "↑ 20 ↓ 6 · cached 4" in usage


def test_tool_summaries_stay_on_one_short_line():
    from max_ai.cli.blocks import tool_summary

    script = "cat > plan.py << 'EOF'\n" + "x = 1\n" * 200 + "EOF"
    assert tool_summary("bash", {"command": script}) == "cat > plan.py << 'EOF' … (+201 lines)"
    assert tool_summary("write_file", {"file_path": "a.py", "content": "x\n" * 300}) == "a.py"
    assert len(tool_summary("bash", {"command": "echo " + "a" * 500})) == 100


def _notes(app):
    from max_ai.cli.blocks import NoteLine
    return [str(note.render()) for note in app.query(NoteLine)]


@pytest.mark.asyncio
async def test_mcp_servers_are_listed_at_start_and_with_slash_mcp(tmp_path, monkeypatch):
    from mcp import Client
    from mcp.server import MCPServer

    from max_ai.capabilities.mcp import StdioMCPServerConfig
    from max_ai.capabilities.mcp import client_manager as module
    from max_ai.types.tools import ToolApprovalMode

    server = MCPServer("acme_hr")

    @server.tool()
    async def send_email(to: str) -> str:
        return to

    @server.tool()
    async def calculate(a: int) -> int:
        return a

    monkeypatch.setattr(module, "create_mcp_client", lambda config, extensions=None: Client(server))
    config = StdioMCPServerConfig(server_id="acme_hr", command="unused", tool_approval_modes={
        "send_email": ToolApprovalMode.ASK_APPROVED, "calculate": ToolApprovalMode.AUTO_APPROVED})
    agent = Agent("test", "d", "i", EchoLLM(), mcp=[config], sandbox=LocalSandbox(str(tmp_path / "work")))
    app = MaxAIApp(agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert any("acme_hr" in note for note in _notes(app))
        await app._run_command("/mcp")
        await pilot.pause()
        # The listing opens in a panel above the composer, not in the transcript.
        panel = app.query_one(InfoPanel)
        assert panel.display
        listing = panel.body
        assert "send_email" in listing and "asks approval" in listing and "calculate" in listing
        await pilot.press("escape")
        assert not panel.display
    await agent.close()


@pytest.mark.asyncio
async def test_clicking_a_menu_command_runs_it(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        prompt = app.query_one(TextArea)
        prompt.text = "/hel"
        await pilot.pause()
        await pilot.click("#menu", offset=(3, 0))
        await pilot.pause()
        panel = app.query_one(InfoPanel)
        assert panel.display and panel.title_text == "Help"


def _question_response(*questions: tuple[str, list[str]]):
    ctx = RunContext()
    records = []
    for index, (text, options) in enumerate(questions):
        record = ToolCallRecord(id=f"q{index}", tool_name=AskUserTool.TOOL_NAME, parameters=_ask(text, options))
        record.await_user_input()
        ctx.tool_state.add(record)
        records.append(record)
    return records, ctx


@pytest.mark.asyncio
async def test_a_number_key_picks_the_option(tmp_path):
    app = make_app(tmp_path)
    (record,), response = _question_response(("Format?", ["CSV", "JSON", "XML"]))
    async with app.run_test(size=(120, 45)) as pilot:
        app._busy = True
        worker = asyncio.create_task(_resolve(app, response))
        await pilot.pause()
        await pilot.press("2")  # no enter needed
        await asyncio.wait_for(worker, 2)
        assert record.result.result == {"answers": {"Format?": "JSON"}} and app.query_one(TextArea).text == ""


@pytest.mark.asyncio
async def test_other_is_typed_inside_the_question_box(tmp_path):
    from textual.widgets import Input

    from max_ai.cli.blocks import QuestionForm

    app = make_app(tmp_path)
    (record,), response = _question_response(("Format?", ["CSV", "JSON"]))
    async with app.run_test(size=(120, 45)) as pilot:
        app._busy = True
        worker = asyncio.create_task(_resolve(app, response))
        await pilot.pause()
        await pilot.press("3")  # "Other"
        await pilot.pause()
        field = app.query_one(QuestionForm).query_one(Input)
        assert app.focused is field
        # esc goes back to the options without cancelling the turn.
        await pilot.press("escape")
        assert app.focused is app.query_one(TextArea) and not worker.done()
        await pilot.press("3")
        await pilot.press(*"parquet", "enter")
        await asyncio.wait_for(worker, 2)
        assert record.result.result == {"answers": {"Format?": "parquet"}}
        summary = str(app.query_one(QuestionForm).query_one(".summary", Static).render())
        assert "Format? → parquet" in summary


def _multi_select_response():
    params = {"questions": [{
        "question": "Files?", "multi_select": True,
        "options": [{"label": "a"}, {"label": "b", "description": "the second"}, {"label": "c"}],
    }]}
    record = ToolCallRecord(id="q", tool_name=AskUserTool.TOOL_NAME, parameters=params)
    record.await_user_input()
    ctx = RunContext()
    ctx.tool_state.add(record)
    return record, ctx


async def _press(pilot, *keys):
    for key in keys:
        await pilot.press(key)
        await pilot.pause()


@pytest.mark.asyncio
async def test_multi_select_toggles_and_confirms_on_submit(tmp_path):
    from max_ai.cli.blocks import OptionRow, QuestionForm

    app = make_app(tmp_path)
    record, response = _multi_select_response()
    async with app.run_test(size=(120, 45)) as pilot:
        app._busy = True
        worker = asyncio.create_task(_resolve(app, response))
        await pilot.pause()
        await _press(pilot, "1", "2", "1")  # the last one takes "a" back
        form = app.query_one(QuestionForm)
        lines = [str(row.render()) for row in form.query(OptionRow) if row.has_class("-current")]
        assert "[ ] a" in lines[0] and "[x] b" in lines[1]
        assert "select all that apply" in str(form.query_one(".question", Static).render())
        assert not worker.done()  # marking never sends
        await _press(pilot, "right", "enter")
        await asyncio.wait_for(worker, 2)
        assert record.result.result == {"answers": {"Files?": ["b"]}}
        assert "Files? → b" in str(form.query_one(".summary", Static).render())


@pytest.mark.asyncio
async def test_multi_select_other_adds_to_the_selection(tmp_path):
    app = make_app(tmp_path)
    record, response = _multi_select_response()
    async with app.run_test(size=(120, 45)) as pilot:
        app._busy = True
        worker = asyncio.create_task(_resolve(app, response))
        await pilot.pause()
        await _press(pilot, "1", "4")  # "Other"
        await pilot.press(*"mine", "enter")
        await pilot.pause()
        await _press(pilot, "right", "enter")
        await asyncio.wait_for(worker, 2)
    assert record.result.result == {"answers": {"Files?": ["a", "mine"]}}


@pytest.mark.asyncio
async def test_multi_select_with_nothing_marked_is_not_answered(tmp_path):
    app = make_app(tmp_path)
    record, response = _multi_select_response()
    async with app.run_test(size=(120, 45)) as pilot:
        app._busy = True
        worker = asyncio.create_task(_resolve(app, response))
        await pilot.pause()
        await _press(pilot, "right", "enter")
        await asyncio.wait_for(worker, 2)
    assert "did not answer" in record.result.result["answers"]["Files?"]


@pytest.mark.asyncio
async def test_esc_keeps_what_the_turn_already_did(tmp_path):
    app = make_app(tmp_path)
    started = asyncio.Event()

    async def run_stream_events(task, ctx, **options):
        ctx.messages.append(UserMessage(source="u", content=task))
        ctx.messages.append(AssistantMessage(source="a", content="", tool_calls=[
            ToolCall(id="p", tool_name="pay", parameters={})]))
        ctx.tool_state.add(ToolCallRecord(id="p", tool_name="pay", parameters={}))
        started.set()
        await asyncio.Event().wait()  # runs until esc
        yield  # pragma: no cover

    app.agent.run_stream_events = run_stream_events
    async with app.run_test() as pilot:
        app._run_turn("hola")
        await asyncio.wait_for(started.wait(), 2)
        await pilot.press("escape")
        while app._busy:
            await pilot.pause(0.01)
        messages = app.context.messages
    assert [m.role for m in messages] == ["user", "assistant", "tool", "user"]
    assert messages[0].content == "hola" and messages[2].success is False
    assert messages[3].content.endswith(harness.TURN_INTERRUPTED)
    assert app.context.tool_state.records["p"].is_consumed


@pytest.mark.asyncio
async def test_submit_accepts_unanswered_questions(tmp_path):
    app = make_app(tmp_path)
    ctx = RunContext()
    params = {"questions": [
        {"question": "A?", "options": [{"label": "x"}, {"label": "y"}]},
        {"question": "B?", "options": [{"label": "z"}, {"label": "w"}]},
    ]}
    record = ToolCallRecord(id="q", tool_name=AskUserTool.TOOL_NAME, parameters=params)
    record.await_user_input()
    ctx.tool_state.add(record)
    response = ctx
    async with app.run_test(size=(120, 45)) as pilot:
        app._busy = True
        worker = asyncio.create_task(_resolve(app, response))
        await pilot.pause()
        await pilot.press("1")  # answers A?, moves on to B?
        await pilot.pause()
        await pilot.press("right")  # the Submit tab
        await pilot.pause()
        await pilot.press("enter")
        await asyncio.wait_for(worker, 2)
    answers = record.result.result["answers"]
    assert answers["A?"] == "x" and "did not answer" in answers["B?"]


@pytest.mark.asyncio
async def test_a_number_key_answers_an_approval(tmp_path):
    app = make_app(tmp_path)
    ctx = RunContext()
    record = ToolCallRecord(id="a", tool_name="send_email", parameters={})
    ctx.tool_state.add(record)
    response = ctx
    async with app.run_test() as pilot:
        app._busy = True
        worker = asyncio.create_task(_resolve(app, response))
        await pilot.pause()
        await pilot.press("1")  # Yes, allow once
        await asyncio.wait_for(worker, 2)
        assert not record.is_pending_approval and record.status.value == "approved"


@pytest.mark.asyncio
async def test_status_shows_a_model_retry_until_the_answer_comes(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test():
        app._begin_busy()
        await app._write_event(ModelRetryEvent(
            source="a", attempt=1, max_attempts=4, delay=4.2, reason="rate_limit"))
        status = str(app.query_one("#status", Static).render())
        assert "Retrying in 4s" in status and "rate limit, try 2/4" in status
        await app._write_event(ModelResponseEvent(source="a", response=""))
        assert "Thinking" in str(app.query_one("#status", Static).render())
