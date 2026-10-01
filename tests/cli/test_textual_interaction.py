"""Real key/button interactions with the Textual CLI."""

import asyncio
from types import SimpleNamespace

import pytest
from textual.widgets import Static, TextArea

from max_ai.cli.app import MaxAIApp
from max_ai.cli.events import event_line
from max_ai.cli.widgets import InfoPanel
from max_ai.core.event_type import (
    BashFinishedEvent,
    FileReadEvent,
    ModelResponseEvent,
    ModelRetryEvent,
)
from max_ai.types.agent_response import AgentResponse
from max_ai.types.completions import Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord


def make_app(tmp_path):
    return MaxAIApp(SimpleNamespace(name="test", memory=None, skills=None, knowledge=[],
                                   workspace=SimpleNamespace(base_root=tmp_path)))


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
@pytest.mark.parametrize("answer", ["yes", "no"])
async def test_approval_enter_while_busy(tmp_path, answer):
    app = make_app(tmp_path)
    ctx = RunContext()
    record = ToolCallRecord(id="a", tool_name="send_email", parameters={})
    ctx.tool_state.add(record)
    response = AgentResponse(source="test", context=ctx, usage=Usage(), finish_reason="approval_needed")
    async with app.run_test() as pilot:
        app._busy = True
        worker = asyncio.create_task(app._resolve_requests(response))
        await pilot.pause()
        app.query_one(TextArea).text = answer
        await pilot.press("enter")
        await asyncio.wait_for(worker, 2)
        assert not record.is_pending_approval


@pytest.mark.asyncio
async def test_always_allow_adds_a_rule_to_the_conversation(tmp_path):
    from max_ai.capabilities.tools.bash import BashTool

    app = MaxAIApp(SimpleNamespace(name="test", memory=None, skills=None, knowledge=[],
                                   tools=[BashTool()], workspace=SimpleNamespace(base_root=tmp_path)))
    ctx = RunContext()

    def pending(record_id, command):
        record = ToolCallRecord(id=record_id, tool_name="bash",
                                parameters={"command": command, "description": "d"})
        ctx.tool_state.add(record)
        return record, AgentResponse(source="test", context=ctx, usage=Usage(),
                                     finish_reason="approval_needed")

    async with app.run_test() as pilot:
        app._busy = True
        first, response = pending("a", "npm install left-pad")
        worker = asyncio.create_task(app._resolve_requests(response))
        await pilot.pause()
        await pilot.press("2")  # Yes, always allow bash(npm:*)
        await asyncio.wait_for(worker, 2)
        # The rule lives in the RunContext, so it is saved with the session.
        assert not first.is_pending_approval and ctx.allowed_rules == ["bash(npm:*)"]

        # Another npm command is covered; anything else still asks.
        second, response = pending("b", "npm test")
        await asyncio.wait_for(app._resolve_requests(response), 2)
        assert not second.is_pending_approval
        other, response = pending("c", "rm notes.txt")
        worker = asyncio.create_task(app._resolve_requests(response))
        await pilot.pause()
        assert not worker.done() and other.is_pending_approval
        await pilot.press("3")
        await asyncio.wait_for(worker, 2)
        assert not other.is_pending_approval


@pytest.mark.asyncio
async def test_question_button_and_token_events(tmp_path):
    app = make_app(tmp_path)
    ctx = RunContext()
    record = ToolCallRecord(id="q", tool_name="ask_user", parameters={})
    record.await_user_input("Format?", ["CSV", "JSON"])
    ctx.tool_state.add(record)
    response = AgentResponse(source="test", context=ctx, usage=Usage(), finish_reason="input_needed")
    async with app.run_test(size=(120, 45)) as pilot:
        app._busy = True
        worker = asyncio.create_task(app._resolve_requests(response))
        await pilot.pause()
        await pilot.press("enter")  # question form: enter picks the highlighted option
        await asyncio.wait_for(worker, 2)
        assert record.user_answer == "CSV"
        for _ in range(2):
            await app._write_event(ModelResponseEvent(source="test", response="", usage=Usage(tokens_input=10, tokens_output=3, tokens_cached=2)))
        assert app._tokens_input == 20
        assert app._tokens_output == 6
        usage = str(app.query_one("#usage", Static).render())
        assert "in 20 · out 6 · cached 4" in usage


def test_runtime_event_lines():
    bash = event_line(BashFinishedEvent(source="test", tool_call_id="b", exit_code=0, duration_ms=12))
    skill = event_line(FileReadEvent(source="test", tool_call_id="s", path="skills/demo/SKILL.md", root_dir="/tmp", content_hash="abc"))
    assert "bash finished" in bash.plain
    assert "skills/demo/SKILL.md" in skill.plain


def test_tool_summaries_stay_on_one_short_line():
    from max_ai.cli.blocks import tool_summary

    script = "cat > plan.py << 'EOF'\n" + "x = 1\n" * 200 + "EOF"
    assert tool_summary("bash", {"command": script}) == "cat > plan.py << 'EOF' … (+201 lines)"
    assert tool_summary("WriteFile", {"file_path": "a.py", "content": "x\n" * 300}) == "a.py"
    assert len(tool_summary("bash", {"command": "echo " + "a" * 500})) == 100


def _notes(app):
    from max_ai.cli.blocks import NoteLine
    return [str(note.render()) for note in app.query(NoteLine)]


@pytest.mark.asyncio
async def test_mcp_servers_are_listed_at_start_and_with_slash_mcp(tmp_path):
    from max_ai.types.tools import ToolApprovalMode

    tools = [
        SimpleNamespace(name="acme_hr_send_email", server_id="acme_hr", description="d",
                        approval_mode=ToolApprovalMode.ASK_APPROVED),
        SimpleNamespace(name="acme_hr_calculate", server_id="acme_hr", description="d",
                        approval_mode=ToolApprovalMode.AUTO_APPROVED),
        SimpleNamespace(name="acme_hr_read_resource", server_id="acme_hr", description="d",
                        available_resources=[SimpleNamespace(uri="acme://guides/parking")],
                        resource_templates=[]),
        SimpleNamespace(name="bash", description="d"),
    ]
    app = MaxAIApp(SimpleNamespace(name="test", memory=None, skills=None, knowledge=[], tools=tools,
                                   workspace=SimpleNamespace(base_root=tmp_path)))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert any("acme_hr (2 tools, 1 resources)" in note for note in _notes(app))
        await app._run_command("/mcp")
        await pilot.pause()
        # The listing opens in a panel above the composer, not in the transcript.
        panel = app.query_one(InfoPanel)
        assert panel.display
        listing = panel.body
        assert "send_email" in listing and "asks approval" in listing
        assert "acme://guides/parking" in listing and "bash" not in listing
        await pilot.press("escape")
        assert not panel.display


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
        record = ToolCallRecord(id=f"q{index}", tool_name="ask_user", parameters={})
        record.await_user_input(text, options)
        ctx.tool_state.add(record)
        records.append(record)
    return records, AgentResponse(source="test", context=ctx, usage=Usage(), finish_reason="input_needed")


@pytest.mark.asyncio
async def test_a_number_key_picks_the_option(tmp_path):
    app = make_app(tmp_path)
    (record,), response = _question_response(("Format?", ["CSV", "JSON", "XML"]))
    async with app.run_test(size=(120, 45)) as pilot:
        app._busy = True
        worker = asyncio.create_task(app._resolve_requests(response))
        await pilot.pause()
        await pilot.press("2")  # no enter needed
        await asyncio.wait_for(worker, 2)
        assert record.user_answer == "JSON" and app.query_one(TextArea).text == ""


@pytest.mark.asyncio
async def test_other_is_typed_inside_the_question_box(tmp_path):
    from textual.widgets import Input

    from max_ai.cli.blocks import QuestionForm

    app = make_app(tmp_path)
    (record,), response = _question_response(("Format?", ["CSV", "JSON"]))
    async with app.run_test(size=(120, 45)) as pilot:
        app._busy = True
        worker = asyncio.create_task(app._resolve_requests(response))
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
        assert record.user_answer == "parquet"
        summary = str(app.query_one(QuestionForm).query_one(".summary", Static).render())
        assert "Format?" in summary and "⎿  parquet" in summary


@pytest.mark.asyncio
async def test_a_number_key_answers_an_approval(tmp_path):
    app = make_app(tmp_path)
    ctx = RunContext()
    record = ToolCallRecord(id="a", tool_name="send_email", parameters={})
    ctx.tool_state.add(record)
    response = AgentResponse(source="test", context=ctx, usage=Usage(), finish_reason="approval_needed")
    async with app.run_test() as pilot:
        app._busy = True
        worker = asyncio.create_task(app._resolve_requests(response))
        await pilot.pause()
        await pilot.press("2")  # unknown tool: no "always", so 2 is "No, deny"
        await asyncio.wait_for(worker, 2)
        assert not record.is_pending_approval and record.status.value != "approved"


@pytest.mark.asyncio
async def test_the_turn_summary_shows_the_gate(tmp_path):
    from max_ai.base.completion_gate import CompletionDecision

    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        done = AgentResponse(source="test", context=RunContext(), usage=Usage(), finish_reason="stop",
                             completion=CompletionDecision(status="completed"))
        app._gate_retries = ["plan still open"]
        await app._write_turn_summary(done)
        await pilot.pause()
        assert "gate ✓ after 1 fix" in _notes(app)[-1]


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
