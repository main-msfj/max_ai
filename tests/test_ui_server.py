from __future__ import annotations

from max_ai.base.agent import Agent
from max_ai.core.event_type import CompactionEvent, ToolCallEvent, ToolCallResponseEvent
from max_ai.core.messages import ImagePart, TextPart, UserMessage
from max_ai.types.tool_call import ToolResult
from max_ai.ui.server import _create_session, _event_payload, _serialize_message, create_app
from tests.test_agent_run import RunClient


def test_serialize_message_preserves_image_preview_data():
    message = UserMessage(
        source="user",
        content=[
            TextPart(text="what is this?"),
            ImagePart(data=b"\xff\xd8", mime_type="image/jpeg"),
        ],
    )

    result = _serialize_message(message)

    assert result["content"] == "what is this?"
    assert result["images"] == [
        {
            "mime_type": "image/jpeg",
            "data_base64": "/9g=",
            "data_url": "data:image/jpeg;base64,/9g=",
        }
    ]


def test_event_payload_hides_bash_commands():
    event = ToolCallEvent(
        source="agent",
        tool_name="bash",
        tool_call_id="call_1",
        parameters={"command": 'cat "$SKILLS_DIR/create-ppt/SKILL.md"'},
    )

    result = _event_payload(event)

    assert result["parameters"] == {"command": "[sandbox command hidden]"}
    assert "$SKILLS_DIR" not in str(result)


def test_event_payload_redacts_internal_runtime_paths_from_tool_results():
    event = ToolCallResponseEvent(
        source="agent",
        tool_call_id="call_1",
        tool_result=ToolResult.execution_error(
            "call_1",
            "cannot read /mnt/skills/create-ppt/SKILL.md",
        ),
    )

    result = _event_payload(event)

    assert "/mnt/skills" not in str(result)
    assert "SKILL.md" not in str(result)
    assert "[internal runtime path]" in result["error"]


def test_event_payload_includes_compaction_phase_and_summary():
    event = CompactionEvent(
        source="agent",
        phase="end",
        strategy="SlidingWindowCompaction",
        changed=True,
        old_message_count=2,
        recent_message_count=3,
        old_token_count=1200,
        recent_token_count=500,
        total_token_count=1700,
        live_message_threshold_tokens=1000,
        live_message_budget_tokens=500,
        summary='{"summary": "done"}',
    )

    result = _event_payload(event)

    assert result["type"] == "compaction"
    assert result["event_type"] == "compaction"
    assert result["phase"] == "end"
    assert result["changed"] is True
    assert result["summary"] == '{"summary": "done"}'


def make_ui_agent() -> Agent:
    return Agent(
        name="ui-agent",
        description="UI test agent",
        instructions="be concise",
        client=RunClient(),
    )


def test_create_app_uses_explicit_user_id_and_session_id():
    app = create_app(
        make_ui_agent(),
        user_id="user_fixed",
        session_id="session_fixed",
    )

    payload = _create_session(app)
    ctx = app.state.sessions["session_fixed"]["contexts"]["ui-agent"]

    assert payload["user_id"] == "user_fixed"
    assert payload["session_id"] == "session_fixed"
    assert ctx.user_id == "user_fixed"
    assert ctx.session_id == "session_fixed"


def test_create_app_generates_runtime_user_id_and_session_id_when_missing():
    app = create_app(make_ui_agent())

    payload = _create_session(app)
    ctx = app.state.sessions[payload["session_id"]]["contexts"]["ui-agent"]

    assert payload["user_id"].startswith("user_")
    assert payload["session_id"]
    assert ctx.user_id == payload["user_id"]
    assert ctx.session_id == payload["session_id"]
