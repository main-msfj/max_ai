from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient

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


def make_ui_agent(name: str = "ui-agent") -> Agent:
    return Agent(
        name=name,
        description="UI test agent",
        instructions="be concise",
        client=RunClient(),
    )


def write_workspace_file(root: Path, user_id: str, session_id: str, path: str, content: str) -> Path:
    target = root / user_id / session_id / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


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


def test_workspace_lists_files_from_all_user_sessions_and_serves_safe_bytes(tmp_path):
    workspace_root = tmp_path / "workspace"
    app = create_app(
        make_ui_agent(),
        workspace_root=workspace_root,
        user_id="user_fixed",
        session_id="initial",
    )
    write_workspace_file(
        workspace_root, "user_fixed", "session_one", "reports/report #1.md", "# first session"
    )
    write_workspace_file(
        workspace_root, "user_fixed", "session_two", "reports/report #1.md", "# second session"
    )
    write_workspace_file(
        workspace_root, "user_fixed", "session_two", "page.html", "<script>alert(1)</script>"
    )
    write_workspace_file(
        workspace_root, "other_user", "private", "secret.txt", "belongs to another user"
    )

    with TestClient(app) as client:
        response = client.get("/api/workspace/files")
        assert response.status_code == 200
        files = response.json()
        paths = {item["path"] for item in files}
        first_path = "session_one/reports/report #1.md"
        second_path = "session_two/reports/report #1.md"
        html_path = "session_two/page.html"
        assert {first_path, second_path, html_path} <= paths
        assert all("other_user" not in item["path"] for item in files)
        assert app.state.agents["ui-agent"].workspace.base_root == workspace_root.resolve()

        encoded_url = next(item["url"] for item in files if item["path"] == first_path)
        assert parse_qs(urlsplit(encoded_url).query)["path"] == [first_path]

        workspace = client.get("/api/workspace").json()
        assert workspace["root"] == "."
        assert workspace["files"] == files

        metadata = client.get("/api/workspace/file", params={"path": first_path})
        assert metadata.status_code == 200
        assert metadata.json()["content"] == "# first session"
        assert metadata.json()["path"] == first_path

        raw = client.get("/api/workspace/raw", params={"path": html_path})
        assert raw.status_code == 200
        assert raw.content == b"<script>alert(1)</script>"
        assert raw.headers["content-type"] == "application/octet-stream"
        assert raw.headers["content-disposition"].startswith("attachment;")
        assert raw.headers["x-content-type-options"] == "nosniff"


def test_workspace_blocks_traversal_symlinks_and_other_users(tmp_path):
    workspace_root = tmp_path / "workspace"
    app = create_app(
        make_ui_agent(),
        workspace_root=workspace_root,
        user_id="user_fixed",
        session_id="initial",
    )
    write_workspace_file(workspace_root, "user_fixed", "session_one", "safe.txt", "safe")
    other_file = write_workspace_file(
        workspace_root, "other_user", "private", "secret.txt", "private"
    )
    linked_file = workspace_root / "user_fixed" / "session_one" / "linked.txt"
    linked_file.symlink_to(other_file)

    with TestClient(app) as client:
        for path in (
            "../other_user/private/secret.txt",
            "other_user/private/secret.txt",
            "session_one/linked.txt",
        ):
            response = client.get("/api/workspace/raw", params={"path": path})
            assert response.status_code == 404
        traversal_metadata = client.get(
            "/api/workspace/file",
            params={"path": "../../other_user/private/secret.txt"},
        )
        assert traversal_metadata.status_code == 404


def test_clear_preserves_user_and_registered_session_context(tmp_path):
    app = create_app(
        make_ui_agent(),
        workspace_root=tmp_path / "workspace",
        user_id="user_fixed",
        session_id="session_before_clear",
    )
    _create_session(app)
    previous_contexts = app.state.contexts
    previous = previous_contexts["ui-agent"]

    with TestClient(app) as client:
        response = client.post("/api/clear")

    assert response.status_code == 200
    assert app.state.contexts is not previous_contexts
    assert app.state.contexts["ui-agent"].user_id == "user_fixed"
    assert app.state.contexts["ui-agent"].session_id != "session_before_clear"
    assert app.state.sessions["session_before_clear"]["contexts"]["ui-agent"] is previous
    assert app.state.sessions["session_before_clear"]["contexts"]["ui-agent"].session_id == "session_before_clear"


def test_workspace_uses_each_agent_root_and_only_explicit_root_overrides_all(tmp_path):
    first_root = tmp_path / "first-workspace"
    second_root = tmp_path / "second-workspace"
    first_agent = make_ui_agent()
    second_agent = make_ui_agent("second-agent")
    first_agent.workspace.base_root = first_root
    second_agent.workspace.base_root = second_root
    agents = [first_agent, second_agent]
    app = create_app(agents, user_id="user_fixed", session_id="shared-session")

    write_workspace_file(first_root, "user_fixed", "session_one", "first.txt", "first root")
    write_workspace_file(second_root, "user_fixed", "session_two", "second.txt", "second root")

    with TestClient(app) as client:
        first_files = client.get("/api/workspace/files").json()
        assert [item["name"] for item in first_files] == ["first.txt"]
        app.state.agent_name = "second-agent"
        second_files = client.get("/api/workspace/files").json()
        assert [item["name"] for item in second_files] == ["second.txt"]

    assert first_agent.workspace.base_root == first_root.resolve()
    assert second_agent.workspace.base_root == second_root.resolve()
    assert app.state.workspace_root == first_root.resolve()

    override_root = tmp_path / "explicit-workspace"
    override_app = create_app(
        [make_ui_agent(), make_ui_agent("second-agent")],
        workspace_root=override_root,
        user_id="user_fixed",
    )
    assert all(
        agent.workspace.base_root == override_root.resolve()
        for agent in override_app.state.agents.values()
    )
