from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from max_ai.base.agent import Agent
from max_ai.ui.server import create_app
from max_ai.workspace.system import LocalWorkSpace
from tests.test_agent_run import RunClient


def make_app(tmp_path: Path):
    workspace = LocalWorkSpace(root=tmp_path / "workspace")
    agent = Agent(
        name="upload-agent",
        description="UI upload test agent",
        instructions="Be concise.",
        client=RunClient(),
        workspace=workspace,
    )
    app = create_app(agent, user_id="user_upload", session_id="session_initial")
    return app


def test_binary_upload_persists_and_reports_sync_status(tmp_path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        content = b"\x00\xffbinary\x10payload"
        uploaded = client.post(
            "/api/workspace/upload",
            params={"path": "payload.bin", "session_id": session_id, "agent_name": "upload-agent"},
            content=content,
            headers={"content-type": "application/octet-stream"},
        )

        assert uploaded.status_code == 200
        assert uploaded.json()["path"] == f"{session_id}/payload.bin"
        listing = client.get("/api/workspace/files")
        assert listing.status_code == 200
        item = next(row for row in listing.json() if row["path"] == f"{session_id}/payload.bin")
        assert item["sync_status"]["state"] == "synced"
        assert client.get("/api/workspace/raw", params={"path": item["path"]}).content == content
        metadata = client.get("/api/workspace/file", params={"path": item["path"]})
        assert metadata.json()["sync_status"]["state"] == "synced"

        collision = client.post(
            "/api/workspace/upload",
            params={"path": "payload.bin", "session_id": session_id},
            content=b"second",
            headers={"content-type": "application/octet-stream"},
        )
        assert collision.status_code == 409


def test_upload_rejects_unknown_or_other_user_sessions_and_unsafe_paths(tmp_path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        app.state.sessions["other-user-session"] = {
            "contexts": {
                "upload-agent": type(app.state.contexts["upload-agent"])(
                    user_id="another_user", session_id="other-user-session"
                )
            }
        }
        headers = {"content-type": "application/octet-stream"}
        for session in ("missing-session", "other-user-session"):
            response = client.post(
                "/api/workspace/upload",
                params={"path": "file.bin", "session_id": session},
                content=b"data",
                headers=headers,
            )
            assert response.status_code == 404

        traversal = client.post(
            "/api/workspace/upload",
            params={"path": "../outside.bin", "session_id": session_id},
            content=b"data",
            headers=headers,
        )
        assert traversal.status_code == 400
        assert str(tmp_path) not in traversal.text


def test_upload_enforces_streamed_size_limit_and_sync_finds_external_files(tmp_path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        too_large = client.post(
            "/api/workspace/upload",
            params={"path": "large.bin", "session_id": session_id},
            content=b"x" * (8 * 1024 * 1024 + 1),
            headers={"content-type": "application/octet-stream"},
        )
        assert too_large.status_code == 413

        external = app.state.agents["upload-agent"].workspace.base_root
        external_path = external / "user_upload" / session_id / "external.txt"
        external_path.parent.mkdir(parents=True, exist_ok=True)
        external_path.write_bytes(b"created outside the UI")

        report = client.get("/api/workspace/sync")
        assert report.status_code == 200
        external_item = next(row for row in report.json()["items"] if row["path"] == f"{session_id}/external.txt")
        assert external_item["state"] in {"synced", "pending"}
        listing = client.get("/api/workspace/files").json()
        assert any(row["path"] == external_item["path"] for row in listing)

