"""Remote workspaces: files follow the user across processes; only changes move."""

from __future__ import annotations

import hashlib
import os
import uuid

import pytest

from max_ai.agents import Agent
from max_ai.capabilities.workspace import (
    AzureBlobWorkspace,
    MinIOWorkspace,
    RemoteWorkspace,
)
from max_ai.capabilities.workspace._remote import RemoteObject
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.core.model.llm import ModelConfig
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode


class MemoryStore(RemoteWorkspace):
    """Object storage in a dict, shared by every 'process' that uses it."""

    def __init__(self, objects: dict, cache_dir) -> None:
        self.objects = objects
        self.puts: list[str] = []
        super().__init__(cache_dir)

    def _location(self) -> str:
        return "memory://test"

    async def _list(self, prefix):
        return [RemoteObject(k[len(prefix):], sha) for k, (_, sha) in self.objects.items() if k.startswith(prefix)]

    async def _get(self, key):
        return self.objects[key][0]

    async def _put(self, key, data, sha256):
        self.puts.append(key)
        self.objects[key] = (data, sha256)

    async def _delete(self, key):
        self.objects.pop(key, None)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def test_files_follow_the_user_to_another_process(tmp_path):
    remote: dict = {}
    first = MemoryStore(remote, tmp_path / "process-1")
    await first.download("ana")
    (first._local("ana") / "notes").mkdir()
    (first._local("ana") / "notes" / "a.txt").write_text("hola")
    await first.upload("ana")
    assert remote["ana/workspace/notes/a.txt"] == (b"hola", sha(b"hola"))

    second = MemoryStore(remote, tmp_path / "process-2")  # a new serverless invocation
    await second.download("ana")
    assert (second._local("ana") / "notes" / "a.txt").read_text() == "hola"

    await second.upload("ana")
    assert second.puts == []  # nothing changed: nothing sent


async def test_edits_and_deletes_travel_both_ways(tmp_path):
    remote: dict = {}
    a, b = MemoryStore(remote, tmp_path / "a"), MemoryStore(remote, tmp_path / "b")
    await a.download("ana")
    (a._local("ana") / "x.txt").write_text("v1")
    (a._local("ana") / "y.txt").write_text("keep")
    await a.upload("ana")

    await b.download("ana")
    (b._local("ana") / "x.txt").write_text("v2")
    (b._local("ana") / "y.txt").unlink()
    await b.upload("ana")
    assert b.puts == ["ana/workspace/x.txt"] and "ana/workspace/y.txt" not in remote

    await a.download("ana")
    assert (a._local("ana") / "x.txt").read_text() == "v2"
    assert not (a._local("ana") / "y.txt").exists()


async def test_a_new_local_file_is_never_deleted_by_a_download(tmp_path):
    remote: dict = {}
    store = MemoryStore(remote, tmp_path / "p")
    await store.download("ana")
    (store._local("ana") / "draft.txt").write_text("not uploaded yet")
    await store.download("ana")
    assert (store._local("ana") / "draft.txt").exists()


async def test_users_are_isolated(tmp_path):
    remote: dict = {}
    store = MemoryStore(remote, tmp_path / "p")
    await store.download("ana")
    (store._local("ana") / "secret.txt").write_text("ana")
    await store.upload("ana")
    other = MemoryStore(remote, tmp_path / "q")
    await other.download("beto")
    assert list(other._local("beto").iterdir()) == []


class WriteFileLLM:
    model = "fake"
    config = ModelConfig()
    generation_options = {"max_tokens": 100}

    def __init__(self):
        self.done = False

    async def run(self, **kwargs):
        if self.done:
            message = AssistantMessage(source="llm", content="guardado")
        else:
            self.done = True
            message = AssistantMessage(source="llm", content="", tool_calls=[ToolCall(
                id="w", tool_name="write_file", parameters={"file_name": "report.md", "content": "# hi"})])
        return ChatCompletionResult(message=message, usage=Usage(), model="fake", finish_reason="stop")


async def test_the_agent_syncs_the_workspace_around_each_run(tmp_path):
    remote: dict = {}
    agent = Agent(name="a", description="d", instructions="i", client=WriteFileLLM(),
                  workspace=MemoryStore(remote, tmp_path / "server"))
    agent._registry.get("write_file").approval_mode = ToolApprovalMode.AUTO_APPROVED
    async with agent:
        await agent.run("escribe", run_context=RunContext(user_id="ana"))
    assert remote["ana/workspace/report.md"][0] == b"# hi"


def test_configs_hold_no_secrets(monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_KEY", "super-secret")
    azure = AzureBlobWorkspace("https://acct.blob.core.windows.net/work")
    stored = azure.serialize().model_dump_json()
    assert "super-secret" not in stored and "AZURE_STORAGE_KEY" in stored
    back = AzureBlobWorkspace.deserialize(stored)
    assert (back.account_name, back.container) == ("acct", "work")
    minio = MinIOWorkspace.deserialize(MinIOWorkspace("http://localhost:9000", bucket="work").serialize())
    assert minio.bucket == "work"


# -------- LIVE (docker-infra) -----------------------------------------------------------
@pytest.mark.skipif(not os.getenv("MAXAI_TEST_AZURITE_URL"), reason="set MAXAI_TEST_AZURITE_URL")
async def test_azure_blob_round_trip(tmp_path):
    url = f"{os.environ['MAXAI_TEST_AZURITE_URL']}/test-{uuid.uuid4().hex[:8]}"
    await live_round_trip(lambda d: AzureBlobWorkspace(url, cache_dir=tmp_path / d))


@pytest.mark.skipif(not os.getenv("MAXAI_TEST_MINIO_URL"), reason="set MAXAI_TEST_MINIO_URL")
async def test_minio_round_trip(tmp_path):
    bucket = f"test-{uuid.uuid4().hex[:8]}"
    await live_round_trip(lambda d: MinIOWorkspace(os.environ["MAXAI_TEST_MINIO_URL"], bucket=bucket,
                                                   cache_dir=tmp_path / d))


async def live_round_trip(make):
    first, second = make("one"), make("two")
    await first.download("ana")
    (first._local("ana") / "docs").mkdir()
    (first._local("ana") / "docs" / "a.txt").write_text("hola")
    await first.upload("ana")
    await second.download("ana")
    assert (second._local("ana") / "docs" / "a.txt").read_text() == "hola"
    (second._local("ana") / "docs" / "a.txt").unlink()
    await second.upload("ana")
    await first.download("ana")
    assert not (first._local("ana") / "docs" / "a.txt").exists()
    await first.disconnect()
    await second.disconnect()
