from __future__ import annotations

import typing as t
from pathlib import Path

import pytest

from max_ai.base.agent import Agent
from max_ai.base.clients import CoreChatCompletionClient
from max_ai.capabilities.workspace.local import WorkspaceLocal
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.core.models import ModelConfig
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.workspace.artifacts import LocalArtifactStore
from max_ai.workspace.azure_artifacts import AzureBlobArtifactStore
from max_ai.workspace.system import LocalWorkSpace
from max_ai.tools.file_system import FileSystemTools


class ApprovalClient(CoreChatCompletionClient):
    def __init__(self, responses: list[ChatCompletionResult]) -> None:
        super().__init__(model="artifact-integration", config=ModelConfig())
        self.responses = responses

    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        return Usage()

    def format_messages(self, ctx, prompts):
        return list(ctx.messages)

    def build_api_messages(self, messages):
        return []

    def build_tool_schema(self, tools):
        return []

    async def complete(self, messages, tools, output_format, **kwargs):
        return self.responses.pop(0)

    async def stream(self, messages, tools, output_format, **kwargs):
        raise NotImplementedError


def _tool_call() -> ChatCompletionResult:
    return ChatCompletionResult(
        message=AssistantMessage(
            source="artifact-integration",
            tool_calls=[
                ToolCall(
                    id="write-report",
                    tool_name="write_file",
                    parameters={"path": "report.txt", "content": "approved report"},
                )
            ],
        ),
        usage=Usage(llm_calls=1, attempts_to_call_api=1),
        model="artifact-integration",
        finish_reason="tool_calls",
    )


def _final() -> ChatCompletionResult:
    return ChatCompletionResult(
        message=AssistantMessage(source="artifact-integration", content="saved"),
        usage=Usage(llm_calls=1, attempts_to_call_api=1),
        model="artifact-integration",
        finish_reason="stop",
    )


def _agent(root: Path, store_root: Path, responses: list[ChatCompletionResult]) -> Agent:
    return Agent(
        name="artifact-agent",
        description="artifact persistence integration",
        instructions="Save the requested report.",
        client=ApprovalClient(responses),
        toolset=FileSystemTools().tools,
        workspace=WorkspaceLocal(
            root=root,
            artifact_store=LocalArtifactStore(store_root),
        ),
    )


@pytest.mark.asyncio
async def test_agent_approval_resume_persists_artifact_across_recreated_agent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    store_root = tmp_path / "artifact-db"
    first_agent = _agent(root, store_root, [_tool_call()])
    paused = await first_agent.run(
        "Save a report.",
        run_context=RunContext(user_id="tenant-1", session_id="conversation-1"),
    )

    assert paused.finish_reason == "approval_needed"
    assert paused.context is not None
    paused_context = RunContext.model_validate(paused.context.model_dump())
    pending = paused.pending_approvals[0]
    paused_context.tool_state.apply_approval(
        pending.id, approved=True, reason="approved for artifact integration"
    )

    # Recreate the agent and the store object as a new process would after a
    # persisted approval checkpoint has been loaded.
    resumed_agent = _agent(root, store_root, [_final()])
    completed = await resumed_agent.resume(run_context=paused_context)

    assert completed.finish_reason == "stop"
    assert completed.context is not None
    assert completed.context.runtime_state.shared_state["workspace_sync"]["items"]
    artifact, data = LocalArtifactStore(store_root).get(
        "tenant-1", "conversation-1/report.txt"
    )
    assert data == b"approved report"
    assert artifact.sha256


def test_legacy_helpers_import_binary_and_restore_remote_files(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    store_root = tmp_path / "artifacts"
    workspace = WorkspaceLocal(
        root=root, artifact_store=LocalArtifactStore(store_root)
    )
    source = tmp_path / "payload.bin"
    source.write_bytes(b"\x00\xffbinary")

    imported = workspace.save_or_upload("alice", source)
    assert imported == "legacy/payload.bin"
    assert workspace.list_files("alice") == ["legacy/payload.bin"]
    assert workspace.get_or_download("alice", "payload.bin").read_bytes() == b"\x00\xffbinary"

    LocalArtifactStore(store_root).put("alice", "other-session/report.txt", b"remote")
    fresh_workspace = WorkspaceLocal(
        root=root, artifact_store=LocalArtifactStore(store_root)
    )
    restored = fresh_workspace.get_or_download("alice", "other-session/report.txt")
    assert restored.read_bytes() == b"remote"
    assert fresh_workspace.get_artifacts_dir("alice", "other-session").name == "other-session"


def test_workspace_store_config_is_safe_and_rejects_unknown_store(tmp_path: Path) -> None:
    azure = AzureBlobArtifactStore(
        "https://example.blob.core.windows.net", "artifacts", prefix="maxai"
    )
    dumped = WorkspaceLocal(root=tmp_path, artifact_store=azure).dump_component()
    rendered = str(dumped.model_dump())
    assert "account_url" in rendered
    assert "credential" not in rendered

    custom = WorkspaceLocal(root=tmp_path, artifact_store=object())
    with pytest.raises(TypeError, match="not serializable"):
        custom.dump_component()


def test_workspace_store_config_rejects_explicit_azure_credentials(
    tmp_path: Path,
) -> None:
    store = AzureBlobArtifactStore(
        "https://example.blob.core.windows.net",
        "artifacts",
        credential="do-not-persist-this",
    )
    with pytest.raises((TypeError, ValueError)):
        WorkspaceLocal(root=tmp_path, artifact_store=store).dump_component()


def test_legacy_helper_validates_regular_file_and_size(tmp_path: Path) -> None:
    workspace = LocalWorkSpace(root=tmp_path / "workspace")
    source = tmp_path / "too-large.bin"
    source.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="8388608"):
        workspace.save_or_upload("alice", source)

    target = workspace.get_artifacts_dir("alice")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link = target / "linked.bin"
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        workspace.get_or_download("alice", "legacy/linked.bin")
