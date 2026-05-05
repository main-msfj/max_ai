from __future__ import annotations

import asyncio
import typing as t
from pathlib import Path

import pytest

from max_ai.base.tools import CoreTool, ToolContext
from max_ai.executor.docker import DockerExecutor
from max_ai.termination import CancellationToken
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import ToolApprovalMode


class CommandTool(CoreTool):
    def __init__(self, timeout_seconds: float = 30) -> None:
        super().__init__(
            name="skill_bash",
            description="Run bash in skills sandbox.",
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            timeout_seconds=timeout_seconds,
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        raise AssertionError("DockerExecutor should not call tool.execute directly")


class FakeProcess:
    def __init__(
        self,
        returncode: int = 0,
        stdout: bytes = b"ok\n",
        stderr: bytes = b"",
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self.stdout, self.stderr

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def make_record(parameters: dict[str, t.Any]) -> ToolCallRecord:
    return ToolCallRecord(tool_name="skill_bash", parameters=parameters)


def make_context(session_id: str = "session_x") -> ToolContext:
    return ToolContext(run_id="run_1", session_id=session_id)


def test_docker_command_uses_default_server_workspace(tmp_path: Path) -> None:
    executor = DockerExecutor(server_workspace=tmp_path)

    command = executor._docker_command("echo hi", make_context("s1"))

    assert command[:3] == ["docker", "run", "--rm"]
    assert "-v" in command
    assert f"{tmp_path.resolve()}:/server_workspace" in command
    assert "-w" in command
    assert "/server_workspace" in command
    assert ["maxai-sandbox:py311", "/bin/bash", "-lc", "echo hi"] == command[-4:]


def test_container_env_uses_configurable_paths(tmp_path: Path) -> None:
    executor = DockerExecutor(
        server_workspace=tmp_path,
        container_workspace="/runtime",
        sessions_subdir="sessions",
        skills_cache_subdir="cache/skills",
    )

    env = executor._container_env(make_context("abc"))

    assert env["SERVER_WORKSPACE"] == "/runtime"
    assert env["SESSIONS_DIR"] == "/runtime/sessions"
    assert env["SKILLS_CACHE_DIR"] == "/runtime/cache/skills"
    assert env["SKILLS_DIR"] == "/runtime/sessions/abc/skills"
    assert env["RUN_ID"] == "run_1"
    assert env["SESSION_ID"] == "abc"


def test_invalid_container_paths_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DockerExecutor(server_workspace=tmp_path, container_workspace="/")

    with pytest.raises(ValueError):
        DockerExecutor(server_workspace=tmp_path, sessions_subdir="../sessions")

    with pytest.raises(ValueError):
        DockerExecutor(server_workspace=tmp_path, skills_cache_subdir="")


@pytest.mark.asyncio
async def test_run_requires_command_argument(tmp_path: Path) -> None:
    executor = DockerExecutor(server_workspace=tmp_path)

    result = await executor.run(
        CommandTool(),
        make_record(parameters={}),
        make_context(),
    )

    assert result.success is False
    assert "command" in (result.error or "")


@pytest.mark.asyncio
async def test_run_returns_stdout_stderr_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, t.Any] = {}

    async def fake_create_subprocess_exec(*cmd: str, **kwargs: t.Any) -> FakeProcess:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProcess(stdout=b"hello\n", stderr=b"note\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    executor = DockerExecutor(server_workspace=tmp_path)
    result = await executor.run(
        CommandTool(),
        make_record(parameters={"command": "printf hello"}),
        make_context("s1"),
    )

    assert result.success is True
    assert result.result == {
        "exit_code": 0,
        "stdout": "hello\n",
        "stderr": "note\n",
    }
    assert captured["cmd"][-4:] == (
        "maxai-sandbox:py311",
        "/bin/bash",
        "-lc",
        "printf hello",
    )
    assert captured["kwargs"]["stdout"] is asyncio.subprocess.PIPE
    assert captured["kwargs"]["stderr"] is asyncio.subprocess.PIPE


@pytest.mark.asyncio
async def test_run_maps_nonzero_exit_to_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_subprocess_exec(*cmd: str, **kwargs: t.Any) -> FakeProcess:
        return FakeProcess(returncode=7, stdout=b"", stderr=b"boom\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    executor = DockerExecutor(server_workspace=tmp_path)
    result = await executor.run(
        CommandTool(),
        make_record(parameters={"command": "false"}),
        make_context(),
    )

    assert result.success is False
    assert "exited with code 7" in (result.error or "")
