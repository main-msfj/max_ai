from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from max_ai.executor.docker.worker import _main_async
from max_ai.types.tool_call import ToolResult


def _payload() -> dict[str, object]:
    return {
        "tool_ref": {
            "kind": "function",
            "module": "docker_tools",
            "qualname": "add",
            "options": {
                "name": "add",
                "description": "Execute add",
                "version": "1.0.0",
                "approval_mode": "auto_approval",
                "timeout_seconds": 300,
                "max_retries": 3,
            },
            "config": {},
        },
        "record": {
            "tool_name": "add",
            "parameters": {"a": 2, "b": 3},
        },
        "context": {
            "run_id": "run_1",
            "session_id": "session_1",
            "retry_count": 0,
            "deps": {},
        },
    }


@pytest.mark.asyncio
async def test_worker_executes_decorated_function_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_source = tmp_path / "tools"
    tool_source.mkdir()
    (tool_source / "docker_tools.py").write_text(
        "\n".join(
            [
                "from max_ai.tools import tool",
                "",
                "@tool",
                "def add(a: int, b: int) -> int:",
                "    return a + b",
            ]
        ),
        encoding="utf-8",
    )

    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")

    monkeypatch.setenv("MAX_AI_TOOL_SOURCE_DIR", str(tool_source))
    monkeypatch.syspath_prepend(str(tool_source))
    sys.modules.pop("docker_tools", None)

    exit_code = await _main_async(input_path, output_path)

    assert exit_code == 0
    result = ToolResult.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert result.success is True
    assert result.result == 5
    assert result.metadata["executor"] == "docker"
    assert result.metadata["tool_module"] == "docker_tools"
    assert result.metadata["tool_qualname"] == "add"


@pytest.mark.asyncio
async def test_worker_reads_stdin_and_writes_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool_source = tmp_path / "tools"
    tool_source.mkdir()
    (tool_source / "docker_tools.py").write_text(
        "\n".join(
            [
                "from max_ai.tools import tool",
                "",
                "@tool",
                "def add(a: int, b: int) -> int:",
                "    return a + b",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("MAX_AI_TOOL_SOURCE_DIR", str(tool_source))
    monkeypatch.syspath_prepend(str(tool_source))
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps(_payload())))
    sys.modules.pop("docker_tools", None)

    exit_code = await _main_async(Path("-"), Path("-"))

    assert exit_code == 0
    stdout = capsys.readouterr().out
    result = ToolResult.model_validate_json(stdout.splitlines()[-1])
    assert result.success is True
    assert result.result == 5


@pytest.mark.asyncio
async def test_worker_reports_missing_input_path(tmp_path: Path) -> None:
    input_path = tmp_path / "missing.json"
    output_path = tmp_path / "output.json"

    exit_code = await _main_async(input_path, output_path)

    assert exit_code == 0
    result = ToolResult.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert result.success is False
    assert "FileNotFoundError" in (result.error or "")
