"""
Docker worker entrypoint.

Runs inside the sandbox container. It receives a JSON invocation from
the host, reconstructs the referenced tool, executes it, and writes a
ToolResult JSON response.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import os
import shutil
import sys
import typing as t
from pathlib import Path

from max_ai.base.tools import CoreTool, ToolContext
from max_ai.tools.function_as_tool import FunctionAsTool
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import DockerToolRef


DEFAULT_TOOL_SOURCE_DIR = "/sandbox/tools"
PAYLOAD_TOOL_SOURCE_DIR = "/tmp/max_ai_tool_sources"


def _add_tool_source_to_path() -> None:
    tool_source_dir = os.environ.get("MAX_AI_TOOL_SOURCE_DIR", DEFAULT_TOOL_SOURCE_DIR)
    if tool_source_dir and tool_source_dir not in sys.path:
        sys.path.insert(0, tool_source_dir)


def _materialize_tool_sources(payload: dict[str, t.Any]) -> None:
    sources = payload.get("tool_sources") or {}
    if not isinstance(sources, dict):
        return

    source_dir = Path(PAYLOAD_TOOL_SOURCE_DIR)
    source_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in sources.items():
        if not isinstance(relative_path, str) or not isinstance(content, str):
            continue

        path = source_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    if sources and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))


def _payload_bytes(payload: object) -> bytes | None:
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if not isinstance(payload, dict):
        return None
    encoding = payload.get("encoding")
    content = payload.get("content")
    if not isinstance(content, str):
        return None
    if encoding == "text":
        return content.encode("utf-8")
    if encoding == "base64":
        try:
            return base64.b64decode(content.encode("ascii"), validate=True)
        except Exception:
            return None
    return None


def _file_payload(path: Path) -> dict[str, str]:
    data = path.read_bytes()
    try:
        return {"encoding": "text", "content": data.decode("utf-8")}
    except UnicodeDecodeError:
        return {
            "encoding": "base64",
            "content": base64.b64encode(data).decode("ascii"),
        }


def _materialize_runtime_files(payload: dict[str, t.Any]) -> None:
    files = payload.get("runtime_files") or {}
    if not isinstance(files, dict):
        return

    root = Path(os.environ.get("RUNTIME_DIR", "/sandbox"))
    for relative_path, file_payload in files.items():
        if not isinstance(relative_path, str):
            continue
        target = (root / relative_path).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            continue
        data = _payload_bytes(file_payload)
        if data is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _normalize_runtime_layout(payload: dict[str, t.Any]) -> None:
    """Flatten accidental ``tmp/{user_id}`` nesting inside the container.

    The host workspace may be ``server/tmp/{user_id}``, but Docker mounts that
    directory as the runtime root. Inside the container the contract is:

        /sandbox/tools
        /sandbox/skills
        /sandbox/artifacts

    Older containers or fallback code can create ``/sandbox/tmp/{user_id}``.
    Merge that layout back into the runtime root before the tool runs.
    """
    runtime_root = Path(os.environ.get("RUNTIME_DIR", "/sandbox")).resolve()
    context_payload = payload.get("context") or {}
    user_id = context_payload.get("user_id") or "default"
    nested_root = (runtime_root / "tmp" / str(user_id)).resolve()

    try:
        nested_root.relative_to(runtime_root)
    except ValueError:
        return

    if not nested_root.exists():
        return

    for name in ("tools", "skills", "artifacts"):
        source = nested_root / name
        target = runtime_root / name
        if not source.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            destination = target / child.name
            if destination.exists():
                if child.is_dir() and destination.is_dir():
                    shutil.copytree(child, destination, dirs_exist_ok=True)
                elif child.is_file():
                    shutil.copy2(child, destination)
            else:
                shutil.move(str(child), str(destination))

    try:
        shutil.rmtree(runtime_root / "tmp")
    except OSError:
        pass


def _collect_workspace_files(context: ToolContext) -> dict[str, dict[str, str]]:
    artifacts_dir = Path(
        os.environ.get(
            "ARTIFACTS_DIR",
            os.environ.get("WORKSPACE_DIR", "/sandbox/artifacts"),
        )
    )
    if not artifacts_dir.exists():
        return {}

    files: dict[str, dict[str, str]] = {}
    root = artifacts_dir.resolve()
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            relative_path = path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        files[relative_path] = _file_payload(path)
    return files


def _resolve_ref(module_name: str, qualname: str) -> t.Any:
    module = importlib.import_module(module_name)

    obj: t.Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)

    return obj


class _WorkerBashTool(CoreTool):
    """Fallback bash implementation for sandbox images without BashTool."""

    def __init__(
        self,
        timeout_seconds: float = 120,
        max_output_chars: int = 20000,
        approval_mode: str = "ask_approved",
    ) -> None:
        super().__init__(
            name="bash",
            description="Run a shell command from the runtime root.",
            approval_mode=approval_mode,
            timeout_seconds=timeout_seconds,
        )
        self.max_output_chars = max_output_chars

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {"type": ["integer", "null"], "minimum": 1},
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: t.Any = None,
    ) -> ToolResult:
        validation = self.validate_parameters(tool_request)
        if not validation.is_tool_valid:
            return ToolResult.invalid_parameters(
                tool_request.id,
                validation.msg_error or "Invalid bash parameters.",
            )

        runtime_root = Path(os.environ.get("RUNTIME_DIR", "/sandbox")).resolve()
        tools_dir = Path(os.environ.get("TOOLS_DIR", runtime_root / "tools")).resolve()
        skills_dir = Path(os.environ.get("SKILLS_DIR", runtime_root / "skills")).resolve()
        artifacts_dir = Path(
            os.environ.get("ARTIFACTS_DIR", os.environ.get("WORKSPACE_DIR", runtime_root / "artifacts"))
        ).resolve()
        for path in (runtime_root, tools_dir, skills_dir, artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)

        command = t.cast(str, tool_request.parameters["command"])
        timeout = tool_request.parameters.get("timeout_seconds") or self.timeout_seconds
        env = os.environ.copy()
        env.update(
            {
                "RUNTIME_DIR": str(runtime_root),
                "TOOLS_DIR": str(tools_dir),
                "SKILLS_DIR": str(skills_dir),
                "ARTIFACTS_DIR": str(artifacts_dir),
                "WORKSPACE_DIR": str(artifacts_dir),
            }
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=runtime_root,
                env=env,
            )
            task = asyncio.create_task(proc.communicate())
            stdout_b, stderr_b = await asyncio.wait_for(task, timeout=float(timeout))
            stdout = stdout_b.decode(errors="replace")
            stderr = stderr_b.decode(errors="replace")
            stdout, stdout_truncated = self._truncate(stdout)
            stderr, stderr_truncated = self._truncate(stderr)
            return ToolResult.success_result(
                tool_request.id,
                {
                    "exit_code": proc.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "cwd": str(runtime_root),
                    "command": command,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                },
                metadata={"name": self.name},
            )
        except asyncio.TimeoutError:
            if "proc" in locals() and proc.returncode is None:
                proc.kill()
                await proc.communicate()
            return ToolResult.timeout(tool_request.id, timeout_seconds=float(timeout))

    def _truncate(self, value: str) -> tuple[str, bool]:
        if len(value) <= self.max_output_chars:
            return value, False
        return value[: self.max_output_chars] + "\n[output truncated]", True


def _build_worker_bash_tool(tool_ref: DockerToolRef) -> CoreTool | None:
    if tool_ref.kind != "class":
        return None
    if tool_ref.module != "max_ai.tools.bash" or tool_ref.qualname != "BashTool":
        return None
    return _WorkerBashTool(**tool_ref.config)


def _build_tool(tool_ref: DockerToolRef) -> CoreTool:
    try:
        obj = _resolve_ref(tool_ref.module, tool_ref.qualname)
    except ModuleNotFoundError:
        fallback = _build_worker_bash_tool(tool_ref)
        if fallback is not None:
            return fallback
        raise

    if tool_ref.kind == "function":
        if isinstance(obj, FunctionAsTool):
            func = obj.func
        elif callable(obj):
            func = obj
        else:
            raise TypeError(
                f"Docker function reference {tool_ref.module}.{tool_ref.qualname} "
                "did not resolve to a callable or FunctionAsTool."
            )

        return FunctionAsTool(
            func=func,
            **tool_ref.options,
        )

    if tool_ref.kind == "class":
        if not isinstance(obj, type):
            raise TypeError(
                f"Docker class reference {tool_ref.module}.{tool_ref.qualname} "
                "did not resolve to a class."
            )

        tool = obj(**tool_ref.config)
        if not isinstance(tool, CoreTool):
            raise TypeError(
                f"Docker class reference {tool_ref.module}.{tool_ref.qualname} "
                "did not construct a CoreTool."
            )

        return tool

    raise ValueError(f"Unsupported Docker tool kind: {tool_ref.kind!r}")


def _build_context(payload: dict[str, t.Any]) -> ToolContext:
    context_payload = payload.get("context") or {}

    return ToolContext(
        run_id=context_payload["run_id"],
        user_id=context_payload.get("user_id", "default"),
        session_id=context_payload["session_id"],
        retry_count=context_payload.get("retry_count", 0),
        deps=context_payload.get("deps") or {},
    )


async def _run_invocation(payload: dict[str, t.Any]) -> ToolResult:
    record = ToolCallRecord.model_validate(payload["record"])
    tool_ref = DockerToolRef.model_validate(payload["tool_ref"])

    _materialize_tool_sources(payload)
    _materialize_runtime_files(payload)
    _normalize_runtime_layout(payload)
    tool = _build_tool(tool_ref)
    context = _build_context(payload)

    result = await tool.execute(
        record,
        context,
        cancellation_token=None,
    )

    metadata = {
        **result.metadata,
        "executor": result.metadata.get("executor", "docker"),
        "tool_module": result.metadata.get("tool_module", tool_ref.module),
        "tool_qualname": result.metadata.get("tool_qualname", tool_ref.qualname),
        "workspace_files": _collect_workspace_files(context),
    }
    return result.model_copy(update={"metadata": metadata})


def _read_input(input_path: Path) -> str:
    if str(input_path) == "-":
        return sys.stdin.read()
    return input_path.read_text(encoding="utf-8")


def _write_result(output_path: Path, result: ToolResult) -> None:
    result_json = result.model_dump_json()
    if str(output_path) == "-":
        print(result_json, flush=True)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result_json, encoding="utf-8")


async def _main_async(input_path: Path, output_path: Path) -> int:
    _add_tool_source_to_path()
    raw_payload: str | None = None

    try:
        raw_payload = _read_input(input_path)
        payload = json.loads(raw_payload)
        result = await _run_invocation(payload)

    except Exception as exc:
        tool_call_id = "unknown"
        if raw_payload:
            try:
                payload = json.loads(raw_payload)
                record_payload = payload.get("record") or {}
                tool_call_id = record_payload.get("id") or "unknown"
            except Exception:
                pass

        result = ToolResult.execution_error(
            tool_call_id,
            f"Docker worker failed: {exc!r}",
        )

    _write_result(output_path, result)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if len(args) != 2:
        print(
            "Usage: python -m max_ai.executor.docker.worker <input.json|-> <output.json|->",
            file=sys.stderr,
        )
        return 2

    input_path = Path(args[0])
    output_path = Path(args[1])

    return asyncio.run(_main_async(input_path, output_path))


if __name__ == "__main__":
    raise SystemExit(main())
