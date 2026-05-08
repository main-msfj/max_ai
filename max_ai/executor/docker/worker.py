"""
Docker worker entrypoint.

Runs inside the sandbox container. It receives a JSON invocation from
the host, reconstructs the referenced tool, executes it, and writes a
ToolResult JSON response.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
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


def _resolve_ref(module_name: str, qualname: str) -> t.Any:
    module = importlib.import_module(module_name)

    obj: t.Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)

    return obj


def _build_tool(tool_ref: DockerToolRef) -> CoreTool:
    obj = _resolve_ref(tool_ref.module, tool_ref.qualname)

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
        session_id=context_payload["session_id"],
        retry_count=context_payload.get("retry_count", 0),
        deps=context_payload.get("deps") or {},
    )


async def _run_invocation(payload: dict[str, t.Any]) -> ToolResult:
    record = ToolCallRecord.model_validate(payload["record"])
    tool_ref = DockerToolRef.model_validate(payload["tool_ref"])

    _materialize_tool_sources(payload)
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
