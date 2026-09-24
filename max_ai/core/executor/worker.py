"""Run one native tool inside an image containing max_ai and its dependencies."""

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

from ...base.tools import ToolContext
from ...capabilities.executor.local import LocalExecutor
from ...capabilities.workspace.local import LocalWorkspace
from ...types.tool_call import ToolCallRecord, ToolResult
from .reference import ToolReference


async def invoke(payload):
    record = ToolCallRecord.model_validate(payload["record"])
    context_data = payload["context"]
    root = Path(payload["workspace_path"])
    if root != Path(os.environ["WORKSPACE"]) or root.name != context_data["user_id"]:
        raise ValueError("Worker workspace identity mismatch")
    workspace = LocalWorkspace(root=root.parent)
    tool = ToolReference.model_validate(payload["reference"]).build()
    if tool.name != record.tool_name:
        raise ValueError("Remote tool name differs from approved tool")
    if tool.parameters != payload["tool_parameters"]:
        raise ValueError("Remote tool schema differs from the registered tool")
    events = []
    context = ToolContext(
        **context_data, deps={"tool_call_id": record.id}, emit_event=events.append,
    )
    executor = LocalExecutor()
    session = await executor.connect(workspace, context.user_id, context.session_id)
    try:
        result = await executor.run_tool(session, tool, record, context)
        metadata = dict(result.metadata)
        metadata["events"] = [event.model_dump(mode="json") for event in events]
        return result.model_copy(update={"metadata": metadata})
    finally:
        await executor.clean(session)


def main():
    payload = json.load(sys.stdin)
    try:
        # Tool prints are diagnostics; stdout is exclusively the result protocol.
        with contextlib.redirect_stdout(sys.stderr):
            result = asyncio.run(invoke(payload))
    except Exception as error:
        result = ToolResult.execution_error(payload["record"]["id"], str(error))
    print(result.model_dump_json(), flush=True)


if __name__ == "__main__":
    main()
