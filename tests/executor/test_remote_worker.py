"""The worker that runs native tools inside Docker/Modal, run as a real process.

Remote executors call ``python -m max_ai.core.executor.worker`` in the
sandbox; nothing else exercises that module, so it is started here exactly
like that (a subprocess, JSON on stdin, the result on stdout).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from max_ai.capabilities.tools.bash import BashTool
from max_ai.core.executor.reference import reference_for
from max_ai.types.tool_call import ToolCallRecord, ToolResult


def test_bash_runs_in_the_users_workspace(tmp_path):
    tool = BashTool()
    record = ToolCallRecord(tool_name="bash", parameters={
        "command": "echo hi > out.txt && pwd", "description": "write a file"})
    user_root = tmp_path / "ana"
    payload = {
        "reference": reference_for(tool).model_dump(mode="json"),
        "record": record.model_dump(mode="json"),
        "context": {"run_id": "r1", "user_id": "ana", "session_id": "s1", "retry_count": 0},
        "workspace_path": str(user_root),
        "tool_parameters": tool.parameters,
    }
    done = subprocess.run(
        [sys.executable, "-m", "max_ai.core.executor.worker"],
        input=json.dumps(payload), capture_output=True, text=True, timeout=60,
        env={**os.environ, "WORKSPACE": str(user_root)},
    )
    assert done.returncode == 0, done.stderr
    result = ToolResult.model_validate_json(done.stdout)
    assert result.tool_call_id == record.id and result.success, result
    assert (user_root / "workspace" / "out.txt").read_text() == "hi\n"
