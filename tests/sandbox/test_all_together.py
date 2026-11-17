"""Everything together, with one small tool: skills reach the sandbox, and
the dispatcher runs a tool that uses them there. Once with the call written
by hand, once with a real model deciding it (``MAXAI_LIVE=1``)."""

import json
import os
import subprocess

import pytest
from dotenv import load_dotenv

from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.capabilities.sandbox.docker import DockerSandbox
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.skills import LocalSkills
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.core.tool import ToolDispatcher, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import ToolApprovalMode

from .test_docker_sandbox import needs_docker

load_dotenv()
MODEL = "mistralai/mistral-small-3.2-24b-instruct"
SCRIPT = 'echo "$GREETING from $(basename $PWD)"\n'


class RunSkill(FunctionAsTool):
    """Runs a skill's script inside the sandbox, from the session's folder."""

    sandbox = True

    async def execute(self, tool_request, tool_context=None, cancellation_token=None):
        sandbox, name = tool_context.sandbox, tool_request.parameters["name"]
        folder = sandbox.layout.workspace / tool_context.session_id
        ran = await sandbox.execute_command(
            f"mkdir -p {folder} && cd {folder} && bash {sandbox.layout.skills}/{name}/run.sh"
        )
        return ToolResult.success_result(tool_request.id, ran.stdout.strip())


def run_skill(name: str) -> str:
    """Run the script of an installed skill and give back what it printed."""
    return ""


@pytest.fixture
async def pieces(request, tmp_path, monkeypatch):
    """Skills, this user's sandbox with them installed, and the dispatcher:
    built one by one, the way an Agent will."""
    # 1. Skills: one folder on this machine, and the variable it needs.
    skill = tmp_path / "source/hello"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: hello\ndescription: Say hello from the sandbox\n---\n")
    (skill / "run.sh").write_text(SCRIPT)
    monkeypatch.setenv("MY_GREETING", "hola")
    skills = LocalSkills(str(tmp_path / "source"), ["hello"], skill_env={"GREETING": "MY_GREETING"})

    # 2. Environment: this user's sandbox, with the skills in it.
    if request.param == "local":
        (tmp_path / "env").mkdir()
        shared = LocalSandbox(root=str(tmp_path / "env"))
    else:
        shared = DockerSandbox()
    sandbox = shared.bind(f"test-all-{request.param}", skills.variables())
    await sandbox.prepare()
    assert await skills.sync(sandbox) == ["hello"]

    # 3. Tools, and the dispatcher where it all meets.
    registry = ToolRegistry([RunSkill(run_skill, approval_mode=ToolApprovalMode.AUTO_APPROVED)])
    yield skills, sandbox, ToolDispatcher(registry, sandbox)

    if request.param == "docker":
        await sandbox.close()
        subprocess.run(["docker", "volume", "rm", "-f", sandbox.name], capture_output=True)


SANDBOXES = ["local", pytest.param("docker", marks=needs_docker)]


@pytest.mark.parametrize("pieces", SANDBOXES, indirect=True)
async def test_a_tool_runs_a_skill_inside_the_users_sandbox(pieces):
    _, _, dispatcher = pieces
    ctx = RunContext(user_id="user-1", session_id="s1")
    call = ToolCallRecord(tool_name="run_skill", parameters={"name": "hello"})
    result = await dispatcher.dispatch(call, ctx)
    # "hola" is the skill's variable; "s1" is the session's folder in the sandbox.
    assert result.success and result.result == "hola from s1"


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("MAXAI_LIVE"), reason="Set MAXAI_LIVE=1 to call a real model")
@pytest.mark.parametrize("pieces", SANDBOXES, indirect=True)
async def test_a_real_model_calls_the_tool_and_reads_its_answer(pieces):
    skills, sandbox, dispatcher = pieces
    client = OpenRouterChatCompletionClient(MODEL)
    tools = client.build_tool_schema(dispatcher.registry.loaded_tools())
    messages = [
        {"role": "system", "content": f"Installed skills:\n{await skills.blocks(sandbox)}"},
        {"role": "user", "content": "Run the hello skill and tell me exactly what it printed."},
    ]

    # The model decides the call; the dispatcher runs it.
    asked = (await client.complete(messages, tools, None)).message.tool_calls[0]
    assert asked.parameters == {"name": "hello"}
    record = ToolCallRecord(id=asked.id, tool_name=asked.tool_name, parameters=asked.parameters)
    result = await dispatcher.dispatch(record, RunContext(user_id="user-1", session_id="s1"))
    assert result.result == "hola from s1"

    # The model reads the answer the dispatcher wrote for it.
    answer = dispatcher.tool_message(record, result)
    messages += [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": asked.id,
                    "type": "function",
                    "function": {"name": asked.tool_name, "arguments": json.dumps(asked.parameters)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": answer.tool_call_id, "content": answer.content},
    ]
    said = (await client.complete(messages, tools, None)).message.text()
    assert "hola from s1" in said
