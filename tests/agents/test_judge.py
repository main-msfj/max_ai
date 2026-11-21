"""The judge: it only takes a question away from a bash command without a
rule, and anything that goes wrong asks the user as before."""

from __future__ import annotations

import json

import httpx
import pytest

from max_ai.base.tools import ToolContext
from max_ai.capabilities.tools.bash import BashTool
from max_ai.capabilities.tools.filesystem import WriteFileTool
from max_ai.cli.app import MaxAIApp
from max_ai.core.harness.hooks import HarnessToolHook
from max_ai.core.harness.judge import Judge
from max_ai.core.harness.permissions import Permissions
from max_ai.core.messages import HARNESS_SOURCE, UserMessage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode


class Server:
    """Stands in for OpenRouter: answers with ``safe`` and keeps what it got."""

    def __init__(self, safe: float | None = 0.97, status: int = 200) -> None:
        self.safe, self.status, self.calls = safe, status, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(json.loads(request.content))
        if self.safe is None:
            raise httpx.ReadTimeout("slow")
        choice = "safe" if self.safe >= 0.5 else "ask"
        verdict = {"choice": choice, "probabilities": {"safe": self.safe, "ask": 1 - self.safe}}
        return httpx.Response(self.status, json={"answers": {"verdict": verdict}})


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv(Judge.API_KEY_ENV, "k")


def judge(server: Server) -> Judge:
    return Judge("typesafe/jev-1.14", transport=httpx.MockTransport(server))


def bash(command: str, description: str = "Run the tests") -> ToolCallRecord:
    return ToolCallRecord(
        id="c1", tool_name="bash", parameters={"command": command, "description": description}
    )


def context() -> ToolContext:
    ctx = RunContext(user_id="u", session_id="s")
    ctx.messages.append(UserMessage(source="user", content="run the tests"))
    ctx.messages.append(UserMessage(source=HARNESS_SOURCE, content="[harness] a note"))
    return ToolContext(ctx)


async def ask(hook: HarnessToolHook, record: ToolCallRecord, tool=None):
    return await hook.permission(tool or BashTool(), record, context())


async def test_a_command_the_judge_is_sure_about_runs_without_asking():
    server = Server(0.97)
    record = bash("pytest -q")
    assert await ask(HarnessToolHook(judge=judge(server)), record) is True
    assert record.approval_reason == "judge: safe (0.97)"
    # It got the user's own words, the model's claim and the exact command.
    sent = server.calls[0]
    assert sent["model"] == "typesafe/jev-1.14"
    assert "run the tests" in sent["state"] and "[harness]" not in sent["state"]
    assert "(its claim):\nRun the tests" in sent["state"]
    assert sent["state"].endswith("The command:\npytest -q")


async def test_a_command_the_judge_is_not_sure_about_asks_and_says_so():
    record = bash("pytest -q")
    assert await ask(HarnessToolHook(judge=judge(Server(0.6))), record) is None
    assert record.approval_reason == "judge: not sure (0.60)"
    assert MaxAIApp._approval_card(record).endswith("$ pytest -q\n\njudge: not sure (0.60)")


@pytest.mark.parametrize("server", [Server(None), Server(0.97, status=500)])
async def test_a_judge_that_fails_asks_the_user(server):
    record = bash("pytest -q")
    assert await ask(HarnessToolHook(judge=judge(server)), record) is None
    assert record.approval_reason == "judge: no answer"


async def test_an_answer_of_another_shape_is_no_verdict():
    assert Judge.read({"answers": {}}) is None
    assert Judge.read({"answers": {"verdict": {"choice": "ask", "probabilities": {"safe": 0.9}}}}) is None
    assert Judge.read({"answers": {"verdict": {"choice": "safe", "probabilities": {"safe": 7}}}}) is None


@pytest.mark.parametrize(
    ("rules", "command", "expected"),
    [
        ({"deny": ["bash(pytest*)"]}, "pytest -q", False),
        ({"ask": ["bash(pytest*)"]}, "pytest -q", None),
        ({}, "timeout 5 rm -rf workspace", None),  # dangerous: always asked
        ({}, "cat .env", None),  # a file of keys: always asked, however safe it looks
        ({"allow": ["bash(pytest -q)"]}, "pytest -q", True),
    ],
)
async def test_the_judge_is_not_called_when_a_rule_decides(rules, command, expected):
    server = Server(0.99)
    hook = HarnessToolHook(permissions=Permissions(rules), judge=judge(server))
    assert await ask(hook, bash(command)) is expected
    assert server.calls == []


@pytest.mark.parametrize(
    "command",
    ["python3 clean.py", "bash build.sh", "timeout 5 pytest -q", "node -e 'x()'",
     "for f in *.py; do wc -l $f; done"],
)
async def test_code_the_judge_cannot_see_is_asked_without_calling_it(command):
    # A script the model wrote: the command says nothing of what it does.
    server = Server(0.99)
    record = bash(command)
    assert await ask(HarnessToolHook(judge=judge(server)), record) is None
    assert server.calls == [] and record.approval_reason is None


async def test_the_judge_only_sees_bash():
    server = Server(0.99)
    record = ToolCallRecord(id="c1", tool_name="write_file", parameters={"path": "a", "content": "x"})
    tool = WriteFileTool()
    tool.approval_mode = ToolApprovalMode.ASK_APPROVED  # a tool that asks, and is not bash
    assert await ask(HarnessToolHook(judge=judge(server)), record, tool) is None
    assert server.calls == []


async def test_without_a_key_or_with_a_command_too_long_nothing_is_sent(monkeypatch):
    server = Server(0.99)
    hook = HarnessToolHook(judge=judge(server))
    # Too long to send whole: cutting it could hide what it does.
    assert await ask(hook, bash("echo " + "a" * Judge.MAX_CHARS)) is None
    monkeypatch.delenv(Judge.API_KEY_ENV)
    assert await ask(hook, bash("pytest -q")) is None
    assert server.calls == []


async def test_there_is_no_judge_until_the_settings_name_a_model(monkeypatch):
    from max_ai.config import setting

    # Whatever the .env of this machine says, the test starts without a model.
    monkeypatch.setattr(setting, "harness_judge_model", "")
    assert HarnessToolHook().judge is None
    monkeypatch.setattr(setting, "harness_judge_model", "typesafe/jev-1.14")
    monkeypatch.setattr(setting, "harness_judge_min_safe", 0.8)
    made = HarnessToolHook().judge
    assert made is not None and made.model == "typesafe/jev-1.14" and made.min_safe == 0.8
