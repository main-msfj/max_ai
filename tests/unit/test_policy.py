"""Policy: one allow / ask / deny decision for every tool call."""

from types import SimpleNamespace

import pytest

from max_ai.agents.agent import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.tools.bash import BashTool
from max_ai.capabilities.tools.file_system import FileSystemTools
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage
from max_ai.core.policy import DEFAULT_DENY, Policy
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode


@pytest.fixture
def tools(tmp_path):
    files = {tool.name: tool for tool in FileSystemTools(tmp_path).tools}
    return {"bash": BashTool(), **files}


def verdict(policy: Policy, tool, isolated: bool = True, extra=(), **parameters) -> tuple[str, str | None]:
    return tuple(policy.decide(tool, parameters, isolated=isolated, extra_allow=extra))


def test_the_default_policy(tools):
    policy, bash = Policy(), tools["bash"]
    assert verdict(policy, bash, command="ls && sudo rm x") == ("deny", "Bash(sudo:*)")
    assert verdict(policy, bash, command="rm -rf /") == ("deny", "Bash(rm -rf /*)")
    assert verdict(policy, bash, command="git status") == ("allow", "Bash(git status)")
    assert verdict(policy, bash, command="git push origin main") == ("ask", "Bash(git push:*)")
    assert verdict(policy, bash, command="python make.py") == ("ask", None)  # bash's default
    assert verdict(policy, tools["read_file"], file_name="a.md") == ("allow", None)
    assert verdict(policy, tools["write_file"], file_name="a.md", content="x") == ("ask", None)


def test_deny_sees_through_paths_wrappers_and_inline_scripts(tools):
    policy, bash = Policy(), tools["bash"]
    for command in ("/usr/bin/sudo ls", "env sudo ls", "env -i FOO=1 sudo ls",
                    "FOO=1 sudo ls", "nohup nice -n 5 sudo ls", "timeout -s KILL 5 sudo ls",
                    "xargs -I {} sudo rm {}", "command -p su", "eval sudo ls",
                    'bash -c "sudo rm x"', "sh -lc 'echo hi; sudo reboot'",
                    "bash -c 'bash -c \"sudo x\"'", "env git push --force origin"):
        assert verdict(policy, bash, command=command)[0] == "deny", command
    # Allow rules stay literal: a path-qualified or wrapped command asks.
    for command in ("/usr/bin/git status", "./git status", "env pwd", "echo sudo", "sudoku",
                    "env git push origin"):
        assert verdict(policy, bash, command=command)[0] == "ask", command
    # ...and can't be allowed again from the "always allow" prompt.
    assert policy.rules_for(bash, {"command": "env pwd"}) == []


def test_rules_by_name_glob_and_path(tools):
    policy = Policy(
        allow=["write_file(docs/**)", "Bash(cat:*)"],
        ask=["READ*"],
        deny=[*DEFAULT_DENY, "write_file(secrets/**)", "delete_file"],
    )
    write = tools["write_file"]
    assert verdict(policy, write, file_name="docs/a.md", content="x") == ("allow", "write_file(docs/**)")
    # The path is matched as the user writes it, whatever form the model sent.
    assert verdict(policy, write, file_name="workspace/secrets/k.txt", content="x")[0] == "deny"
    assert verdict(policy, write, file_name="./secrets/k.txt", content="x")[0] == "deny"
    assert verdict(policy, tools["delete_file"], file_name="a.md") == ("deny", "delete_file")
    assert verdict(policy, tools["read_file"], file_name="a.md") == ("ask", "READ*")  # case-insensitive glob
    # Every command in the line must be allowed, and bash must see all of it.
    assert verdict(policy, tools["bash"], command="cat a.txt")[0] == "allow"
    assert verdict(policy, tools["bash"], command="cat a.txt && rm a.txt")[0] == "ask"
    assert verdict(policy, tools["bash"], command="cat $(ls)")[0] == "ask"


def test_without_a_sandbox_only_the_user_allows_commands(tools):
    policy, bash = Policy(), tools["bash"]
    assert verdict(policy, bash, isolated=False, command="git status") == (
        "ask", "commands on this machine always ask")
    # "Always allow" from the user counts even on their own machine.
    assert verdict(policy, bash, isolated=False, extra=["bash(npm:*)"], command="npm test") == (
        "allow", "bash(npm:*)")
    assert verdict(policy, bash, isolated=False, extra=["bash(npm:*)"],
                   command="git status && npm test")[0] == "allow"
    # The user's rules never beat ask or deny.
    assert verdict(policy, bash, isolated=False, extra=["bash(git:*)"],
                   command="git push origin")[0] == "ask"


def test_always_allow_rules_for_a_call(tools):
    policy = Policy()
    assert policy.rules_for(tools["bash"], {"command": "npm i && npm test"}) == ["bash(npm:*)"]
    assert policy.rules_for(tools["bash"], {"command": "cat $(ls)"}) == []  # can't be allowed again
    assert policy.rules_for(tools["write_file"], {"file_name": "a.md", "content": "x"}) == ["write_file"]


def test_a_new_tool_needs_nothing_and_can_name_its_subject():
    from max_ai.capabilities.tools import tool

    @tool(approval_mode=ToolApprovalMode.ASK_APPROVED, policy_subject="url")
    def fetch(url: str) -> str:
        return url

    policy = Policy(allow=["fetch(https://api.acme.com/*)"], deny=["fetch(*internal*)"])
    assert verdict(policy, fetch, url="https://api.acme.com/v1")[0] == "allow"
    assert verdict(policy, fetch, url="https://internal.acme.com")[0] == "deny"
    assert verdict(policy, fetch, url="https://example.com")[0] == "ask"  # the tool's default


def test_bad_rules_are_refused():
    with pytest.raises(ValueError, match="Not a rule"):
        Policy(deny=["Bash(sudo"])


class BashOnceClient:
    """Runs ``command`` with bash once, then answers."""

    model = "fake"
    config = SimpleNamespace(tokenizer_base="o200k_base")

    def __init__(self, command: str):
        self.command = command
        self.calls = 0

    async def run(self, ctx, prompts, tools=None, **kwargs):
        self.calls += 1
        calls = [ToolCall(tool_name="bash", parameters={"command": self.command,
                                                        "description": "x"})]
        calls = calls if self.calls == 1 else []
        message = AssistantMessage(source="fake", content="" if calls else "done", tool_calls=calls)
        return ChatCompletionResult(message=message, usage=Usage(), model=self.model,
                                    finish_reason="tool_calls" if calls else "stop")


async def test_the_agent_never_runs_a_denied_command(tmp_path):
    marker = tmp_path / "ran"
    agent = Agent(name="t", description="d", instructions="i",
                  client=BashOnceClient(f"pwd && env sudo touch {marker}"),
                  workspace=LocalWorkspace(root=tmp_path / "ws"))
    ctx = RunContext(user_id="u", session_id="s")
    async with agent:
        response = await agent.run("go", run_context=ctx)

    assert response.final_text == "done"
    [answer] = [m for m in ctx.messages if isinstance(m, ToolMessage)]
    assert not answer.success and "Denied by policy: Bash(sudo:*)" in answer.text()
    assert not marker.exists()


def test_the_policy_is_saved_with_the_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    policy = Policy(allow=["Bash(npm test)"], deny=[*DEFAULT_DENY, "delete_file"])
    agent = Agent(name="t", description="d", instructions="i",
                  client=OpenAIChatCompletionClient(model="m", api_key="sk-test"),
                  workspace=LocalWorkspace(root=tmp_path), policy=policy)
    assert Agent.deserialize(agent.serialize()).policy == policy
