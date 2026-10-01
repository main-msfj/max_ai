"""Policy: one allow / ask / deny decision for every tool call."""

import pytest

from max_ai.capabilities.tools.bash import BashTool
from max_ai.capabilities.tools.file_system import FileSystemTools
from max_ai.core.policy import DEFAULT_DENY, Policy
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
    assert verdict(policy, tools["ReadFile"], file_path="a.md") == ("allow", None)
    assert verdict(policy, tools["WriteFile"], file_path="a.md", content="x") == ("ask", None)


def test_rules_by_name_glob_and_path(tools):
    policy = Policy(
        allow=["WriteFile(docs/**)", "Bash(cat:*)"],
        ask=["read*"],
        deny=[*DEFAULT_DENY, "WriteFile(secrets/**)", "DeleteFile"],
    )
    write = tools["WriteFile"]
    assert verdict(policy, write, file_path="docs/a.md", content="x") == ("allow", "WriteFile(docs/**)")
    # The path is matched as the user writes it, whatever form the model sent.
    assert verdict(policy, write, file_path="workspace/secrets/k.txt", content="x")[0] == "deny"
    assert verdict(policy, write, file_path="./secrets/k.txt", content="x")[0] == "deny"
    assert verdict(policy, tools["DeleteFile"], file_path="a.md") == ("deny", "DeleteFile")
    assert verdict(policy, tools["ReadFile"], file_path="a.md") == ("ask", "read*")  # case-insensitive glob
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


def test_always_allow_rules_for_a_call(tools):
    policy = Policy()
    assert policy.rules_for(tools["bash"], {"command": "npm i && npm test"}) == ["bash(npm:*)"]
    assert policy.rules_for(tools["bash"], {"command": "cat $(ls)"}) == []  # can't be allowed again
    assert policy.rules_for(tools["WriteFile"], {"file_path": "a.md", "content": "x"}) == ["WriteFile"]


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
