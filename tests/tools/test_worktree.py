import subprocess

from max_ai.base.tools import ToolContext
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.worktree import EnterWorktreeTool, ExitWorktreeTool
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord


def call(tool, **parameters):
    return ToolCallRecord(id="c1", tool_name=tool, parameters=parameters)


async def session(tmp_path, repository=True):
    box = LocalSandbox(str(tmp_path)).bind("u")
    await box.prepare()
    ctx = RunContext(user_id="u", session_id="s")
    workspace = box.session_paths("s")["workspace"]
    subprocess.run(["mkdir", "-p", workspace], check=True)
    if repository:
        git = ["git", "-C", workspace, "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([*git, "init", "-q"], check=True)
        subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "start"], check=True)
    return ToolContext(ctx, box), ctx, workspace


async def test_enter_moves_the_agent_to_a_new_branch_and_exit_brings_it_back(tmp_path):
    tool_ctx, ctx, workspace = await session(tmp_path)
    entered = await EnterWorktreeTool().execute(call("enter_worktree", name="fix-login"), tool_ctx)
    target = f"{workspace}/.worktrees/fix-login"
    assert entered.success and ctx.cwd == target
    branch = subprocess.run(
        ["git", "-C", target, "branch", "--show-current"], capture_output=True, text=True
    )
    assert branch.stdout.strip() == "fix-login"
    # Keep: the agent is back and the worktree stays.
    kept = await ExitWorktreeTool().execute(call("exit_worktree", action="keep"), tool_ctx)
    assert kept.success and ctx.cwd is None
    assert subprocess.run(["test", "-d", target]).returncode == 0


async def test_remove_deletes_a_clean_worktree_and_refuses_one_with_changes(tmp_path):
    tool_ctx, ctx, workspace = await session(tmp_path)
    await EnterWorktreeTool().execute(call("enter_worktree", name="draft"), tool_ctx)
    target = f"{workspace}/.worktrees/draft"
    subprocess.run(["touch", f"{target}/new.txt"], check=True)
    # Changes not committed: git refuses and the agent stays where it is.
    refused = await ExitWorktreeTool().execute(call("exit_worktree", action="remove"), tool_ctx)
    assert not refused.success and ctx.cwd == target
    subprocess.run(["rm", f"{target}/new.txt"], check=True)
    # From a folder inside the worktree it is still the whole worktree that goes.
    subprocess.run(["mkdir", f"{target}/src"], check=True)
    ctx.cwd = f"{target}/src"
    removed = await ExitWorktreeTool().execute(call("exit_worktree", action="remove"), tool_ctx)
    assert removed.success and ctx.cwd is None
    assert subprocess.run(["test", "-d", target]).returncode != 0


async def test_a_folder_that_is_not_a_repository_and_exit_outside_a_worktree_say_why(tmp_path):
    tool_ctx, ctx, _ = await session(tmp_path, repository=False)
    entered = await EnterWorktreeTool().execute(call("enter_worktree", name="x"), tool_ctx)
    assert not entered.success and "not a git repository" in entered.error and ctx.cwd is None
    left = await ExitWorktreeTool().execute(call("exit_worktree", action="keep"), tool_ctx)
    assert not left.success and "not in a worktree" in left.error
