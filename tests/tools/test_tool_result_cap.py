"""One cap for every tool result the model sees, whatever tool made it."""

from types import SimpleNamespace

from max_ai.agents.agent import Agent
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.config import setting
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage
from max_ai.core.tool.dispatcher import ToolDispatcher
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode

BIG = "START" + "x" * 1_000_000 + "END"  # like a large MCP or API answer


def export_report() -> str:
    """Return the full report."""
    return BIG


def broken_export() -> str:
    """Fail with a huge message."""
    raise ValueError(BIG)


class OneCallClient:
    """Calls ``tool`` once, then answers; keeps what each call was shown."""

    model = "fake"
    config = SimpleNamespace(tokenizer_base="o200k_base")

    def __init__(self, tool: str):
        self.tool = tool
        self.seen = []

    async def run(self, ctx, prompts, tools=None, **kwargs):
        self.seen.append(list(ctx.messages))
        calls = [ToolCall(tool_name=self.tool, parameters={})] if len(self.seen) == 1 else []
        message = AssistantMessage(source="fake", content="" if calls else "done", tool_calls=calls)
        return ChatCompletionResult(message=message, usage=Usage(), model=self.model,
                                    finish_reason="tool_calls" if calls else "stop")


async def run_once(tmp_path, function):
    client = OneCallClient(function.__name__)
    agent = Agent(name="t", description="d", instructions="i", client=client,
                  toolset=[FunctionAsTool(function, approval_mode=ToolApprovalMode.AUTO_APPROVED)],
                  workspace=LocalWorkspace(root=tmp_path))
    ctx = RunContext(user_id="u", session_id="s")
    async with agent:
        await agent.run("go", run_context=ctx)
    shown = [m for m in client.seen[1] if isinstance(m, ToolMessage)][-1]
    return shown, ctx


def test_short_text_is_untouched():
    assert ToolDispatcher._capped("small result") == "small result"


def test_long_text_keeps_both_ends_and_says_what_was_cut():
    limit = setting.max_tool_result_chars
    capped = ToolDispatcher._capped(BIG)

    assert len(capped) < limit + 200
    assert capped.startswith("START") and capped.endswith("END")
    assert f"of {len(BIG):,} characters cut" in capped


async def test_the_model_sees_a_capped_result_and_the_record_keeps_it_all(tmp_path):
    shown, ctx = await run_once(tmp_path, export_report)

    assert len(shown.text()) < setting.max_tool_result_chars + 200
    assert "characters cut" in shown.text()
    [record] = ctx.tool_state.records.values()
    assert record.result.result == BIG


async def test_a_huge_error_is_capped_too(tmp_path):
    shown, _ = await run_once(tmp_path, broken_export)

    assert not shown.success
    assert len(shown.text()) < setting.max_tool_result_chars + 200
    assert "characters cut" in shown.text()
