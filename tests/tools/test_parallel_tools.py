"""Read-only tool calls of one model reply run at the same time; everything
else runs alone and in order."""

from __future__ import annotations

import asyncio
import time

import pytest

from max_ai.agents import Agent
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.config import setting
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode

DELAY = 0.3
TIMELINE: list[tuple[str, float, float]] = []


async def lookup(city: str) -> dict:
    """Read-only lookup."""
    start = time.monotonic()
    await asyncio.sleep(DELAY)
    TIMELINE.append((f"lookup:{city}", start, time.monotonic()))
    return {"city": city}


async def save(city: str) -> dict:
    """Writes something."""
    start = time.monotonic()
    await asyncio.sleep(DELAY)
    TIMELINE.append((f"save:{city}", start, time.monotonic()))
    return {"saved": city}


class OneRound:
    """Asks for the given calls in one reply, then answers."""

    model = "fake"
    config = ModelConfig()
    generation_options = {"max_tokens": 100}

    def __init__(self, *calls: tuple[str, str]):
        self.calls = calls
        self.done = False

    async def run(self, **kwargs):
        if self.done:
            message = AssistantMessage(source="llm", content="listo")
        else:
            self.done = True
            message = AssistantMessage(source="llm", content="", tool_calls=[
                ToolCall(id=f"{name}-{city}", tool_name=name, parameters={"city": city})
                for name, city in self.calls])
        return ChatCompletionResult(message=message, usage=Usage(), model="fake", finish_reason="stop")


@pytest.fixture(autouse=True)
def clear():
    TIMELINE.clear()


async def run(tmp_path, llm, *extra_tools):
    tools = [FunctionAsTool(lookup, approval_mode=ToolApprovalMode.AUTO_APPROVED, read_only=True),
             FunctionAsTool(save, approval_mode=ToolApprovalMode.AUTO_APPROVED), *extra_tools]
    async with Agent(name="a", description="d", instructions="i", client=llm, toolset=tools,
                     workspace=LocalWorkspace(root=tmp_path)) as agent:
        started = time.monotonic()
        response = await agent.run("go", run_context=RunContext(user_id="u"))
        return response, time.monotonic() - started


def overlaps(a: str, b: str) -> bool:
    (_, s1, e1), (_, s2, e2) = (next(t for t in TIMELINE if t[0] == name) for name in (a, b))
    return s1 < e2 and s2 < e1


async def test_read_only_calls_run_at_the_same_time_and_keep_their_order(tmp_path):
    cities = ["Lima", "Quito", "Cusco", "Bogota"]
    response, elapsed = await run(tmp_path, OneRound(*[("lookup", c) for c in cities]))
    assert elapsed < 2 * DELAY, f"{elapsed:.2f}s: they did not run in parallel"
    results = [m.tool_call_id for m in response.context.messages if isinstance(m, ToolMessage)]
    assert results == [f"lookup-{c}" for c in cities]  # transcript in call order


async def test_a_write_splits_the_reads_around_it(tmp_path):
    await run(tmp_path, OneRound(("lookup", "A"), ("lookup", "B"), ("save", "A"), ("lookup", "C")))
    assert overlaps("lookup:A", "lookup:B")
    assert not overlaps("lookup:B", "save:A") and not overlaps("save:A", "lookup:C")
    order = [name for name, *_ in sorted(TIMELINE, key=lambda t: t[1])]
    assert order.index("save:A") < order.index("lookup:C")


async def test_parallelism_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(setting, "max_parallel_tools", 2)
    _, elapsed = await run(tmp_path, OneRound(*[("lookup", c) for c in "ABCD"]))
    assert 2 * DELAY <= elapsed < 3.5 * DELAY  # two waves of two


async def test_read_only_mcp_tools_run_in_parallel_on_one_connection(tmp_path, monkeypatch):
    from mcp import Client
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations

    from max_ai.capabilities.mcp import StdioMCPServerConfig
    from max_ai.capabilities.mcp import client_manager as module

    server = MCPServer("web")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def search(city: str) -> str:
        await asyncio.sleep(DELAY)
        return f"found {city}"

    monkeypatch.setattr(module, "create_mcp_client", lambda config: Client(server))
    llm = OneRound(*[("web_search", c) for c in ("Lima", "Quito", "Cusco")])
    async with Agent(name="a", description="d", instructions="i", client=llm,
                     workspace=LocalWorkspace(root=tmp_path),
                     mcp=[StdioMCPServerConfig(server_id="web", command="unused")]) as agent:
        started = time.monotonic()
        response = await agent.run("go", run_context=RunContext(user_id="u"))
        elapsed = time.monotonic() - started
        assert agent._registry.get("web_search").read_only is True
    assert elapsed < 2 * DELAY, f"{elapsed:.2f}s: MCP calls did not overlap"
    assert [m.content for m in response.context.messages if isinstance(m, ToolMessage)] == [
        '"found Lima"', '"found Quito"', '"found Cusco"']


def test_built_in_read_tools_are_read_only_and_writes_are_not(tmp_path):
    agent = Agent(name="a", description="d", instructions="i", client=OneRound(),
                  workspace=LocalWorkspace(root=tmp_path))
    read_only = {tool.name for tool in agent.tools if tool.read_only}
    assert read_only == {"list_directory", "find_files", "search_text", "read_file", "file_info"}


async def test_tool_results_carry_their_real_start_and_end(tmp_path):
    response, _ = await run(tmp_path, OneRound(("lookup", "Lima")))
    result = response.context.tool_state.records["lookup-Lima"].result
    assert DELAY * 1000 <= result.duration_ms < DELAY * 1000 + 200
