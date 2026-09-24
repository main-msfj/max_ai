"""One Agent serves many users at once: runs overlap and never share state."""

from __future__ import annotations

import asyncio
import json
import time

from max_ai.agents import Agent
from max_ai.base.tools import ToolContext
from max_ai.capabilities.memory import LocalMemoryRegistry
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.event_type import ToolCallResponseEvent
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.types.agent_response import AgentResponse
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode

DELAY = 0.3  # seconds per model call


def whoami(context: ToolContext | None) -> dict:
    """Who is running this tool."""
    return {"user": context.user_id, "session": context.session_id,
            "workspace": context.deps["workspace_dir"]}


class SlowLLM:
    """Asks for whoami, then answers with what the tool said. Records the
    system prompt of each call to check memories don't cross users."""

    model = "slow"
    config = ModelConfig()
    generation_options = {"max_tokens": 200}

    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []
        self.prompts: dict[str, list[str]] = {}

    async def run(self, *, ctx, prompts, **kwargs):
        start = time.monotonic()
        await asyncio.sleep(DELAY)
        self.calls.append((start, time.monotonic()))
        user = ctx.user_id
        self.prompts.setdefault(user, []).append("\n".join(prompts.rendered_layers.values()))
        last = ctx.messages[-1]
        if isinstance(last, ToolMessage):
            message = AssistantMessage(source="llm", content=f"tool said {last.content}")
        else:
            message = AssistantMessage(source="llm", content="", tool_calls=[
                ToolCall(id=f"call-{user}-{len(ctx.messages)}", tool_name="whoami", parameters={})])
        return ChatCompletionResult(message=message, usage=Usage(), model="slow", finish_reason="stop")


def make_agent(tmp_path, llm) -> Agent:
    memory = LocalMemoryRegistry(base_path=tmp_path / "memory")
    return Agent(name="shared", description="d", instructions="i", client=llm,
                 toolset=[FunctionAsTool(whoami, approval_mode=ToolApprovalMode.AUTO_APPROVED)],
                 workspace=LocalWorkspace(root=tmp_path / "work"), memory=memory)


async def ask(agent: Agent, user: str) -> tuple[AgentResponse, list]:
    events, response = [], None
    ctx = RunContext(user_id=user, session_id=f"s-{user}")
    async for item in agent.run_stream_events("¿quién soy?", run_context=ctx):
        if isinstance(item, AgentResponse):
            response = item
        else:
            events.append(item)
    return response, events


async def test_ten_users_run_at_the_same_time_without_mixing(tmp_path):
    llm = SlowLLM()
    users = [f"user{i}" for i in range(10)]
    # Each user has a memory only they should ever see in their prompt.
    memory = LocalMemoryRegistry(base_path=tmp_path / "memory")
    for user in users:
        await memory.bind(user, f"s-{user}").create_or_update("secret", f"clave-de-{user}")

    async with make_agent(tmp_path, llm) as agent:
        started = time.monotonic()
        results = await asyncio.gather(*(ask(agent, user) for user in users))
        elapsed = time.monotonic() - started

    # Two model calls per run: in parallel ~0.6s, one after another ~6s.
    assert elapsed < 4 * DELAY, f"runs did not overlap: {elapsed:.2f}s"

    for user, (response, events) in zip(users, results):
        said = json.loads(response.final_text.removeprefix("tool said "))
        assert (said["user"], said["session"]) == (user, f"s-{user}")
        assert f"/{user}/" in said["workspace"]
        # This stream only carries this user's tool results.
        tool_events = [e for e in events if isinstance(e, ToolCallResponseEvent)]
        assert tool_events and all(e.tool_result.result["user"] == user for e in tool_events)
        # And the prompt only ever held this user's memory.
        for prompt in llm.prompts[user]:
            assert f"clave-de-{user}" in prompt
            assert not any(f"clave-de-{other}" in prompt for other in users if other != user)


async def test_the_same_session_runs_one_message_at_a_time(tmp_path):
    llm = SlowLLM()
    ctx = RunContext(user_id="ana", session_id="s-ana")
    async with make_agent(tmp_path, llm) as agent:
        first = asyncio.create_task(agent.run("uno", run_context=ctx))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(agent.run("dos", run_context=ctx))
        await asyncio.gather(first, second)
    starts = sorted(start for start, _ in llm.calls)
    ends = sorted(end for _, end in llm.calls)
    # 4 calls in total and never two at once: each starts after the previous ends.
    assert len(llm.calls) == 4
    assert all(starts[i + 1] >= ends[i] - 0.01 for i in range(3))


async def test_the_shared_loop_is_never_modified_by_a_run(tmp_path):
    llm = SlowLLM()
    async with make_agent(tmp_path, llm) as agent:
        before = dict(vars(agent.reasoning))
        await asyncio.gather(ask(agent, "ana"), ask(agent, "beto"))
        assert vars(agent.reasoning) == before


async def test_mcp_connects_once_for_every_concurrent_run(tmp_path, monkeypatch):
    from mcp import Client
    from mcp.server import MCPServer

    from max_ai.capabilities.mcp import StdioMCPServerConfig
    from max_ai.capabilities.mcp import client_manager as module

    server = MCPServer("docs")

    @server.tool()
    async def search(q: str) -> str:
        await asyncio.sleep(0.05)
        return f"found {q}"

    connections = []

    def connect(config):
        connections.append(config.server_id)
        return Client(server)

    monkeypatch.setattr(module, "create_mcp_client", connect)

    class McpLLM(SlowLLM):
        async def run(self, *, ctx, prompts, **kwargs):
            await asyncio.sleep(DELAY)
            last = ctx.messages[-1]
            if isinstance(last, ToolMessage):
                message = AssistantMessage(source="llm", content=last.content)
            else:
                message = AssistantMessage(source="llm", content="", tool_calls=[
                    ToolCall(id=f"q-{ctx.user_id}", tool_name="docs_search",
                             parameters={"q": ctx.user_id})])
            return ChatCompletionResult(message=message, usage=Usage(), model="slow", finish_reason="stop")

    agent = Agent(name="shared", description="d", instructions="i", client=McpLLM(),
                  workspace=LocalWorkspace(root=tmp_path),
                  mcp=[StdioMCPServerConfig(server_id="docs", command="unused",
                                            approval_mode=ToolApprovalMode.AUTO_APPROVED)])
    async with agent:
        users = [f"user{i}" for i in range(8)]
        responses = await asyncio.gather(*(ask(agent, user) for user in users))
        assert [json.loads(r.final_text) for r, _ in responses] == [f"found {u}" for u in users]
        assert connections == ["docs"]  # one connection for all eight runs
    assert all(s.worker is None for s in agent._mcp_manager._servers.values())
