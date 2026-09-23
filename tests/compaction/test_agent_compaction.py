"""Compaction wired into a real Agent + ReactLoop, end to end."""

from __future__ import annotations

import tempfile
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.compaction import SlidingWindowCompaction, SummaryCompaction
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.compaction import CompactionOutput
from max_ai.core.event_type import CompactionEvent
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.core.model.llm import ModelConfig
from max_ai.types.agent_response import AgentResponse
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext

PLAN = {"rationale": "demo", "steps": [
    {"id": 1, "description": "Definir la tarea", "status": "done"},
    {"id": 2, "description": "Ejecutar el avance", "status": "active"},
]}


class FakeLLM:
    """Answers every turn, plans on the first one, summarizes when asked."""

    model = "fake"

    def __init__(self, window: int):
        self.config = ModelConfig(max_context_window=window, supports_function_calling=True)
        self.generation_options = {"max_tokens": 500}
        self.system_prompts: list[str] = []
        self.summary_calls = 0

    async def run(self, *, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        if output_format is CompactionOutput:
            self.summary_calls += 1
            out = CompactionOutput(summary=f"resumen #{self.summary_calls}", objective=["charlar"])
            message = AssistantMessage(source="llm", content=out.summary, structured_output=out)
            return ChatCompletionResult(message=message, usage=Usage(), model="fake", finish_reason="stop")

        self.system_prompts.append("\n".join(prompts.rendered_layers.values()))
        planned = any(m.role == "tool" for m in ctx.messages) or len(self.system_prompts) > 1
        if not planned:
            message = AssistantMessage(source="llm", content="", tool_calls=[
                ToolCall(id="plan1", tool_name="update_plan", parameters=PLAN)])
        else:
            message = AssistantMessage(source="llm", content="respuesta " + "bla " * 150)
        return ChatCompletionResult(
            message=message, usage=Usage(tokens_input=100, tokens_output=10),
            model="fake", finish_reason="stop",
        )


async def talk(agent: Agent, ctx: RunContext, turns: int):
    events = []
    for n in range(turns):
        async for item in agent.run_stream_events(f"pregunta {n} " + "palabra " * 300, run_context=ctx):
            if isinstance(item, CompactionEvent):
                events.append(item)
            elif isinstance(item, AgentResponse):
                ctx = item.context
    return ctx, events


async def test_summary_compaction_end_to_end():
    root = Path(tempfile.mkdtemp())
    probe = FakeLLM(window=100_000)
    async with Agent(name="demo", description="d", instructions="i", client=probe,
                     workspace=LocalWorkspace(root=root)) as agent:
        prompt_tokens = (await agent._prompts(RunContext())).prompt_tokens
    assert prompt_tokens > 0  # the real prompt is measured now

    llm = FakeLLM(window=prompt_tokens + 5_000)  # room for ~4 turns
    async with Agent(name="demo", description="d", instructions="i", client=llm,
                     workspace=LocalWorkspace(root=root),
                     compaction=SummaryCompaction(summary_max_tokens=300)) as agent:
        ctx, events = await talk(agent, RunContext(user_id="u", session_id="s"), turns=12)

    ends = [e for e in events if e.phase == "end" and e.changed and not e.pruned_only]
    assert ends, "the window never compacted"
    first = ends[0]
    assert first.tokens_after < first.tokens_before and first.old_messages
    assert first.summary.startswith("<conversation_summary>")

    # State applied to ctx and travels with the session.
    assert ctx.compaction.compactions >= 1 and "summary" in ctx.compaction.state
    restored = RunContext.model_validate_json(ctx.model_dump_json())
    assert restored.compaction.state == ctx.compaction.state

    # The model sees the summary and the plan in its system prompt afterwards,
    # even though the update_plan call left the transcript.
    last_prompt = llm.system_prompts[-1]
    assert "<conversation_summary>" in last_prompt and "resumen #" in last_prompt
    assert "<current_plan>" in last_prompt and "[active] 2. Ejecutar el avance" in last_prompt
    assert not any(m.role == "tool" for m in ctx.messages)

    # Never an invalid window, and it stays bounded.
    SummaryCompaction()._validate(ctx.messages)
    assert len(ctx.messages) < 24


async def test_sliding_window_end_to_end_without_llm_summaries():
    root = Path(tempfile.mkdtemp())
    llm = FakeLLM(window=0)  # unknown window: only the turn count applies
    async with Agent(name="demo", description="d", instructions="i", client=llm,
                     workspace=LocalWorkspace(root=root),
                     compaction=SlidingWindowCompaction(max_turns=3)) as agent:
        ctx, events = await talk(agent, RunContext(user_id="u", session_id="s"), turns=6)

    users = [m.text().split()[1] for m in ctx.messages if m.role == "user"]
    assert users == ["3", "4", "5"]
    assert llm.summary_calls == 0 and ctx.compaction.state == {}
    assert "<conversation_summary>" not in llm.system_prompts[-1]


async def test_no_compaction_configured_changes_nothing():
    root = Path(tempfile.mkdtemp())
    llm = FakeLLM(window=4_000)
    async with Agent(name="demo", description="d", instructions="i", client=llm,
                     workspace=LocalWorkspace(root=root)) as agent:
        ctx, events = await talk(agent, RunContext(user_id="u", session_id="s"), turns=4)
    assert events == [] and ctx.compaction.compactions == 0
    assert len([m for m in ctx.messages if m.role == "user"]) == 4
