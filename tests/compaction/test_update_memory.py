"""CoreCompaction._update_memory: durable facts leaving the window go to memory."""

from __future__ import annotations

from types import SimpleNamespace

from max_ai.capabilities.compaction import SummaryCompaction
from max_ai.capabilities.memory import LocalMemoryRegistry
from max_ai.core.compaction import (
    CompactionOutput,
    MemoryFactUpdate,
    MemoryMaintenanceOutput,
)
from max_ai.core.messages import AssistantMessage, UserMessage
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx

WINDOW = 10_000


class FakeLLM:
    """Summarizes, and answers memory requests with ``memory_output``."""

    def __init__(self, memory_output=None, prose: str | None = None):
        self.memory_output = memory_output
        self.prose = prose
        self.memory_tasks: list[str] = []
        self.generation_options = {"max_tokens": 1_000}
        self.config = SimpleNamespace(max_context_window=0)

    async def run(self, *, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        if output_format is MemoryMaintenanceOutput:
            self.memory_tasks.append(ctx.messages[0].text())
            output = self.memory_output
        else:
            output = CompactionOutput(summary="resumen")
        content = self.prose if (self.prose and output_format is MemoryMaintenanceOutput) else ""
        message = AssistantMessage(
            source="llm", content=content, structured_output=None if content else output,
        )
        return ChatCompletionResult(message=message, usage=Usage(), model="fake", finish_reason="stop")


def prompts() -> PromptCtx:
    return PromptCtx.model_construct(
        stack=None, variables={}, rendered_layers={}, layer_usage={}, prompt_tokens=1_000,
    )


def chat() -> list:
    first = [UserMessage(source="marvin", content="mi perro se llama Toby", token_count=500),
             AssistantMessage(source="a", content="anotado", token_count=500)]
    rest = [m for n in range(7) for m in (
        UserMessage(source="marvin", content=f"pregunta {n}", token_count=500),
        AssistantMessage(source="a", content=f"respuesta {n}", token_count=500),
    )]
    return first + rest


async def compact(strategy, client, memory):
    return await strategy.compact(
        ctx=RunContext(user_id="u", session_id="s", messages=chat()),
        prompts=prompts(), max_context_tokens=WINDOW, client=client, memory=memory,
    )


async def test_durable_facts_are_merged_into_memory(tmp_path):
    memory = LocalMemoryRegistry(user_id="u", session_id="s", base_path=tmp_path)
    await memory.create_or_update("mascotas", "Tiene un gato llamado Michi.")
    client = FakeLLM(MemoryMaintenanceOutput(updates=[MemoryFactUpdate(
        category="mascotas", content="Tiene un gato llamado Michi y un perro llamado Toby.",
    )]))
    result = await compact(SummaryCompaction(summary_max_tokens=500), client, memory)

    assert result.changed and len(client.memory_tasks) == 1
    task = client.memory_tasks[0]
    assert "- mascotas: Tiene un gato llamado Michi." in task  # existing memories sent
    assert "mi perro se llama Toby" in task  # messages leaving the window sent
    stored = {r.category: r.memory for r in await memory.get_context()}
    assert stored == {"mascotas": "Tiene un gato llamado Michi y un perro llamado Toby."}


async def test_nothing_new_leaves_memory_untouched(tmp_path):
    memory = LocalMemoryRegistry(user_id="u", session_id="s", base_path=tmp_path)
    client = FakeLLM(MemoryMaintenanceOutput(updates=[]))
    await compact(SummaryCompaction(summary_max_tokens=500), client, memory)
    assert client.memory_tasks and await memory.get_context() == []


async def test_prose_instead_of_structured_output_saves_nothing(tmp_path):
    memory = LocalMemoryRegistry(user_id="u", session_id="s", base_path=tmp_path)
    client = FakeLLM(prose="El perro se llama Toby.")
    result = await compact(SummaryCompaction(summary_max_tokens=500), client, memory)
    assert result.changed and await memory.get_context() == []


async def test_can_be_turned_off(tmp_path):
    memory = LocalMemoryRegistry(user_id="u", session_id="s", base_path=tmp_path)
    client = FakeLLM(MemoryMaintenanceOutput(updates=[MemoryFactUpdate(category="x", content="y")]))
    strategy = SummaryCompaction(summary_max_tokens=500, update_memory=False)
    result = await compact(strategy, client, memory)
    assert result.changed and client.memory_tasks == [] and await memory.get_context() == []
    restored = SummaryCompaction.deserialize(strategy.serialize())
    assert restored.config.update_memory is False
