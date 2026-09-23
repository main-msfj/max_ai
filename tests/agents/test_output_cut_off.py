"""A reply cut off at the output limit is explained to the model, never run."""

from __future__ import annotations

from max_ai.agents import Agent
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.core.model.llm import ModelConfig
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext


class CutOffLLM:
    """First reply: a half-written write_file cut at max_tokens."""

    model = "fake"

    def __init__(self, cut_offs: int = 1):
        self.config = ModelConfig()
        self.generation_options = {"max_tokens": 1500}
        self.seen: list[str] = []
        broken = ToolCall(id="c1", tool_name="write_file", parameters={
            "raw_error_content": '{"file_name": "snake.html", "content": "<html>', "parsing_error": True})
        self.replies = [
            (AssistantMessage(source="llm", content="", tool_calls=[broken]), "length")
        ] * cut_offs + [(AssistantMessage(source="llm", content="Lo divido en partes."), "stop")]

    async def run(self, *, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        self.seen.append("\n".join(str(m.content) for m in ctx.messages))
        message, reason = self.replies.pop(0)
        return ChatCompletionResult(message=message, usage=Usage(), model="fake", finish_reason=reason)


async def test_cut_off_call_is_dropped_and_explained(tmp_path):
    llm = CutOffLLM()
    async with Agent(name="a", description="d", instructions="i", client=llm,
                     workspace=LocalWorkspace(root=tmp_path)) as agent:
        response = await agent.run("haz un snake", run_context=RunContext(user_id="u"))

    assert "output limit" not in llm.seen[0]
    assert "output limit (~1,500 tokens)" in llm.seen[1]  # the model learns why
    assert not response.context.tool_state.records  # the broken call never ran
    assert response.final_text == "Lo divido en partes."


async def test_repeated_cut_offs_stop_the_turn(tmp_path):
    llm = CutOffLLM(cut_offs=5)
    async with Agent(name="a", description="d", instructions="i", client=llm,
                     workspace=LocalWorkspace(root=tmp_path)) as agent:
        response = await agent.run("haz un snake", run_context=RunContext(user_id="u"))
    assert response.finish_reason == "output_limit"
    assert len(llm.seen) == 2  # not every iteration burned on the same cut-off
