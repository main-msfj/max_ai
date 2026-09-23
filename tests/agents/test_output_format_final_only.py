"""output_format shapes only the accepted final answer, never the working calls."""

from __future__ import annotations

from pydantic import BaseModel

from max_ai.agents import Agent
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.core.model.llm import ModelConfig
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode


class Weather(BaseModel):
    city: str
    temp_c: int


def get_weather(city: str) -> dict:
    """Current weather for a city."""
    return {"city": city, "temp_c": 24}


class RecordingLLM:
    """Replays answers and records what each call was given."""

    model = "fake"

    def __init__(self, answers: list[AssistantMessage]):
        self.answers = answers
        self.calls: list[tuple[type | None, int]] = []
        self.config = ModelConfig()
        self.generation_options = {"max_tokens": 200}

    async def run(self, *, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        self.calls.append((output_format, len(tools or [])))
        message = self.answers.pop(0)
        if output_format is not None:
            message = message.model_copy(
                update={"structured_output": output_format.model_validate_json(message.content)}
            )
        return ChatCompletionResult(message=message, usage=Usage(), model="fake", finish_reason="stop")


def say(text: str) -> AssistantMessage:
    return AssistantMessage(source="llm", content=text)


async def test_only_the_final_answer_is_shaped(tmp_path):
    llm = RecordingLLM([
        AssistantMessage(source="llm", content="", tool_calls=[
            ToolCall(id="c1", tool_name="get_weather", parameters={"city": "Lima"})]),
        say("En Lima hace 24 grados."),
        say('{"city": "Lima", "temp_c": 24}'),
    ])
    tool = FunctionAsTool(get_weather, approval_mode=ToolApprovalMode.AUTO_APPROVED)
    async with Agent(name="w", description="d", instructions="i", client=llm, toolset=[tool],
                     workspace=LocalWorkspace(root=tmp_path), output_format=Weather) as agent:
        response = await agent.run("¿clima en Lima?", run_context=RunContext(user_id="u"))

    (first, tools_1), (second, tools_2), (shaping, tools_3) = llm.calls
    assert first is None and second is None and tools_1 > 0 and tools_2 > 0
    assert shaping is Weather and tools_3 == 0

    final = response.final_message
    assert final.text() == "En Lima hace 24 grados."  # the transcript keeps the prose
    assert final.structured_output == Weather(city="Lima", temp_c=24)


async def test_without_output_format_there_is_no_extra_call(tmp_path):
    llm = RecordingLLM([say("hola")])
    async with Agent(name="w", description="d", instructions="i", client=llm,
                     workspace=LocalWorkspace(root=tmp_path)) as agent:
        response = await agent.run("hola", run_context=RunContext(user_id="u"))
    assert len(llm.calls) == 1 and response.final_message.structured_output is None
