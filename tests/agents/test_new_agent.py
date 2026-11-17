import pytest
from pydantic import BaseModel

from max_ai.agents.agent import Agent
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.capabilities.loop import ReactLoop
from max_ai.capabilities.memory import Memory
from max_ai.capabilities.tools.decorator import tool as as_tool


class Answer(BaseModel):
    city: str


def client():
    return OpenRouterChatCompletionClient("openai/gpt-test", api_key="sk-test")


def agent(**pieces):
    return Agent("helper", "Answers questions.", "Be brief.", pieces.pop("client", client()), **pieces)


def test_an_agent_takes_its_pieces_and_the_loop_has_a_default():
    made = agent(memory=Memory(read_only=True))
    assert isinstance(made.reasoning, ReactLoop)
    assert made.toolset == [] and made.sandbox is None and made.memory.read_only


@pytest.mark.parametrize(
    ("piece", "value", "said"),
    [
        ("client", "gpt", "client must be a CoreChatCompletionClient; got str"),
        ("reasoning", object(), "reasoning must be a CoreLoop"),
        ("toolset", [len], "toolset must be a CoreTool"),
        ("mcp", [{"url": "x"}], "mcp must be a MCPServerConfig; got dict"),
        ("memory", True, "memory must be a Memory; got bool"),
        ("skills", "skills/", "skills must be a SkillsCore"),
        ("sandbox", "/tmp", "sandbox must be a SandboxCore"),
        ("prompt_layers", ["Be nice"], "prompt_layers must be a CoreLayer"),
        ("middlewares", [print], "middlewares must be a CoreMiddleware"),
        ("gates", [1], "gates must be a CompletionBase"),
        ("knowledge", ["docs"], "knowledge must be a CoreKnowledgeRegistry"),
        ("output_format", Answer(city="x"), "output_format must be a pydantic BaseModel class"),
    ],
)
def test_a_piece_that_is_not_its_core_class_is_refused_by_name(piece, value, said):
    with pytest.raises(TypeError, match=said):
        agent(**{piece: value})


def test_an_agent_without_a_name_is_refused():
    with pytest.raises(ValueError, match="name cannot be empty"):
        Agent("  ", "d", "i", client())


def test_an_agent_is_stored_and_built_again(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")  # only the variable's name is stored
    made = agent(
        reasoning=ReactLoop(max_iterations=7), memory=Memory(read_only=True),
        output_format=Answer,
    )
    stored = made.serialize().model_dump_json()
    assert "sk-test" not in stored  # the key never leaves the object
    again = Agent.deserialize(stored)
    assert isinstance(again, Agent) and (again.name, again.instructions) == ("helper", "Be brief.")
    assert again.client.model == "openai/gpt-test" and again.reasoning.max_iterations == 7
    assert again.memory.read_only and "city" in again.output_format.model_json_schema()["properties"]


def test_a_python_function_as_tool_cannot_be_stored():
    def weather(city: str) -> str:
        """The weather in a city."""
        return city

    with pytest.raises(TypeError, match="expose them through an MCP server"):
        agent(toolset=[as_tool(weather)]).serialize()


async def test_the_agent_runs_a_turn_with_its_tools_in_a_sandbox(tmp_path):
    from max_ai.capabilities.sandbox.local import LocalSandbox
    from max_ai.core.messages import AssistantMessage, ToolCall
    from max_ai.types.completions import ChatCompletionResult, Usage
    from max_ai.types.run_context import RunContext

    answers = [
        AssistantMessage(source="m", content="", tool_calls=[
            ToolCall(id="c1", tool_name="write_file", parameters={"path": "notes.md", "content": "hi"})
        ]),
        AssistantMessage(source="m", content="Written."),
    ]
    seen = []

    async def model(messages, tools=None, output_format=None, *, retry=None, on_retry=None):
        seen.append((messages, [tool.name for tool in tools or []]))
        message = answers.pop(0)
        reason = "tool_calls" if message.tool_calls else "stop"
        return ChatCompletionResult(message=message, finish_reason=reason, usage=Usage(), model="m")

    made = agent(sandbox=LocalSandbox(str(tmp_path)), memory=Memory())
    made.client.run = model
    ctx = RunContext(user_id="u", session_id="s")
    state = await made.run("Write hi to notes.md", ctx)
    assert state.finish_reason == "stop" and state.message.text() == "Written."
    assert (tmp_path / ".agent" / "workspace" / "s" / "notes.md").read_text() == "hi"
    system, tools = str(seen[0][0][0].content), seen[0][1]
    # Name, description and instructions come from the agent policy layer.
    assert "<name>helper.</name>" in system and "Answers questions." in system
    assert "Be brief." in system and "[harness]" in system and "Memory files" in system
    assert system.index("<WORKSPACE>") < system.index("<MEMORY>")  # memory is its own layer
    assert str(tmp_path / ".agent" / "workspace" / "s") in system
    assert set(tools) >= {"ask_user", "write_todos", "read_file", "write_file", "edit_file"}
    assert ctx.context is not None and ctx.context.tools > 0
    await made.close()
    with pytest.raises(RuntimeError, match="closed"):
        await made.run("again", ctx)


async def test_close_waits_for_a_run_that_is_still_going():
    import asyncio

    from max_ai.core.messages import AssistantMessage
    from max_ai.types.completions import ChatCompletionResult, Usage

    started, release = asyncio.Event(), asyncio.Event()

    async def model(messages, tools=None, output_format=None, *, retry=None, on_retry=None):
        started.set()
        await release.wait()
        message = AssistantMessage(source="m", content="Done.")
        return ChatCompletionResult(message=message, finish_reason="stop", usage=Usage(), model="m")

    made = agent()
    made.client.run = model
    running = asyncio.create_task(made.run("hi"))
    await started.wait()
    closing = asyncio.create_task(made.close())
    await asyncio.sleep(0.05)
    assert not closing.done()  # the run is alive: the agent is not killed
    with pytest.raises(RuntimeError, match="closed"):
        await made.run("another")
    release.set()
    assert (await running).message.text() == "Done."
    await closing


async def test_run_stream_gives_only_the_text_as_it_is_written():
    from max_ai.core.messages import AssistantMessage
    from max_ai.types.completions import ChatCompletionChunk, Usage

    async def model(messages, tools=None, output_format=None, *, retry=None, on_retry=None):
        for piece in ("Hel", "lo"):
            yield ChatCompletionChunk(content=piece, model="m", is_complete=False)
        message = AssistantMessage(source="m", content="Hello")
        yield ChatCompletionChunk(
            content="", model="m", finish_reason="stop", usage=Usage(), message=message,
            is_complete=True,
        )

    made = agent()
    made.client.stream = model
    assert [text async for text in made.run_stream("hi")] == ["Hel", "lo"]
