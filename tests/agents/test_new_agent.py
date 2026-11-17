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
