"""The dispatcher turns a call through call_tool into the call to the real tool."""

from max_ai.capabilities.tools.call_tool import CallToolTool
from max_ai.capabilities.tools.decorator import tool as as_tool
from max_ai.core.tool import ToolDispatcher, ToolHook, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord

WEATHER: list[str] = []


async def weather(city: str) -> str:
    """Current forecast for a city."""
    WEATHER.append(city)
    return f"sunny in {city}"


async def hello(name: str) -> str:
    """Say hello."""
    return f"hello {name}"


class Sees(ToolHook):
    """Approves everything and remembers which tool it was asked about."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def permission(self, tool, record, tool_ctx):
        self.asked.append(tool.name)
        return True


def dispatcher(hook: ToolHook | None = None) -> ToolDispatcher:
    registry = ToolRegistry([as_tool(weather, deferred=True), as_tool(hello)])
    registry.register(CallToolTool(registry))
    return ToolDispatcher(registry, hook=hook or Sees())


async def call(dispatcher: ToolDispatcher, **parameters):
    record = ToolCallRecord(id="c", tool_name="call_tool", parameters=parameters)
    result = await dispatcher.dispatch(record, RunContext(user_id="u"))
    return record, result


async def test_the_deferred_tool_runs_and_the_answer_keeps_the_called_name():
    WEATHER.clear()
    hook = Sees()
    box = dispatcher(hook)
    record, result = await call(box, tool="weather", arguments={"city": "Lima"})
    assert result.success and WEATHER == ["Lima"]
    assert (record.tool_name, record.parameters, record.called_as) == (
        "weather", {"city": "Lima"}, "call_tool")
    assert hook.asked == ["weather"]  # permission is asked for the real tool
    assert box.tool_message(record, result).tool_name == "call_tool"


async def test_arguments_sent_as_a_json_string_work():
    WEATHER.clear()
    _, result = await call(dispatcher(), tool="weather", arguments='{"city": "Cusco"}')
    assert result.success and WEATHER == ["Cusco"]


async def test_a_wrong_call_tells_the_model_what_to_fix():
    record, result = await call(dispatcher(), tool="hello", arguments={"name": "x"})
    assert not result.success and "already loaded: call it directly" in result.error
    assert record.tool_name == "call_tool"
    _, result = await call(dispatcher(), tool="nope", arguments={})
    assert "No tool named 'nope'. Find it with search_tools first." in result.error
    _, result = await call(dispatcher(), tool="weather", arguments="not json")
    assert not result.success and "is not of type 'object'" in result.error
