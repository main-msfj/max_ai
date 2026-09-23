import pytest
from max_ai.base.environment import ExecutionResult

from max_ai.base.tools import CoreTool, ToolContext
from max_ai.capabilities.tools.bash import BashTool
from max_ai.core.event_type import BashFinishedEvent, BashStartedEvent
from max_ai.core.tool.dispatcher import ToolDispatcher
from max_ai.core.tool.registry import ToolRegistry
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode


class EchoTool(CoreTool):
    def __init__(self, name="echo", mode=ToolApprovalMode.AUTO_APPROVED):
        super().__init__(name, "echo", approval_mode=mode, timeout_seconds=2)
        self.calls = 0

    @property
    def parameters(self):
        return {"type": "object", "properties": {"value": {"type": "string"}},
                "required": ["value"], "additionalProperties": False}

    async def execute(self, record, context=None, cancellation_token=None):
        self.calls += 1
        from max_ai.types.tool_call import ToolResult
        return ToolResult.success_result(record.id, record.parameters["value"])


def test_registry_wraps_callables_and_preserves_host_reference():
    registry = ToolRegistry()
    tool = registry.register(lambda value: value, reference="provider", host=True)
    assert registry.get(tool.name) is tool
    assert registry.reference(tool.name) == "provider"
    assert registry.runs_on_host(tool.name)
    assert registry.definitions()[0].name == tool.name


def test_registry_rejects_duplicates_and_noncallables():
    registry = ToolRegistry([EchoTool()])
    with pytest.raises(ValueError):
        registry.register(EchoTool())
    with pytest.raises(TypeError):
        registry.register(42)


@pytest.mark.asyncio
async def test_dispatcher_requests_approval_then_resumes_and_emits_events():
    tool = EchoTool(mode=ToolApprovalMode.ASK_APPROVED)
    dispatcher = ToolDispatcher(ToolRegistry([tool]))
    events = []
    context = ToolContext("run", emit_event=events.append)
    record = ToolCallRecord(tool_name="echo", parameters={"value": "ok"})
    assert await dispatcher.dispatch(record, context) is None
    assert len(events) == 2
    record.approve("reviewed")
    result = await dispatcher.dispatch(record, context)
    assert result.success and tool.calls == 1
    assert record.is_consumed


@pytest.mark.asyncio
async def test_dispatcher_routes_host_tool_without_manager():
    tool = EchoTool()
    registry = ToolRegistry()
    registry.register(tool, host=True)
    dispatcher = ToolDispatcher(registry, manager=object())
    record = ToolCallRecord(tool_name="echo", parameters={"value": "host"})
    result = await dispatcher.dispatch(record, ToolContext("run"))
    assert result.result == "host"


@pytest.mark.asyncio
async def test_dispatcher_routes_native_tool_through_provider_manager():
    class ProviderExecutor:
        def __init__(self): self.calls = []
        async def run_tool(self, session, tool, record, context, cancellation_token):
            self.calls.append((session, tool.name, context.deps["tool_reference"]))
            from max_ai.types.tool_call import ToolResult
            return ToolResult.success_result(record.id, "provider")
    class Manager:
        def __init__(self): self.executor = ProviderExecutor()
        def acquire(self, user_id, session_id):
            class Lease:
                async def __aenter__(self): return "session"
                async def __aexit__(self, *args): pass
            return Lease()
    tool = EchoTool()
    registry = ToolRegistry()
    registry.register(tool, reference="ref")
    manager = Manager()
    dispatcher = ToolDispatcher(registry, manager=manager)
    record = ToolCallRecord(tool_name="echo", parameters={"value": "provider"})
    result = await dispatcher.dispatch(record, ToolContext("run", user_id="u", session_id="c"))
    assert result.result == "provider"
    assert manager.executor.calls == [("session", "echo", "ref")]


def test_bash_parser_rejects_shell_operators_and_escape_paths():
    tool = BashTool()
    for command in ("cat a | rg a", "cat ../secret", "cat $HOME/file", "rm file"):
        with pytest.raises(ValueError):
            tool._command(command)
    assert tool._command("ls -la")[-1] == "."


def test_bash_parser_allows_only_workspace_relative_literal_paths(tmp_path):
    tool = BashTool()
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "link").symlink_to(root / "real")
    with pytest.raises(ValueError):
        tool._command("cat link", str(root))
    assert tool._command("cat $WORKSPACE/file", str(root))[-1] == str(root / "file")


@pytest.mark.asyncio
async def test_bash_emits_start_and_finish_events_for_bound_environment(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    class Env:
        user_id = "u"; conversation_id = "c"
        variables = {"WORKSPACE": str(root)}
        async def start(self): pass
        async def stop(self): pass
        async def execute(self, command, **kwargs):
            return ExecutionResult("out", "", 0)
    events = []
    tool = BashTool(Env(), approval_mode=ToolApprovalMode.AUTO_APPROVED)
    record = ToolCallRecord(tool_name="bash", parameters={"command": "ls", "action": "list_directory", "description": "list"})
    result = await tool.execute(record, ToolContext("r", "c", "u", emit_event=events.append))
    assert result.success
    assert isinstance(events[0], BashStartedEvent)
    assert isinstance(events[1], BashFinishedEvent)
