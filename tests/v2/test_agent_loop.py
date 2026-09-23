import pytest

from max_ai.base.agent import Agent
from max_ai.base.tools import CoreTool
from max_ai.base.workspace import Workspace
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode


class CountingTool(CoreTool):
    def __init__(self):
        super().__init__("count", "count", approval_mode=ToolApprovalMode.AUTO_APPROVED)
        self.calls = 0
    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "additionalProperties": False}
    async def execute(self, record, context=None, cancellation_token=None):
        self.calls += 1
        from max_ai.types.tool_call import ToolResult
        return ToolResult.success_result(record.id, {"calls": self.calls})


class FakeClient:
    model = "fake"
    def __init__(self): self.calls = 0
    async def run(self, ctx, prompt, **kwargs):
        self.calls += 1
        return ChatCompletionResult(
            message=AssistantMessage(source="fake", content="done"), usage=Usage(),
            model=self.model, finish_reason="stop")


class ToolThenFinalClient(FakeClient):
    async def run(self, ctx, prompt, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ChatCompletionResult(
                message=AssistantMessage(source="fake", content="", tool_calls=[
                    ToolCall(id="call-1", tool_name="count", parameters={})
                ]), usage=Usage(), model=self.model, finish_reason="tool_calls")
        assert any(isinstance(message, ToolMessage) and message.tool_call_id == "call-1"
                   for message in ctx.messages)
        return ChatCompletionResult(
            message=AssistantMessage(source="fake", content="final"), usage=Usage(),
            model=self.model, finish_reason="stop")


@pytest.mark.asyncio
async def test_agent_mocked_model_loop_returns_text_and_calls_model_once(tmp_path):
    client = FakeClient()
    agent = Agent("test", "desc", "instructions", client, workspace=Workspace(tmp_path / "agents"))
    try:
        response = await agent.run("hello", run_context=RunContext(user_id="u", session_id="c"))
        assert response.finish_reason == "stop"
        assert response.context.messages[-1].text() == "done"
        assert client.calls == 1
    finally:
        await agent.close()
        await agent._manager.discard("u", "c")


@pytest.mark.asyncio
async def test_agent_resumes_approved_tool_record_before_next_model_call(tmp_path):
    client = FakeClient()
    tool = CountingTool()
    agent = Agent("test", "desc", "instructions", client, toolset=[tool], workspace=Workspace(tmp_path / "agents"))
    try:
        context = RunContext(user_id="u", session_id="c")
        record = ToolCallRecord(tool_name="count", parameters={})
        record.auto_approve()
        context.tool_state.records[record.id] = record
        response = await agent.run(run_context=context)
        assert tool.calls == 1
        assert any(isinstance(message, ToolMessage) and message.tool_call_id == record.id
                   for message in response.context.messages)
        assert response.context.messages[-1].text() == "done"
        assert client.calls == 1
    finally:
        await agent.close()
        await agent._manager.discard("u", "c")


@pytest.mark.asyncio
async def test_agent_model_requested_tool_is_fed_back_before_final_response(tmp_path):
    client = ToolThenFinalClient()
    tool = CountingTool()
    agent = Agent("test", "desc", "instructions", client, toolset=[tool], workspace=Workspace(tmp_path / "agents"))
    try:
        response = await agent.run("use the counter", run_context=RunContext(user_id="u", session_id="c"))
        assert response.finish_reason == "stop"
        assert response.context.messages[-1].text() == "final"
        assert tool.calls == 1
        assert client.calls == 2
        assert any(isinstance(message, ToolMessage) and message.tool_call_id == "call-1"
                   for message in response.context.messages)
    finally:
        await agent.close()
        await agent._manager.discard("u", "c")
