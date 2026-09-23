"""Agent runtime built around a tool registry, dispatcher and execution provider.

The previous full-featured implementation is preserved in agent_copy.py. This
initial runtime deliberately exposes a smaller constructor while migration is
in progress; unsupported legacy arguments raise rather than being ignored.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from contextlib import aclosing
from typing import Any

from pydantic import BaseModel

from ..capabilities.executor.local import LocalExecutor
from ..capabilities.executor.reference import ToolReference
from ..capabilities.tools.ask_user import AskUserTool
from ..capabilities.tools.file_system import FileSystem
from ..capabilities.tools.plan import AgentUpdatePlanTool
from ..config import setting
from ..core.environment.manager import EnvironmentManager
from ..core.event_type import ModelResponseEvent, ModelStreamChunkEvent
from ..core.messages import (
    AssistantMessage,
    CoreMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from ..manager.stacks import LayerContainer
from ..stacks.agent_policy_layer import AgentPolicyLayer
from ..termination import CancellationToken
from ..types.agent_response import AgentResponse
from ..types.completions import ChatCompletionResult, Usage
from ..types.run_context import RunContext
from ..types.stacks import PromptCtx
from ..types.tool_call import ToolCallRecord
from .clients import CoreChatCompletionClient
from .executor import Executor
from .skills import CoreSkillRegistry
from .tool_dispatcher import ToolDispatcher
from .tool_registry import ToolRegistry
from .tools import CoreTool, ToolContext
from .workspace import Workspace


class Agent:
    """Own the model/tool loop; all execution goes through the dispatcher.

    Supply CoreTool instances or decorated functions through toolset.
    The tool registry is managed internally by the agent.
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        client: CoreChatCompletionClient,
        toolset: Sequence[CoreTool | Callable[..., Any]] | None = None,
        *,
        executor: Executor | None = None,
        workspace: Workspace | None = None,
        skills: CoreSkillRegistry | None = None,
        max_iterations: int = 20,
        idle_timeout: float = 300,
        output_format: type[BaseModel] | None = None,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.name, self.description, self.instructions = name, description, instructions
        self.client = client
        self.workspace = workspace or Workspace(root=setting.root_dir / ".agents")
        self.executor = executor or LocalExecutor()
        if not isinstance(self.executor, Executor):
            raise TypeError("Use an executor from max_ai.capabilities.executor")
        self._registry = ToolRegistry()
        for tool in toolset or ():
            self._registry.register(tool)
        # host=True: touches the persistent Workspace, not Bash's sandbox.
        for tool in FileSystem().get_toolset().tools:
            if self._registry.get(tool.name) is None:
                self._registry.register(
                    tool,
                    host=True,
                    reference=ToolReference(
                        module="max_ai.capabilities.tools.file_system",
                        qualname="FileSystemTools",
                        kind="factory",
                        tool_name=tool.name,
                    ),
                )
        # host=True: pure ToolCallRecord state, no I/O.
        if self._registry.get(AskUserTool.TOOL_NAME) is None:
            self._registry.register(AskUserTool(), host=True)
        # host=True: mutates ctx.plan in-process (see tools/plan/_agent_tool.py).
        if self._registry.get(AgentUpdatePlanTool.TOOL_NAME) is None:
            self._registry.register(AgentUpdatePlanTool(), host=True)
        self.skills = skills
        self.max_iterations = max_iterations
        self.output_format = output_format
        self._manager = EnvironmentManager(
            self.executor, self.workspace, idle_timeout=idle_timeout
        )
        self.dispatcher = ToolDispatcher(
            self._registry, source=name, manager=self._manager
        )
        self._turn_lock = asyncio.Lock()
        self._closed = False
        self._policy = AgentPolicyLayer()
        self._stack = LayerContainer([self._policy])

    async def prepare(self):
        if self.skills is not None:
            await self.skills.prepare()

    def _prompts(self, ctx):
        variables = dict(
            name=self.name, description=self.description, instructions=self.instructions
        )
        rendered = self._policy.render(variables)
        rendered += (
            f"\nWorkspace belongs to user {ctx.user_id}. Skills are in skills/. "
            f"New files belong in conversation {ctx.session_id}/. "
            "File tools use user-relative paths, except write_file which uses "
            "conversation-relative paths. Other conversations of this user are accessible. "
            "Execution tools receive WORKSPACE in their environment; do not assume a host path."
        )
        if self.skills is not None:
            rendered += "\nAvailable skills: " + ", ".join(self.skills.skills)
        return PromptCtx(
            stack=self._stack,
            variables=variables,
            rendered_layers={AgentPolicyLayer: rendered},
        )

    async def _complete(self, ctx, stream_tokens, emit, **kwargs):
        output = await self.client.run(
            ctx,
            self._prompts(ctx),
            tools=self._registry.all_tools(),
            output_format=self.output_format,
            stream=stream_tokens,
            **kwargs,
        )
        if not stream_tokens:
            return output
        text, thinking, calls, usage, structured = [], [], {}, Usage(), None
        async with aclosing(output):
            async for chunk in output:
                text.append(chunk.content)
                if chunk.thinking:
                    thinking.append(chunk.thinking)
                emit(
                    ModelStreamChunkEvent(
                        source=self.name,
                        chunk=chunk.content,
                        thinking=chunk.thinking,
                        is_final=chunk.is_complete,
                    )
                )
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.structured_output is not None:
                    structured = chunk.structured_output
                fragment = chunk.tool_call_chunk
                if fragment:
                    call_id = fragment.get("id")
                    if not call_id:
                        raise ValueError(
                            "Provider emitted a tool fragment without an id"
                        )
                    current = calls.setdefault(call_id, {"name": "", "arguments": ""})
                    fn = fragment.get("function") or {}
                    current["name"] = fn.get("name") or current["name"]
                    current["arguments"] += fn.get("arguments") or ""
        message = AssistantMessage(
            source=self.name,
            content="".join(text),
            thinking="".join(thinking) or None,
            structured_output=structured,
            tool_calls=[
                ToolCall(
                    id=key,
                    tool_name=value["name"],
                    parameters=json.loads(value["arguments"] or "{}"),
                )
                for key, value in calls.items()
            ],
        )
        return ChatCompletionResult(
            message=message,
            usage=usage,
            model=self.client.model,
            finish_reason="tool_calls" if calls else "stop",
        )

    @staticmethod
    def _append_results(ctx):
        present = {m.tool_call_id for m in ctx.messages if isinstance(m, ToolMessage)}
        for record in ctx.tool_state.records.values():
            if not record.is_consumed or record.id in present:
                continue
            result = record.result
            if result is None:
                raise ValueError("Consumed tool call has no result")
            ctx.messages.append(
                ToolMessage(
                    source=record.tool_name,
                    tool_call_id=record.id,
                    tool_name=record.tool_name,
                    success=result.success,
                    error=result.error,
                    content=json.dumps(result.result, ensure_ascii=False, default=str)
                    if result.success
                    else result.error or "Tool failed",
                )
            )

    async def _drive(self, ctx, emit, cancellation_token, stream_tokens, kwargs):
        await self.prepare()
        directory = self.workspace.materialize(ctx.user_id, ctx.session_id)
        if self.skills is not None:
            self.skills.materialize(directory)
        pending = ctx.runtime_state.shared_state.pop("native_agent_loop", {})
        iterations = pending.get("iterations", 0)
        usage = Usage.model_validate(pending.get("usage", {}))
        started = time.monotonic()
        context = ToolContext(
            run_id=ctx.run_id,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            emit_event=emit,
            deps={
                "runtime_root": str(directory.root),
                "conversation_dir": str(directory.conversation_dir),
                "scratch_dir": str(directory.scratch_dir),
                "skills_dir": str(directory.skill_dir),
                "filesystem_root": str(self.workspace.base_root),
                "workspace_filesystem": self.workspace.get_filesystem(),
                # Generic hook for tools needing the live RunContext
                # in-process (e.g. AgentUpdatePlanTool writing ctx.plan).
                "run_context": ctx,
            },
        )
        finish_reason = "max_iterations"
        while True:
            if cancellation_token is not None and cancellation_token.is_cancelled():
                raise asyncio.CancelledError
            self._append_results(ctx)
            # Resume unresolved calls before requesting another model response.
            for record in list(ctx.tool_state.records.values()):
                if record.is_consumed:
                    continue
                if record.is_awaiting_input:
                    finish_reason = "input_needed"
                    break
                result = await self.dispatcher.dispatch(
                    record, context, cancellation_token
                )
                if result is None:
                    finish_reason = (
                        "input_needed"
                        if record.is_awaiting_input
                        else "approval_needed"
                    )
                    break
                usage.tool_calls += 1
                self._append_results(ctx)
            if finish_reason in ("approval_needed", "input_needed"):
                break
            if iterations >= self.max_iterations:
                break
            completion = await self._complete(ctx, stream_tokens, emit, **kwargs)
            iterations += 1
            usage = usage + completion.usage.model_copy(
                update={"llm_calls": 1, "tool_calls": 0}
            )
            ctx.messages.append(completion.message)
            emit(
                ModelResponseEvent(
                    source=self.name,
                    response=completion.message.text(),
                    has_tool_calls=bool(completion.message.tool_calls),
                    usage=completion.usage,
                )
            )
            if not completion.message.tool_calls:
                finish_reason = "stop"
                break
            for call in completion.message.tool_calls:
                ctx.tool_state.add(
                    ToolCallRecord(
                        id=call.id,
                        tool_name=call.tool_name,
                        parameters=call.parameters,
                        session_id=ctx.session_id,
                    )
                )
        usage.duration_ms += int((time.monotonic() - started) * 1000)
        if finish_reason in ("approval_needed", "input_needed"):
            ctx.runtime_state.shared_state["native_agent_loop"] = {
                "iterations": iterations,
                "usage": usage.model_dump(),
            }
        return AgentResponse(
            context=ctx, source=self.name, usage=usage, finish_reason=finish_reason
        )

    async def run_stream_events(
        self,
        task: str | CoreMessage | list[CoreMessage] | None = None,
        run_context: RunContext | None = None,
        cancellation_token: CancellationToken | None = None,
        stream_tokens: bool = False,
        **kwargs,
    ):
        async with self._turn_lock:
            if self._closed:
                raise RuntimeError("Agent is closed")
            ctx = run_context if run_context is not None else RunContext()
            ctx.session_id = ctx.session_id or ctx.run_id
            if task is not None:
                if any(not r.is_consumed for r in ctx.tool_state.records.values()):
                    raise ValueError(
                        "Resolve pending tool calls before adding another task"
                    )
                ctx.messages.extend(
                    [UserMessage(source=ctx.user_id, content=task)]
                    if isinstance(task, str)
                    else task
                    if isinstance(task, list)
                    else [task]
                )
            queue = asyncio.Queue()
            sentinel = object()

            async def produce():
                try:
                    return await self._drive(
                        ctx, queue.put_nowait, cancellation_token, stream_tokens, kwargs
                    )
                finally:
                    queue.put_nowait(sentinel)

            producer = asyncio.create_task(produce())
            if cancellation_token is not None:
                cancellation_token.link_future(producer)
            try:
                while (item := await queue.get()) is not sentinel:
                    yield item
                response = await producer
            finally:
                if not producer.done():
                    producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)
            yield response

    async def run(self, *args, **kwargs):
        async with aclosing(self.run_stream_events(*args, **kwargs)) as stream:
            async for item in stream:
                if isinstance(item, AgentResponse):
                    return item
        raise RuntimeError("Agent produced no response")

    async def run_stream(self, *args, **kwargs):
        kwargs["stream_tokens"] = True
        async with aclosing(self.run_stream_events(*args, **kwargs)) as stream:
            async for item in stream:
                if isinstance(item, ModelStreamChunkEvent) and item.chunk:
                    yield item.chunk

    async def close(self):
        self._closed = True
        async with self._turn_lock:
            await self._manager.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
