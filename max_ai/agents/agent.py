"""Agent runtime built around a tool registry, dispatcher and execution provider.

Parallel copy of base/agent.py for testing the new stack in isolation.
base/agent.py is left untouched.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from contextlib import aclosing
from typing import Any

from pydantic import BaseModel

from ..base.clients import CoreChatCompletionClient
from ..base.executor import ExecutorBase
from ..base.skills import CoreSkillRegistry
from ..base.reasoning import BaseReasoning
from ..capabilities.skills.local import LocalSkillRegistry
from ..reasoning.react import ReactLoop
from ..core.tool.dispatcher import ToolDispatcher
from ..core.tool.registry import ToolRegistry
from ..core.events_bus import EventBus
from ..base.tools import CoreTool, ToolContext
from ..base.workspace import WorkspaceBase
from ..capabilities.workspace.local import LocalWorkspace
from ..config import setting
from ..core.event_type import ModelStreamChunkEvent
from ..core.messages import (
    CoreMessage,
    UserMessage,
)
from ..core.environment.manager import EnvironmentManager
from ..manager.stacks import LayerContainer
from ..capabilities.executor.local import LocalExecutor
from ..capabilities.executor.reference import ToolReference
from ..stacks.agent_policy_layer import AgentPolicyLayer
from ..termination import CancellationToken
from ..capabilities.tools.ask_user import AskUserTool
from ..capabilities.tools.bash import BashTool
from ..capabilities.tools.file_system import FileSystem
from ..capabilities.tools.plan import AgentUpdatePlanTool
from ..types.agent_response import AgentResponse
from ..types.completions import Usage
from ..types.run_context import RunContext
from ..types.stacks import PromptCtx


class Agent:
    """Connect a pluggable reasoning loop to workspace, skills and execution.

    Supply CoreTool instances or decorated functions through toolset.
    The tool registry is managed internally by the agent.

    Defaults: LocalWorkspace, LocalExecutor, an empty LocalSkillRegistry,
    an owned EnvironmentManager and ReactLoop. Local execution is not a
    sandbox. A supplied environment owns its executor/workspace and remains
    the caller's responsibility to close. Do not share a reasoning instance
    between concurrently running agents: bind() stores runtime dependencies.
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        client: CoreChatCompletionClient,
        toolset: Sequence[CoreTool | Callable[..., Any]] | None = None,
        *,
        executor: ExecutorBase | None = None,
        workspace: WorkspaceBase | None = None,
        skills: CoreSkillRegistry | None = None,
        environment: EnvironmentManager | None = None,
        reasoning: BaseReasoning | None = None,
        max_iterations: int = 20,
        idle_timeout: float = 300,
        output_format: type[BaseModel] | None = None,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.name, self.description, self.instructions = name, description, instructions
        self.client = client
        if environment is not None:
            if not isinstance(environment, EnvironmentManager):
                raise TypeError("environment must be an EnvironmentManager")
            if executor is not None and executor is not environment.executor:
                raise ValueError("executor must match the supplied environment.executor")
            if workspace is not None and workspace is not environment.workspace:
                raise ValueError("workspace must match the supplied environment.workspace")
            workspace, executor = environment.workspace, environment.executor
        self.workspace = workspace if workspace is not None else LocalWorkspace(
            root=setting.root_dir / ".agents"
        )
        self.executor = executor if executor is not None else LocalExecutor()
        self.reasoning = reasoning if reasoning is not None else ReactLoop(
            max_loop_iterations=max_iterations
        )
        if not isinstance(self.reasoning, BaseReasoning):
            raise TypeError("reasoning must implement BaseReasoning")
        if not isinstance(self.executor, ExecutorBase):
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
        if self.reasoning.enable_human_input and self._registry.get(AskUserTool.TOOL_NAME) is None:
            self._registry.register(AskUserTool(), host=True)
        # host=True: mutates ctx.plan in-process (see tools/plan/_agent_tool.py).
        if self._registry.get(AgentUpdatePlanTool.TOOL_NAME) is None:
            self._registry.register(AgentUpdatePlanTool(), host=True)
        if self._registry.get("bash") is None:
            self._registry.register(BashTool())
        self.skills = skills if skills is not None else LocalSkillRegistry(
            source=setting.root_dir, skills=[]
        )
        self._skill_blocks = []
        self.max_iterations = max_iterations
        self.output_format = output_format
        self._owns_environment = environment is None
        self.environment = environment if environment is not None else EnvironmentManager(
            self.executor, self.workspace, idle_timeout=idle_timeout
        )
        self._manager = self.environment
        self.dispatcher = ToolDispatcher(
            self._registry, source=name, manager=self._manager
        )
        self.completion_bus = EventBus()
        self._turn_lock = asyncio.Lock()
        self._closed = False
        self._policy = AgentPolicyLayer()
        self._stack = LayerContainer([self._policy])

    async def prepare(self):
        await self.skills.prepare()
        self._skill_blocks = await self.skills.get_skills()

    def _prompts(self, ctx):
        variables = dict(
            name=self.name, description=self.description, instructions=self.instructions
        )
        rendered = self._policy.render(variables)
        rendered += (
            f"\nWorkspace belongs to user {ctx.user_id}. Skills are in skills/. "
            f"New files belong in conversation {ctx.session_id}/. "
            "File tools default to the current conversation. Pass only a short relative "
            "name such as report.md or documents/report.docx; do not repeat workspace, "
            "user, or conversation paths. find_files and search_text search all of this "
            "user's conversations. "
            "Execution tools start in this same conversation directory and receive it as "
            "WORKSPACE; do not assume a host path or access another user's workspace. "
            "Intermediate artifacts (downloaded pages, scraped raw data, throwaway scripts) "
            "belong in $SCRATCHPAD, not the conversation directory. If you do not need a file "
            "again after this step, do not save it at all — pipe/process it inline instead. "
            "Only save to the conversation directory the files the user actually asked for or "
            "will want to revisit. Clean up $SCRATCHPAD explicitly when you are done with it. "
            "Work through multi-step tasks autonomously: after a tool call, keep going to the "
            "next step yourself instead of stopping to report progress and wait. Only stop "
            "mid-task when you are genuinely blocked — need approval, need missing information "
            "from the user, or the task is fully complete."
        )
        if self._skill_blocks:
            rendered += "\nAvailable skills (read SKILL.md before using one):"
            for skill in self._skill_blocks:
                rendered += (
                    f"\n- {skill.name}: {skill.description} "
                    f"[skills/{skill.name}/SKILL.md]"
                )
        return PromptCtx(
            stack=self._stack,
            variables=variables,
            rendered_layers={AgentPolicyLayer: rendered},
        )

    async def _drive(self, ctx, sink, cancellation_token, stream_tokens, kwargs):
        def emit(event):
            self.completion_bus.emit(event)
            sink(event)

        await self.prepare()
        directory = self.workspace.materialize(ctx.user_id, ctx.session_id)
        if self.skills is not None:
            self.skills.materialize(directory)
        pending = ctx.runtime_state.shared_state.pop("reasoning_loop", {})
        loop_state = self.reasoning.LOOP_STATE_CLS()
        loop_state.apply_metrics(pending.get("metrics", {}))
        if hasattr(loop_state, "guard_state"):
            loop_state.guard_state.update(pending.get("guard_state", {}))
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
        reasoning = self.reasoning.bind(
            name=self.name, client=self.client,
            dispatcher=self.dispatcher, tool_context=context,
            completion_bus=self.completion_bus,
        )
        async with aclosing(reasoning.execute_reasoning_loop(
            ctx=ctx, prompts=self._prompts(ctx), loop_state=loop_state,
            stream_tokens=stream_tokens, cancellation_token=cancellation_token,
            output_format=self.output_format, **kwargs,
        )) as stream:
            async for event in stream:
                emit(event)
        duration_ms = pending.get("duration_ms", 0) + int(
            (time.monotonic() - started) * 1000
        )
        if loop_state.finish_reason in ("approval_needed", "input_needed"):
            ctx.runtime_state.shared_state["reasoning_loop"] = {
                "metrics": loop_state.metrics_snapshot(),
                "guard_state": getattr(loop_state, "guard_state", {}),
                "duration_ms": duration_ms,
            }
        usage = Usage(
            duration_ms=duration_ms, retries=loop_state.retries,
            llm_calls=loop_state.llm_calls,
            attempts_to_call_api=loop_state.attempts_to_call_api,
            tool_calls=loop_state.tool_calls,
            tokens_input=loop_state.tokens_input,
            tokens_output=loop_state.tokens_output,
            tokens_cached=loop_state.tokens_cached,
        )
        return AgentResponse(
            context=ctx, source=self.name, usage=usage,
            finish_reason=loop_state.finish_reason,
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

    async def resume(self, run_context: RunContext, **kwargs):
        """Continue after callers apply approvals or answers to tool records."""
        return await self.run(run_context=run_context, **kwargs)

    async def resume_stream_events(self, run_context: RunContext, **kwargs):
        async with aclosing(self.run_stream_events(run_context=run_context, **kwargs)) as stream:
            async for item in stream:
                yield item

    async def resume_stream(self, run_context: RunContext, **kwargs):
        async with aclosing(self.run_stream(run_context=run_context, **kwargs)) as stream:
            async for chunk in stream:
                yield chunk

    async def close(self):
        self._closed = True
        async with self._turn_lock:
            if self._owns_environment:
                await self._manager.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
