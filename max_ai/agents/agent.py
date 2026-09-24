"""Agent runtime built around a tool registry, dispatcher and execution provider.

Parallel copy of base/agent.py for testing the new stack in isolation.
base/agent.py is left untouched.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import weakref
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import aclosing
from types import TracebackType
from typing import Any, Self, get_args

from pydantic import BaseModel

from ..base.clients import CoreChatCompletionClient
from ..base.compaction import CoreCompaction
from ..base.completion_gate import CompletionDecision
from ..base.component import ComponentBase
from ..base.executor import ExecutorBase
from ..base.knowledge import CoreKnowledgeRegistry
from ..base.layer import CoreLayer
from ..base.memory import CoreMemoryRegistry
from ..base.middleware import CoreMiddleware, MiddlewareContext, StopRun
from ..base.reasoning import BaseReasoning
from ..base.skills import CoreSkillRegistry
from ..base.tools import CoreTool, ToolContext
from ..base.workspace import WorkspaceBase
from ..capabilities.completion_gate import RuntimeCompletionGate, RuntimeGateConfig
from ..capabilities.executor.local import LocalExecutor
from ..capabilities.mcp import MCPClientManager, MCPServerConfig
from ..capabilities.mcp._model import deserialize_mcp_servers, serialize_mcp_servers
from ..capabilities.reasoning.react import ReactLoop
from ..capabilities.stacks import (
    KnowledgeLayer,
    MemoryLayer,
    RenderingLayer,
    SessionStateLayer,
    SkillsLayer,
    TaskAnalysisLayer,
)
from ..capabilities.stacks.agent_policy_layer import AgentPolicyLayer
from ..capabilities.tools.ask_user import AskUserTool
from ..capabilities.tools.bash import BashTool
from ..capabilities.tools.file_system import FileSystem
from ..capabilities.tools.plan import AgentUpdatePlanTool
from ..capabilities.workspace.local import LocalWorkspace
from ..config import setting
from ..core.compaction import TokenCounter
from ..core.environment.manager import EnvironmentManager
from ..core.event_type import CoreEvent, ModelStreamChunkEvent
from ..core.events_bus import CompletionHandler, EventBus
from ..core.executor.reference import ToolReference
from ..core.messages import (
    CoreMessage,
    UserMessage,
)
from ..core.middleware import MiddlewareChain
from ..core.model.agent import AgentSpec
from ..core.model.json_schema import model_from_schema, schema_of
from ..core.stacks.container import LayerContainer
from ..core.termination import CancellationToken
from ..core.tool.dispatcher import ToolDispatcher
from ..core.tool.registry import ToolRegistry
from ..types.agent_response import AgentResponse, FinishReason
from ..types.completions import ChatCompletionResult, Usage
from ..types.run_context import RunContext
from ..types.stacks import PromptCtx


def _storable(component: Any, what: str) -> ComponentBase[Any]:
    """``component`` if it can be stored; a clear error saying what to do if not."""
    try:
        component.serialize()
    except (NotImplementedError, AttributeError, TypeError) as error:
        name = getattr(component, "name", None) or type(component).__name__
        hint = (
            "Tools written as Python functions are code and cannot be stored: "
            "expose them through an MCP server (mcp=[...]) instead."
            if what == "tool" else
            "Give it a component_schema, _to_config and _from_config."
        )
        raise TypeError(f"Cannot serialize {what} {name!r}: {hint}") from error
    return component


class Agent(ComponentBase[AgentSpec]):
    """Connect a pluggable reasoning loop to workspace, skills and execution.

    Supply CoreTool instances or decorated functions through toolset.
    The tool registry is managed internally by the agent.

    Memory, skills and knowledge are optional and only enabled when supplied.
    Supplied registries remain caller-owned. Configure memory with its backend
    only: each run binds it to that RunContext's user_id/session_id, so one
    agent serves every user and session without ever mixing their memories.

    Defaults: LocalWorkspace, LocalExecutor and ReactLoop. The agent manages
    execution sessions internally and closes them on close(). Local execution
    is not a sandbox.

    Concurrency: one Agent serves many runs at once. Each run executes on
    its own bound copy of the reasoning loop, MCP connections are opened
    once and shared, and only messages of the same (user_id, session_id)
    wait for each other. ``close()`` waits for the active runs.

    ``serialize()`` turns the agent into storable data (``AgentSpec``) and
    ``Agent.deserialize(row)`` rebuilds it: store once, rebuild per request.
    """

    component_type = "agent"
    component_schema = AgentSpec
    component_provider_override = "maxai.agents.Agent"

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        client: CoreChatCompletionClient,
        *,
        toolset: Sequence[CoreTool | Callable[..., Any]] | None = None,
        mcp: Sequence[MCPServerConfig] | None = None,
        executor: ExecutorBase | None = None,
        workspace: WorkspaceBase | None = None,
        skills: CoreSkillRegistry | None = None,
        memory: CoreMemoryRegistry | None = None,
        knowledge: Sequence[CoreKnowledgeRegistry] | None = None,
        reasoning: BaseReasoning | None = None,
        output_format: type[BaseModel] | None = None,
        completion_handlers: Sequence[CompletionHandler] | None = None,
        completion: RuntimeGateConfig | None = None,
        compaction: CoreCompaction | None = None,
        middlewares: Sequence[CoreMiddleware] | None = None,
    ) -> None:
        self._validate_configuration(executor, reasoning)
        if compaction is not None and not isinstance(compaction, CoreCompaction):
            raise TypeError("compaction must implement CoreCompaction")
        # None: the window is never compacted (fine for short sessions).
        self.compaction = compaction

        self.name = name
        self.description = description
        self.instructions = instructions
        self.client = client
        self.output_format = output_format
        self.middlewares = list(middlewares or [])
        for middleware in self.middlewares:
            if not isinstance(middleware, CoreMiddleware):
                raise TypeError("middlewares must implement CoreMiddleware")
        self._middleware = MiddlewareChain(self.middlewares)

        self._configure_environment(executor, workspace)
        # Iteration limit lives in the loop (default: setting.max_loop_iterations).
        self.reasoning = reasoning if reasoning is not None else ReactLoop()
        self._configure_tools(toolset)
        self.mcp_servers = tuple(mcp or ())
        self._mcp_manager = MCPClientManager()
        for config in self.mcp_servers:
            self._mcp_manager.add_server(config)
        self._configure_capabilities(memory, skills, knowledge)
        self._configure_runtime(completion_handlers, completion)
        self._stack = self._build_prompt_stack()

    @staticmethod
    def _validate_configuration(
        executor: ExecutorBase | None,
        reasoning: BaseReasoning | None,
    ) -> None:
        """Reject invalid options before constructing runtime components."""
        if reasoning is not None and not isinstance(reasoning, BaseReasoning):
            raise TypeError("reasoning must implement BaseReasoning")
        if executor is not None and not isinstance(executor, ExecutorBase):
            raise TypeError("Use an executor from max_ai.capabilities.executor")

    def _configure_environment(
        self,
        executor: ExecutorBase | None,
        workspace: WorkspaceBase | None,
    ) -> None:
        """Create the internal session manager from the public components."""
        self.workspace = (
            workspace
            if workspace is not None
            else LocalWorkspace(root=setting.root_dir / ".agents")
        )
        self.executor = executor if executor is not None else LocalExecutor()
        # Idle sessions close after setting.environment_idle_timeout.
        self._manager = EnvironmentManager(self.executor, self.workspace)

    def _configure_tools(
        self,
        toolset: Sequence[CoreTool | Callable[..., Any]] | None,
    ) -> None:
        """User tools take precedence over built-in defaults."""
        # The developer's tools; built-ins are recreated, never stored.
        self.toolset = list(toolset or ())
        self._registry = ToolRegistry(self.toolset)
        self._register_filesystem_tools()
        self._register_control_tools()

    def _register_filesystem_tools(self) -> None:
        """Filesystem tools access the persistent workspace on the host."""
        for tool in FileSystem().get_toolset().tools:
            if self._registry.get(tool.name) is not None:
                continue
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

    def _register_control_tools(self) -> None:
        """Plan and user input run on the host; Bash uses the executor."""
        if (
            self.reasoning.enable_human_input
            and self._registry.get(AskUserTool.TOOL_NAME) is None
        ):
            self._registry.register(AskUserTool(), host=True)
        if self._registry.get(AgentUpdatePlanTool.TOOL_NAME) is None:
            self._registry.register(AgentUpdatePlanTool(), host=True)
        if self._registry.get("bash") is None:
            self._registry.register(BashTool())

    def _configure_capabilities(
        self,
        memory: CoreMemoryRegistry | None,
        skills: CoreSkillRegistry | None,
        knowledge: Sequence[CoreKnowledgeRegistry] | None,
    ) -> None:
        """Keep optional registries and register only their exposed tools."""
        self.skills = skills
        self.memory = memory
        self.knowledge = tuple(knowledge or ())
        self._skill_blocks = []
        self._memory_tools = self._register_capability_tools(
            memory.tools if memory is not None else ()
        )
        self._knowledge_tools = self._register_capability_tools(
            [tool for source in self.knowledge for tool in source.tools]
        )

    def _configure_runtime(
        self,
        completion_handlers: Sequence[CompletionHandler] | None,
        completion: RuntimeGateConfig | None,
    ) -> None:
        """Connect dispatch, completion checks and turn synchronization."""
        self.dispatcher = ToolDispatcher(
            self._registry, source=self.name, manager=self._manager,
            middleware=self._middleware,
        )
        self.completion = completion or RuntimeGateConfig()
        self.completion_handlers = list(completion_handlers or [])
        self.completion_bus = EventBus(
            handlers=[
                RuntimeCompletionGate(self.workspace, self.completion),
                *self.completion_handlers,
            ]
        )
        # Runs of different sessions overlap; messages of one session queue.
        self._session_locks: weakref.WeakValueDictionary[tuple[str, str], asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._active_runs = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._mcp_lock = asyncio.Lock()
        self._closed = False

    def _register_capability_tools(self, tools: Sequence[CoreTool]) -> list[str]:
        """Register bound registry tools on the host and retain their names."""
        for tool in tools:
            self._registry.register(tool, host=True)
        return [tool.name for tool in tools]

    def _build_prompt_stack(self) -> LayerContainer:
        """Only configured capabilities contribute layers; context is deferred."""
        layers: list[CoreLayer] = [
            AgentPolicyLayer(),
            TaskAnalysisLayer(),
            RenderingLayer(),
        ]
        if self.skills is not None:
            layers.append(SkillsLayer())
        if self.knowledge:
            layers.append(KnowledgeLayer())
        if self.memory is not None:
            layers.append(MemoryLayer())
        # Compaction summary and plan: must survive compaction of the transcript.
        layers.append(SessionStateLayer())
        return LayerContainer(layers)

    # -------- SERIALIZATION ------------------------------------------------------------
    def _to_config(self) -> AgentSpec:
        def dump(component: ComponentBase[Any]) -> dict[str, Any]:
            return component.serialize().model_dump(exclude_none=True)

        return AgentSpec(
            name=self.name,
            description=self.description,
            instructions=self.instructions,
            client=dump(self.client),
            reasoning=dump(self.reasoning),
            workspace=dump(self.workspace),
            executor=dump(self.executor),
            memory=dump(self.memory) if self.memory is not None else None,
            skills=dump(self.skills) if self.skills is not None else None,
            knowledge=[dump(source) for source in self.knowledge],
            compaction=dump(self.compaction) if self.compaction is not None else None,
            toolset=[dump(_storable(tool, "tool")) for tool in self.toolset],
            mcp=json.loads(serialize_mcp_servers(self.mcp_servers)),
            completion=self.completion.model_dump(),
            completion_handlers=[
                dump(_storable(gate, "completion handler")) for gate in self.completion_handlers
            ],
            output_format=schema_of(self.output_format) if self.output_format else None,
            middlewares=[dump(_storable(m, "middleware")) for m in self.middlewares],
        )

    @classmethod
    def _from_config(cls, config: AgentSpec) -> Self:
        def load(data: dict[str, Any] | None, kind: type[Any]) -> Any:
            return kind.deserialize(data) if data is not None else None

        return cls(
            name=config.name,
            description=config.description,
            instructions=config.instructions,
            client=load(config.client, CoreChatCompletionClient),
            toolset=[load(tool, CoreTool) for tool in config.toolset],
            mcp=deserialize_mcp_servers(json.dumps(config.mcp)),
            executor=load(config.executor, ExecutorBase),
            workspace=load(config.workspace, WorkspaceBase),
            skills=load(config.skills, CoreSkillRegistry),
            memory=load(config.memory, CoreMemoryRegistry),
            knowledge=[load(source, CoreKnowledgeRegistry) for source in config.knowledge],
            reasoning=load(config.reasoning, BaseReasoning),
            output_format=(
                model_from_schema(config.output_format) if config.output_format else None
            ),
            completion_handlers=[
                load(gate, ComponentBase) for gate in config.completion_handlers
            ],
            completion=RuntimeGateConfig.model_validate(config.completion),
            compaction=load(config.compaction, CoreCompaction),
            middlewares=[load(m, CoreMiddleware) for m in config.middlewares],
        )

    @property
    def tools(self) -> list[CoreTool]:
        """Every tool the model can call (host, capability and native)."""
        return self._registry.all_tools()

    async def prepare(self) -> None:
        """Load selected skill metadata; retrieval backends connect on demand."""
        if self.skills is not None:
            self._skill_blocks = await self.skills.get_skills()

    async def _memory_for(self, ctx: RunContext) -> CoreMemoryRegistry | None:
        """The memory scoped to this run's user and session. The registry is
        connected first so every bound copy shares one backend client."""
        if self.memory is None:
            return None
        await self.memory._ensure_connected()
        return self.memory.bind(ctx.user_id, ctx.session_id or ctx.run_id)

    async def _prompt_variables(self, ctx: RunContext) -> dict[str, Any]:
        """Collect current data without making the templates access registries."""
        variables: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
        }
        if self.skills is not None:
            variables["loaded_skills"] = self._skill_blocks
        if self.knowledge:
            variables["retrieval_tools"] = self._knowledge_tools
        memory = await self._memory_for(ctx)
        if memory is not None:
            variables["persistent_memories"] = await memory.get_context()
            variables["memory_tools"] = self._memory_tools
        return variables

    async def _prompts(self, ctx: RunContext) -> PromptCtx:
        """Render a fresh snapshot for each run/resume in stack order."""
        variables = await self._prompt_variables(ctx)
        prompts = PromptCtx(
            stack=self._stack,
            variables=variables,
            rendered_layers={
                type(layer): layer.render(variables) for layer in self._stack
            },
        )
        # The real prompt size, so compaction budgets the window correctly.
        tokenizer = getattr(self.client.config, "tokenizer_base", None)
        prompts.measure(TokenCounter(tokenizer_base=tokenizer) if tokenizer else TokenCounter())
        return prompts

    async def _drive(
        self,
        ctx: RunContext,
        sink: Callable[[CoreEvent], None],
        cancellation_token: CancellationToken | None,
        stream_tokens: bool,
        kwargs: dict[str, Any],
        task: list[CoreMessage] | None = None,
    ) -> AgentResponse:
        """Make sure MCP tools are available, then run."""
        await self._ensure_mcp()
        return await self._drive_connected(
            ctx, sink, cancellation_token, stream_tokens, kwargs, task
        )

    async def _ensure_mcp(self) -> None:
        """Connect MCP servers once and share them with every run; a dropped
        connection reconnects on the next run. Closed by ``close()``."""
        if not self.mcp_servers:
            return
        async with self._mcp_lock:
            await self._mcp_manager.connect_all()
            for tool in self._mcp_manager.get_tools():
                if self._registry.get(tool.name) is not tool:
                    if self._registry.get(tool.name) is not None:
                        self._registry.unregister(tool.name)
                    self._registry.register(tool, host=True)

    async def _drive_connected(
        self,
        ctx: RunContext,
        sink: Callable[[CoreEvent], None],
        cancellation_token: CancellationToken | None,
        stream_tokens: bool,
        kwargs: dict[str, Any],
        task: list[CoreMessage] | None = None,
    ) -> AgentResponse:
        def emit(event: CoreEvent) -> None:
            self.completion_bus.emit(event, ctx)
            sink(event)

        mw = MiddlewareContext(ctx=ctx, agent=self.name, emit=emit)

        await self.prepare()
        directory = self.workspace.materialize(ctx.user_id, ctx.session_id)
        if self.skills is not None:
            self.skills.materialize(directory)
        pending = ctx.runtime_state.shared_state.pop("reasoning_loop", {})
        loop_state = self.reasoning.LOOP_STATE_CLS()
        loop_state.apply_metrics(pending.get("metrics", {}))
        if pending.get("completion") is not None:
            loop_state.last_completion_decision = CompletionDecision.model_validate(
                pending["completion"]
            )
        if pending.get("last_result") is not None:
            loop_state.last_result = ChatCompletionResult.model_validate(
                pending["last_result"]
            )
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
                "workspace_dir": str(directory.workspace_dir),
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
            name=self.name,
            client=self.client,
            dispatcher=self.dispatcher,
            tool_context=context,
            completion_bus=self.completion_bus,
            middleware_chain=self._middleware,
            compaction=self.compaction,
            # Custom clients may not declare a window: 0 = unknown.
            max_context_tokens=getattr(self.client.config, "max_context_window", 0) or 0,
            memory=await self._memory_for(ctx),
        )
        stopped: StopRun | None = None
        try:
            await self._middleware.run_start(mw, task)
            async with aclosing(
                reasoning.execute_reasoning_loop(
                    ctx=ctx,
                    prompts=await self._prompts(ctx),
                    loop_state=loop_state,
                    stream_tokens=stream_tokens,
                    cancellation_token=cancellation_token,
                    output_format=self.output_format,
                    **kwargs,
                )
            ) as stream:
                async for event in stream:
                    emit(event)
        except StopRun as stop:
            # A middleware ended the turn (budget, policy...): a clean finish.
            stopped = stop
            known = get_args(FinishReason)
            loop_state.finish_reason = stop.finish_reason if stop.finish_reason in known else "stopped"
        except BaseException as error:
            # Crash or cancel: middlewares still close what they opened.
            await self._middleware.run_error(mw, error)
            raise
        duration_ms = pending.get("duration_ms", 0) + int(
            (time.monotonic() - started) * 1000
        )
        if loop_state.finish_reason in (
            "approval_needed",
            "input_needed",
            "waiting",
            "tool_denied",
        ):
            ctx.runtime_state.shared_state["reasoning_loop"] = {
                "metrics": loop_state.metrics_snapshot(),
                "guard_state": getattr(loop_state, "guard_state", {}),
                "duration_ms": duration_ms,
                "completion": (
                    loop_state.last_completion_decision.model_dump(mode="json")
                    if loop_state.last_completion_decision is not None
                    else None
                ),
                "last_result": (
                    loop_state.last_result.model_dump(mode="json")
                    if loop_state.finish_reason == "waiting"
                    and loop_state.last_result is not None
                    else None
                ),
            }
        usage = Usage(
            duration_ms=duration_ms,
            retries=loop_state.retries,
            llm_calls=loop_state.llm_calls,
            attempts_to_call_api=loop_state.attempts_to_call_api,
            tool_calls=loop_state.tool_calls,
            tokens_input=loop_state.tokens_input,
            tokens_output=loop_state.tokens_output,
            tokens_cached=loop_state.tokens_cached,
        )
        response = AgentResponse(
            context=ctx,
            source=self.name,
            usage=usage,
            finish_reason=loop_state.finish_reason,
            completion=None if stopped else loop_state.last_completion_decision,
            stop_message=stopped.message if stopped else None,
        )
        await self._middleware.run_end(mw, response)
        return response

    async def run_stream_events(
        self,
        task: str | CoreMessage | list[CoreMessage] | None = None,
        run_context: RunContext | None = None,
        cancellation_token: CancellationToken | None = None,
        stream_tokens: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[CoreEvent | AgentResponse, None]:
        if self._closed:
            raise RuntimeError("Agent is closed")
        ctx = run_context if run_context is not None else RunContext()
        ctx.session_id = ctx.session_id or ctx.run_id
        key = (ctx.user_id, ctx.session_id)
        lock = self._session_locks.get(key)
        if lock is None:
            lock = self._session_locks[key] = asyncio.Lock()
        async with lock, self._tracking_run():
            if self._closed:
                raise RuntimeError("Agent is closed")
            if task is not None:
                if any(not r.is_consumed for r in ctx.tool_state.records.values()):
                    raise ValueError(
                        "Resolve pending tool calls before adding another task"
                    )
                # New task: drop last task's tool_state and gate evidence so
                # completion gates only see evidence from the task actually
                # in progress.
                ctx.tool_state.reset()
                ctx.completion_state.clear()
                # An unfinished plan belongs to the conversation (e.g. walked
                # step by step with the user); only a finished one is dropped.
                if ctx.plan is not None and not ctx.plan.has_unfinished_steps():
                    ctx.plan = None
                ctx.runtime_state.shared_state.pop("reasoning_loop", None)
                new_messages = (
                    [UserMessage(source=ctx.user_id, content=task)]
                    if isinstance(task, str)
                    else task
                    if isinstance(task, list)
                    else [task]
                )
                ctx.messages.extend(new_messages)
            queue = asyncio.Queue()
            sentinel = object()

            async def produce() -> AgentResponse:
                try:
                    return await self._drive(
                        ctx, queue.put_nowait, cancellation_token, stream_tokens, kwargs,
                        task=new_messages if task is not None else None,
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

    async def run(self, *args: Any, **kwargs: Any) -> AgentResponse:
        async with aclosing(self.run_stream_events(*args, **kwargs)) as stream:
            async for item in stream:
                if isinstance(item, AgentResponse):
                    return item
        raise RuntimeError("Agent produced no response")

    async def run_stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
        kwargs["stream_tokens"] = True
        async with aclosing(self.run_stream_events(*args, **kwargs)) as stream:
            async for item in stream:
                if isinstance(item, ModelStreamChunkEvent) and item.chunk:
                    yield item.chunk

    async def resume(self, run_context: RunContext, **kwargs: Any) -> AgentResponse:
        """Continue after callers apply approvals or answers to tool records."""
        return await self.run(run_context=run_context, **kwargs)

    async def resume_stream_events(
        self,
        run_context: RunContext,
        **kwargs: Any,
    ) -> AsyncGenerator[CoreEvent | AgentResponse, None]:
        async with aclosing(
            self.run_stream_events(run_context=run_context, **kwargs)
        ) as stream:
            async for item in stream:
                yield item

    async def resume_stream(
        self,
        run_context: RunContext,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        async with aclosing(
            self.run_stream(run_context=run_context, **kwargs)
        ) as stream:
            async for chunk in stream:
                yield chunk

    @contextlib.asynccontextmanager
    async def _tracking_run(self) -> AsyncGenerator[None, None]:
        """Count active runs so close() waits for them."""
        self._active_runs += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._active_runs -= 1
            if self._active_runs == 0:
                self._idle.set()

    async def close(self) -> None:
        """Refuse new runs, wait for the active ones, then release MCP
        connections and execution sessions."""
        self._closed = True
        await self._idle.wait()
        await self._mcp_manager.disconnect_all()
        await self._manager.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
