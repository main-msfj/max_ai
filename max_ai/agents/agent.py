"""The agent: where the harness decides how a run is done."""

from __future__ import annotations

import asyncio
import json
import typing as t
import weakref
from collections.abc import AsyncGenerator, Sequence
from datetime import UTC, datetime

from pydantic import BaseModel

from ..base.clients import CoreChatCompletionClient
from ..base.component import ComponentBase
from ..base.knowledge import CoreKnowledgeRegistry
from ..base.layer import CoreLayer
from ..base.loop import CoreLoop, LoopState, Stop
from ..base.middleware import CoreMiddleware
from ..base.sandbox import SandboxCore
from ..base.skills import SkillsCore
from ..base.tools import CoreTool
from ..capabilities.loop import ReactLoop
from ..capabilities.mcp import MCPClientManager
from ..capabilities.mcp._model import (
    HTTPServerConfig,
    MCPServerConfig,
    StdioMCPServerConfig,
    serialize_mcp_servers,
)
from ..capabilities.memory import Memory
from ..capabilities.stacks import (
    STANDARD_ORDER,
    AgentPolicyLayer,
    KnowledgeLayer,
    MemoryFilesLayer,
    PromptBuilder,
    RenderingLayer,
    SessionLayer,
    SkillsLayer,
    TaskAnalysisLayer,
    ToolsLayer,
    WorkspaceLayer,
)
from ..capabilities.tools.call_tool import CallToolTool
from ..capabilities.tools.search_tool import SearchToolsTool
from ..capabilities.tools.skill import ReadSkillTool
from ..config import setting
from ..core.event_type import CoreEvent, FatalErrorEvent, ModelStreamChunkEvent
from ..core.harness.hooks import HarnessLoopHook, HarnessToolHook
from ..core.harness.permissions import Permissions
from ..core.harness.rules import Harness
from ..core.middleware.chain import MiddlewareChain
from ..core.model.agent import AgentSpec
from ..core.model.json_schema import model_from_schema, schema_of
from ..core.tool import ToolDispatcher, ToolRegistry
from ..errors.middleware import StopRun
from ..types.run_context import RunContext

AnyClient = CoreChatCompletionClient[t.Any]
AnyLoop = CoreLoop[t.Any]
ServerConfig = StdioMCPServerConfig | HTTPServerConfig


class Agent(ComponentBase[AgentSpec]):
    """A model with its tools, skills, memory and sandbox, run by a loop.

    Every piece is given already built and must come from its Core class.
    ``serialize()`` turns the agent into data and ``Agent.deserialize()``
    builds it again.
    """

    component_type = "agent"
    component_schema = AgentSpec
    component_provider_override = "maxai.agents.Agent"

    NOT_STORABLE: t.ClassVar[str] = "Cannot serialize {what} {name!r}: {hint}"
    TOOL_HINT: t.ClassVar[str] = (
        "tools written as Python functions are code and cannot be stored; "
        "expose them through an MCP server (mcp=[...]) instead."
    )
    COMPONENT_HINT: t.ClassVar[str] = "give it a component_schema, _to_config and _from_config."
    CLOSED: t.ClassVar[str] = "The agent is closed"
    SANDBOX_LOCK: t.ClassVar[str] = "\0sandbox"  # never a session id: those are validated

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        client: AnyClient,
        *,
        toolset: Sequence[CoreTool] | None = None,
        mcp: Sequence[ServerConfig] | None = None,
        memory: Memory | None = None,
        knowledge: Sequence[CoreKnowledgeRegistry] | None = None,
        skills: SkillsCore | None = None,
        reasoning: AnyLoop | None = None,
        middlewares: Sequence[CoreMiddleware] | None = None,
        output_format: type[BaseModel] | None = None,
        prompt_layers: Sequence[CoreLayer] | None = None,
        sandbox: SandboxCore | None = None,
    ) -> None:
        # 1. Keep what was given; the loop is the only piece with a default.
        self.name = name
        self.description = description
        self.instructions = instructions
        self.client = client
        self.toolset = list(toolset or ())
        self.mcp = list(mcp or ())
        self.memory = memory
        self.knowledge = list(knowledge or ())
        self.skills = skills
        # The loop the harness gives when none is passed compacts with a summary.
        self.reasoning: AnyLoop = (
            reasoning if reasoning is not None else ReactLoop(compaction=Harness.compaction_rules())
        )
        self.middlewares = list(middlewares or ())
        self.output_format = output_format
        self.prompt_layers = list(prompt_layers or ())
        self.sandbox = sandbox

        self._validate()
        self._run_harness_validation()
        # What the runs share: MCP connections and one lock per session.
        self._mcp = MCPClientManager()
        for server in self.mcp:
            self._mcp.add_server(server)
        self._mcp_lock = asyncio.Lock()
        self._locks: weakref.WeakValueDictionary[tuple[str, str], asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._closed = False
        self._boxes: dict[str, SandboxCore] = {}  # one sandbox per user, closed by close()
        # close() waits for the runs that are still going.
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()

    # -------- CHECKS WHEN IT IS BUILT ----------------------------------------

    def _validate(self) -> None:
        """Every piece must come from its Core class; the error names the one that does not."""
        self._validate_directives()
        self._validate_capabilities()
        self._validate_runtime()

    def _validate_directives(self) -> None:
        """Who the agent is and how it answers: name, description, instructions, output format."""
        # 1. The three texts.
        for what in ("name", "description", "instructions"):
            if not isinstance(getattr(self, what), str):
                raise TypeError(f"{what} must be text")
        if not self.name.strip():
            raise ValueError("name cannot be empty")
        # 2. The answer's format is a class, not an object.
        form = t.cast("object", self.output_format)
        if form is not None and not (isinstance(form, type) and issubclass(form, BaseModel)):
            raise TypeError("output_format must be a pydantic BaseModel class")

    def _validate_capabilities(self) -> None:
        """What the agent can do: tools, MCP, memory, knowledge, skills and prompt layers."""
        # 1. The ones that are a single object.
        if self.memory is not None:
            self._check(self.memory, Memory, "memory")
        if self.skills is not None:
            self._check(self.skills, SkillsCore, "skills")
        # 2. The ones that are a list: each element is checked.
        lists: tuple[tuple[Sequence[object], type, str], ...] = (
            (self.toolset, CoreTool, "toolset"),
            (self.mcp, MCPServerConfig, "mcp"),
            (self.knowledge, CoreKnowledgeRegistry, "knowledge"),
            (self.prompt_layers, CoreLayer, "prompt_layers"),
        )
        for values, kind, what in lists:
            for value in values:
                self._check(value, kind, what)

    def _validate_runtime(self) -> None:
        """What runs the agent: client, loop, middlewares and sandbox."""
        # 1. The model and the loop are always there.
        self._check(self.client, CoreChatCompletionClient, "client")
        self._check(self.reasoning, CoreLoop, "reasoning")
        # 2. The sandbox is optional; middlewares are a list.
        if self.sandbox is not None:
            self._check(self.sandbox, SandboxCore, "sandbox")
        for value in self.middlewares:
            self._check(value, CoreMiddleware, "middlewares")

    @staticmethod
    def _check(value: object, kind: type, what: str) -> None:
        """Raise when ``value`` is not a ``kind``, saying what was given."""
        if not isinstance(value, kind):
            given = type(value).__name__
            raise TypeError(f"{what} must be a {kind.__name__}; got {given}")

    def _run_harness_validation(self) -> None:
        """Apply the harness's rules: an error names the rule that is broken."""
        # 1. The texts and the tools the developer wrote.
        Harness.description_rules(self.description)
        Harness.instruction_rules(self.instructions)
        Harness.custom_prompt_rules(self.prompt_layers)
        Harness.tool_rules(self.toolset)
        # 2. The model's limits and the sandbox's time are the harness's.
        Harness.model_rules(self.client.config)
        if self.sandbox is not None:
            Harness.sandbox_rules()
        # 3. What the harness adds: tracing and its own tools.
        self.middlewares = Harness.middleware_rules(self.middlewares, self.name)
        self._chain = MiddlewareChain(self.middlewares)
        self.tools = Harness.tool_combination(self.toolset, self.sandbox, self.skills)

    # -------- WHAT THE APPLICATION CALLS -------------------------------------

    async def run(
        self,
        task: str | None = None,
        ctx: RunContext | None = None,
    ) -> LoopState:
        """Run one turn to its end. The answer is ``state.message``, why it
        ended ``state.finish_reason``; the conversation stays in ``ctx``."""
        state = LoopState()
        async for _ in self.run_stream_events(task, ctx, state=state):
            pass
        return state

    async def run_stream(
        self,
        task: str | None = None,
        ctx: RunContext | None = None,
        *,
        state: LoopState | None = None,
    ) -> AsyncGenerator[str]:
        """Run one turn and give out only the text of the answer, as the model writes it."""
        events = self.run_stream_events(task, ctx, state=state, stream=True)
        async for event in events:
            if isinstance(event, ModelStreamChunkEvent) and event.chunk:
                yield event.chunk

    async def run_stream_events(
        self,
        task: str | None = None,
        ctx: RunContext | None = None,
        *,
        state: LoopState | None = None,
        stream: bool = False,
    ) -> AsyncGenerator[CoreEvent]:
        """Run one turn and give out what happens. ``task`` is what the user
        says now; None goes on with ``ctx`` (an approval was answered).
        ``state`` is filled with how the turn ended;"""
        if self._closed:
            raise RuntimeError(self.CLOSED)
        # A run that started is counted until it ends, so close() can wait for it.
        self._active += 1
        self._idle.clear()
        try:
            async for item in self._turn(task, ctx, state or LoopState(), stream):
                yield item
        finally:
            self._active -= 1
            if self._active == 0:
                self._idle.set()

    async def approve(self, ctx: RunContext, call_id: str, *, always: bool = False) -> bool:
        """Approve a call that waits for the user; call ``run(None, ctx)``
        after. ``always`` keeps a rule in the user's settings so it is not
        asked again; a dangerous call is approved this once only. Whether a
        rule was kept."""
        # 1. Approve this call.
        record = ctx.tool_state.records[call_id]
        record.approve()
        # 2. Keep the rule, unless the harness always asks for this call.
        if not always or not Permissions.can_be_kept(record.tool_name, record.parameters):
            return False
        # 3. In the user's sandbox, made ready if no turn has run yet.
        box = await self._sandbox(ctx)
        if box is None:
            return False
        for rule in Permissions.rules_for(record.tool_name, record.parameters):
            await box.add_permission("allow", rule)
        return True

    async def close(self) -> None:
        """Refuse new runs, wait for the ones still going, then release the
        MCP connections, the sandbox and the client."""
        first = not self._closed
        self._closed = True
        await self._idle.wait()
        if not first:
            return
        await self._mcp.disconnect_all()
        # The users' sandboxes, not self.sandbox: that one has no user.
        for box in self._boxes.values():
            await box.close()
        await self.client.aclose()

    # -------- ONE TURN, STEP BY STEP -----------------------------------------

    async def _turn(
        self,
        task: str | None,
        ctx: RunContext | None,
        state: LoopState,
        stream: bool,
    ) -> AsyncGenerator[CoreEvent]:
        """One turn over ``ctx``: each step below is one small method."""
        ctx = ctx if ctx is not None else RunContext()
        ctx.session_id = ctx.session_id or ctx.run_id
        # 1. Messages of one session wait for each other; other sessions overlap.
        async with self._lock(ctx):
            # 2. What the user says enters as steering: the loop puts it in the
            #    conversation and closes a question left without answer.
            if task is not None:
                ctx.inbox.append(task)
            # 3. The turn, with the middlewares around all of it.
            try:
                async for event in self._chain.turn(ctx, lambda: self._steps(ctx, state, stream)):
                    yield event
            except StopRun as error:
                # A middleware ended it: nothing is kept for the next turn.
                ctx.inbox.clear()
                ToolDispatcher.close_interrupted(ctx, f"Interrupted by the system: {error.label}.")
                yield self.reasoning.end(
                    ctx, state, Stop(reason=error.finish_reason, answer=error.message)
                )
            except Exception:
                # It failed before the loop took what the user said: a new try sends it again.
                if task is not None and ctx.inbox and ctx.inbox[-1] == task:
                    ctx.inbox.pop()
                raise

    async def _steps(self, ctx: RunContext, state: LoopState, stream: bool) -> AsyncGenerator[CoreEvent]:
        """What a turn does: get ready, then the loop. A failure while
        getting ready ends like one in the loop: an event, then the error."""
        try:
            # 1. Where the tools run, which tools there are and who watches them.
            box = await self._sandbox(ctx)
            registry = await self._registry()
            dispatcher = await self._dispatcher(registry, box)
            loop_hook = HarnessLoopHook(memory=self.memory, sandbox=box, mcp=self._mcp)
            # 2. The prompt, with the memory as it is now.
            if ctx.memory_index is None:
                await loop_hook.refresh_memory(ctx)
            prompt = await self._prompt(ctx, box, registry)
            # 3. The harness's limits, and the window known before the model.
            counter = self.reasoning.compaction.counter
            Harness.tools_total_rules(registry.tokens(counter))
            Harness.prompt_rules(counter.count_text(prompt), self.client.config.max_context_window)
            self.reasoning.measure(ctx, prompt, self.client, dispatcher)
        except Exception as error:
            state.finish_reason = "error"
            yield FatalErrorEvent(
                source=self.name, error_message=str(error), error_type=type(error).__name__
            )
            raise
        # 4. The loop.
        events = self.reasoning.run(
            ctx, prompt, self.client, dispatcher, stream=stream, hook=loop_hook,
            state=state, output_format=self.output_format,
            retry=Harness.retry_rules(self.client.retry),
        )
        async for event in events:
            yield event

    def _lock(self, ctx: RunContext, what: str = "") -> asyncio.Lock:
        """The lock of this user's session, or of ``what`` for the user
        (their sandbox); made the first time it is asked for."""
        key = (ctx.user_id, what or ctx.session_id or ctx.run_id)
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        return lock

    async def _sandbox(self, ctx: RunContext) -> SandboxCore | None:
        """The user's sandbox, ready and with the skills installed; None
        without one. Made ready once per user: later turns get the same one."""
        if self.sandbox is None:
            return None
        async with self._lock(ctx, self.SANDBOX_LOCK):
            box = self._boxes.get(ctx.user_id)
            if box is None:
                variables = self.skills.variables() if self.skills is not None else None
                box = self.sandbox.bind(ctx.user_id, variables)
                await box.prepare()
                if self.skills is not None:
                    await self.skills.sync(box)
                # Kept only once it is ready: a failure tries again next turn.
                self._boxes[ctx.user_id] = box
        return box

    async def _registry(self) -> ToolRegistry:
        """The tools of a run: the agent's, the MCP servers' and what reaches them."""
        tools = list(self.tools)
        # 1. MCP servers connect once; one that is down is skipped.
        if self.mcp:
            async with self._mcp_lock:
                await self._mcp.connect_all()
            tools += self._mcp.get_tools()
        if self.skills is not None:
            tools.append(ReadSkillTool())
        # 2. A tool with a name already taken is left out: the first one wins.
        registry = ToolRegistry()
        for tool in tools:
            if registry.get(tool.name) is None:
                registry.register(tool)
        # 3. Deferred tools are found and called through these two.
        if registry.deferred_tools():
            for extra in (SearchToolsTool(registry), CallToolTool(registry)):
                if registry.get(extra.name) is None:
                    registry.register(extra)
        return registry

    async def _dispatcher(self, registry: ToolRegistry, box: SandboxCore | None) -> ToolDispatcher:
        """Who runs the tools of a run, watched by the harness's hook with the user's rules."""
        rules = Permissions(await box.permissions()) if box is not None else None
        hook = HarnessToolHook(memory=self.memory, agent=self.name, permissions=rules)
        return ToolDispatcher(
            registry, box, hook=hook, middleware=self._chain, timeout=setting.tool_timeout_seconds
        )

    async def _prompt(self, ctx: RunContext, box: SandboxCore | None, registry: ToolRegistry) -> str:
        """The system prompt: the native layers always, then the developer's,
        in the standard order. Name, description and instructions go in the
        agent policy layer."""
        # 1. The native layers; each optional one only when the agent has its piece.
        layers: list[CoreLayer] = [
            AgentPolicyLayer(), TaskAnalysisLayer(), RenderingLayer(), ToolsLayer(), SessionLayer(),
        ]
        variables: dict[str, t.Any] = {
            "name": self.name, "description": self.description,
            "instructions": self.instructions,
            "current_date": datetime.now(UTC).date().isoformat(),
        }
        if box is not None:
            layers.append(WorkspaceLayer())
            variables.update(box.session_paths(ctx.session_id or ctx.run_id))
            if self.skills is not None:
                layers.append(SkillsLayer())
                installed = await box.installed_skills()
                variables["skills"] = [
                    {"name": name, "description": installed[name]["description"]}
                    for name in self.skills.skills
                    if name in installed
                ]
            if self.memory is not None:
                layers.append(MemoryFilesLayer())
                variables["memory"] = self.memory.block(box, ctx.memory_index or "")
        if self.knowledge:
            layers.append(KnowledgeLayer())
        # 2. The developer's layers join them, all in the standard order.
        builder = PromptBuilder([*layers, *self.prompt_layers], registry, order=STANDARD_ORDER)
        return builder.build(variables)

    # -------- STORED AND BUILT AGAIN -----------------------------------------

    def _to_config(self) -> AgentSpec:
        dump = self._dump
        form = self.output_format
        return AgentSpec(
            name=self.name,
            description=self.description,
            instructions=self.instructions,
            client=dump(self.client, "client"),
            reasoning=dump(self.reasoning, "reasoning"),
            toolset=[dump(tool, "tool") for tool in self.toolset],
            mcp=json.loads(serialize_mcp_servers(self.mcp)),
            # Memory has no settings of its own beyond this one.
            memory={"read_only": self.memory.read_only} if self.memory is not None else None,
            knowledge=[dump(source, "knowledge") for source in self.knowledge],
            skills=dump(self.skills, "skills") if self.skills is not None else None,
            middlewares=[dump(item, "middleware") for item in self.middlewares],
            output_format=schema_of(form) if form is not None else None,
            prompt_layers=[dump(layer, "prompt layer") for layer in self.prompt_layers],
            sandbox=dump(self.sandbox, "sandbox") if self.sandbox is not None else None,
        )

    @classmethod
    def _from_config(cls, config: AgentSpec) -> t.Self:
        load = cls._load
        form = config.output_format
        # What comes back is checked again by the constructor.
        return cls(
            name=config.name,
            description=config.description,
            instructions=config.instructions,
            client=load(config.client),
            reasoning=load(config.reasoning),
            toolset=[load(tool) for tool in config.toolset],
            mcp=list(config.mcp),
            memory=Memory(**config.memory) if config.memory is not None else None,
            knowledge=[load(source) for source in config.knowledge],
            skills=load(config.skills),
            middlewares=[load(item) for item in config.middlewares],
            output_format=model_from_schema(form) if form is not None else None,
            prompt_layers=[load(layer) for layer in config.prompt_layers],
            sandbox=load(config.sandbox),
        )

    @staticmethod
    def _dump(component: ComponentBase[t.Any], what: str) -> dict[str, t.Any]:
        """A component as storable data; a clear error when it cannot be stored."""
        try:
            return component.serialize().model_dump(exclude_none=True)
        except (NotImplementedError, AttributeError, TypeError) as error:
            name = getattr(component, "name", None) or type(component).__name__
            hint = Agent.TOOL_HINT if what == "tool" else Agent.COMPONENT_HINT
            raise TypeError(Agent.NOT_STORABLE.format(what=what, name=name, hint=hint)) from error

    @staticmethod
    def _load(data: dict[str, t.Any] | None) -> t.Any:
        """The component stored in ``data``; None when nothing was stored."""
        return ComponentBase[t.Any].deserialize(data) if data is not None else None
