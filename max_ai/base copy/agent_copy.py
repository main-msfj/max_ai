"""Main Class and Base for Agent Implementation"""

from __future__ import annotations

import asyncio
import logging
import time
import typing as t
from contextlib import aclosing

from pydantic import BaseModel

from ..base.middleware import CoreMiddleware
from ..compaction import SlidingWindowCompaction
from ..config import setting
from ..core.compaction import MemoryMaintenanceOutput
from ..core.event_type import (
    CompactionEvent,
    CoreEvent,
    ErrorEvent,
    ModelStreamChunkEvent,
    ReasoningIterationEvent,
)
from ..core.messages import CoreMessage, UserMessage
from ..core.models import AgentComponentConfig, AgentConfig
from ..environment.manager import EnvironmentFactory, EnvironmentManager
from ..errors.agent import AgentError
from ..executor.local import LocalExecutor
from ..executor.routing import RoutingExecutor
from ..loggers import ScopedLogger
from ..manager.capabilities import AgentCapabilities
from ..manager.stacks import PromptVariablesBuilder, build_default_stack
from ..reasoning.react_self_directed import ReActLoopSelfDirected
from ..stacks import CoreLayer
from ..termination import CancellationToken
from ..types.agent_response import AgentResponse
from ..types.completions import Usage
from ..types.run_context import RunContext
from ..types.stacks import PromptCtx, PromptLayerUsage
from .capability import CoreAgentCapabilities
from .clients import CoreChatCompletionClient
from .compaction import (
    CompactionResult,
    CoreCompaction,
    TokenCounter,
    client_max_output_tokens,
    live_message_budget_tokens,
    live_message_capacity_tokens,
    live_message_threshold_tokens,
)
from .component import ComponentBase, is_component_class
from .context import CoreLogBookRegistry
from .environment import Environment
from .executor import CoreExecutor
from .knowledge import CoreKnowledgeRegistry
from .memory import CoreMemoryRegistry, MemoryRecord
from .routines import CoreRoutineRegistry
from .skills import CoreSkillRegistry
from .tool_executor import ToolExecutor
from .tools import CoreTool
from .workspace import Workspace, WorkSpaceRegistry

if t.TYPE_CHECKING:
    from ..manager.stacks import LayerContainer
    from ..persistence.core import RunContextStore
    from .reasoning import BaseReasoning

RunYield = t.Union[CoreEvent, AgentResponse]
ConfigT = t.TypeVar("ConfigT", bound=BaseModel)
CapabilitiesT: t.TypeAlias = CoreAgentCapabilities[BaseModel]


# -------- LOGGER -----------------------------------------------------------
logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["Agent"])


class Agent(ComponentBase[BaseModel]):
    """
    Core interface for all MaxAI agents.

    Lifecycle:
      1. ``__init__`` — sync. Stores configuration, builds the
         ``AgentCapabilities`` (validating tool name uniqueness,
         priority_tools coherence, etc.) and the ``LayerContainer``
         (validating every layer's template against its declared
         contract). Anything broken at this stage raises immediately.
      2. ``await agent.prepare()`` — async. Loads skills via their
         registry, fetches the current memory snapshot and session
         summary, renders every layer of the prompt stack with the
         right variables, and stores the result.
      3. ``await agent.run(...)`` (or one of its streaming variants) —
         async. By the time the run starts, ``self.rendered_layers``
         is fully populated and every tool is registered.

    Three public APIs share a single engine:
      - ``run_stream_events`` — yields every CoreEvent produced during
        the run, terminated by a single AgentResponse. The engine.
      - ``run_stream`` — yields only assistant text chunks. Convenience
        for CLI / chat use.
      - ``run`` — consumes everything, returns the AgentResponse.
        Convenience for scripts / tests.
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        client: CoreChatCompletionClient,
        toolset: t.Sequence[CoreTool | t.Callable[..., t.Any]] | None = None,
        memory: CoreMemoryRegistry | None = None,
        skills: CoreSkillRegistry | None = None,
        logbook: CoreLogBookRegistry | None = None,
        routines: CoreRoutineRegistry | None = None,
        knowledge: t.Sequence[CoreKnowledgeRegistry] | None = None,
        workspace: WorkSpaceRegistry | None = None,
        middlewares: t.Sequence[CoreMiddleware] | None = None,
        framework_layers: t.Sequence[CoreLayer] | None = None,
        executor: CoreExecutor | None = None,
        reasoning: BaseReasoning | None = None,
        compaction: CoreCompaction | None = None,
        output_format: t.Type[BaseModel] | None = None,
        priority_tools: list[str] | None = None,
        config: AgentConfig | None = None,
        store: "RunContextStore | None" = None,
        environment: EnvironmentFactory | None = None,
        environment_manager: EnvironmentManager | None = None,
    ) -> None:
        """
        Args:
            name: Unique identifier for the agent.
            description: External metadata for orchestrators or
                multi-agent discovery.
            instructions: Agent core instructions as a string of
                directives.
            client: LLM provider abstraction for API interactions.
            memory: Persistent user-fact backend.
            skills: Skill registry — resolved during ``prepare()``.
            logbook: Conversation-logbook registry (per session
                summaries + cross-session search).
            toolset: Executable functions available to the agent.
            routines: Repetitive task procedures discovered via tools.
            knowledge: Persistent sources of truth (RAG backends).
            middlewares: Logic hooks to intercept and process
                operations.
            framework_layers: Custom prompt layers to override or extend
                the default stack. Each entry replaces the default
                layer of the same concrete type. Unknown layer types
                are appended. Validation runs during construction —
                a broken layer raises before the agent is ever used.
            reasoning: Optional user-provided reasoning loop. Defaults
                to ReActLoopSelfDirected. Config-only at construction;
                runtime wiring is injected later via ``bind()``.
            compaction: Optional user-provided context compaction strategy.
                Defaults to SlidingWindowCompaction.
            executor: Optional user-provided execution strategy. Defaults to
                ``LocalExecutor`` when the agent has no skills, or to a
                ``RoutingExecutor`` (local tools in-process, runtime tools
                in a Docker sandbox) when skills are present.
            workspace Optional User-provided Workspace Registry. Default to
                ``LocalWorkspace``.
            output_format: Pydantic model for structured response.
                Forwarded to the client on every LLM call.
            priority_tools: Tool names the agent should strongly prefer
                when relevant. Not mandatory — relevance to the user's
                query remains the deciding factor.
            config: Optional ``AgentConfig`` for execution tuning. If
                None, system defaults are used for timeouts, retries,
                and loops.
            environment: Optional factory (workspace, user_id, conversation_id)
                returning an Environment. DockerEnvironment or functools.partial
                can be used. Containers start lazily when Bash executes.
            environment_manager: Optional application-owned manager shared across
                agents. Otherwise this agent owns its manager; call close().
            store: Optional ``RunContextStore``. When set, the run
                checkpoints its ``RunContext`` at safe points (task
                appended, before each reasoning iteration, on every
                pause and on the terminal response), keyed by
                ``ctx.run_id``. A failed save logs a warning and never
                kills the run.
        """
        self.name = self.require_type(name, str, "name")
        self.description = self.require_type(description, str, "description")
        self.instructions = self.require_type(instructions, str, "instructions")
        self.client = self.require_type(client, CoreChatCompletionClient, "client")
        self.config = self.require_type(config or AgentConfig(), AgentConfig, "config")

        raw_capabs = self._collect_capabilities(locals().values())
        self.capabilities = self.build_registries(raw_capabs, priority_tools, toolset)

        self.registries = self.capabilities

        self.executor = self.validate_executor_object(executor)
        self.workspace = self.validate_workspace_object(workspace)
        self.reasoning = reasoning
        self.store = store
        if environment is not None and not callable(environment):
            raise TypeError("environment must be an Environment factory, not an instance")
        self.environment_factory = environment
        self.environment_manager = environment_manager or EnvironmentManager()
        self._owns_environment_manager = environment_manager is None
        self._turn_lock = asyncio.Lock()
        # The reasoning loop currently running a turn. None when idle.
        self._active_reasoning: BaseReasoning | None = None
        self.compaction = self.validate_compaction_object(compaction)
        self.output_format = output_format
        self.middlewares = list(middlewares or [])
        self.prompt_stack: LayerContainer = build_default_stack(framework_layers)
        self._variables_builder: PromptVariablesBuilder = PromptVariablesBuilder(self)
        self._token_counter = TokenCounter()
        self._prepared: bool = False
        self._rendered_layers: dict[type[CoreLayer], str] = {}
        self._rendered_layer_usage: dict[str, PromptLayerUsage] = {}
        self._prompt_tokens: int = 0

    component_schema = AgentComponentConfig
    component_type = "agent"

    # -------- COMPONENT SERIALIZATION -----------------------------------------------------------
    def _to_config(self) -> AgentComponentConfig:
        return AgentComponentConfig(
            name=self.name,
            description=self.description,
            instructions=self.instructions,
            client=self.client.dump_component().model_dump(exclude_none=True),
            config=self.config,
            toolset=self._dump_components(self.capabilities.toolset, "toolset"),
            capabilities=self._dump_components(
                [
                    cap
                    for cap in self.capabilities.capabilities
                    if not isinstance(cap, WorkSpaceRegistry)
                ],
                "capabilities",
            ),
            workspace=self.workspace.dump_component().model_dump(exclude_none=True)
            if self._is_dumpable_component(self.workspace)
            else None,
            middlewares=self._dump_components(self.middlewares, "middlewares"),
            framework_layers=self._dump_components(
                list(self.prompt_stack), "framework_layers"
            ),
            executor=self.executor.dump_component().model_dump(exclude_none=True)
            if self._is_dumpable_component(self.executor)
            else None,
            output_format=self._type_ref(self.output_format),
            priority_tools=list(self.capabilities.priority_tools),
        )

    @classmethod
    def _from_config(cls, config: AgentComponentConfig) -> t.Self:
        client = ComponentBase.load_component(
            config.client, expected=CoreChatCompletionClient
        )
        toolset = [
            ComponentBase.load_component(item, expected=CoreTool)
            for item in config.toolset
        ]
        capabilities = [
            ComponentBase.load_component(item, expected=CoreAgentCapabilities)
            for item in config.capabilities
        ]
        workspace = (
            ComponentBase.load_component(config.workspace, expected=WorkSpaceRegistry)
            if config.workspace is not None
            else None
        )
        middlewares = [
            ComponentBase.load_component(item, expected=CoreMiddleware)
            for item in config.middlewares
        ]
        framework_layers = [
            ComponentBase.load_component(item, expected=CoreLayer)
            for item in config.framework_layers
        ]
        executor = (
            ComponentBase.load_component(config.executor, expected=CoreExecutor)
            if config.executor is not None
            else None
        )

        memory = next(
            (cap for cap in capabilities if isinstance(cap, CoreMemoryRegistry)), None
        )
        skills = next(
            (cap for cap in capabilities if isinstance(cap, CoreSkillRegistry)), None
        )
        logbook = next(
            (cap for cap in capabilities if isinstance(cap, CoreLogBookRegistry)), None
        )
        routines = next(
            (cap for cap in capabilities if isinstance(cap, CoreRoutineRegistry)), None
        )
        knowledge = [
            cap for cap in capabilities if isinstance(cap, CoreKnowledgeRegistry)
        ]

        return cls(
            name=config.name,
            description=config.description,
            instructions=config.instructions,
            client=client,
            toolset=toolset,
            memory=memory,
            skills=skills,
            logbook=logbook,
            routines=routines,
            knowledge=knowledge,
            workspace=workspace,
            middlewares=middlewares,
            framework_layers=framework_layers,
            executor=executor,
            output_format=cls._load_type_ref(config.output_format),
            priority_tools=config.priority_tools,
            config=config.config,
        )

    @staticmethod
    def _is_dumpable_component(value: t.Any) -> bool:
        return isinstance(value, ComponentBase) and is_component_class(type(value))

    @classmethod
    def _dump_components(
        cls, values: t.Iterable[t.Any], field: str
    ) -> list[dict[str, t.Any]]:
        dumped: list[dict[str, t.Any]] = []
        for value in values:
            if not cls._is_dumpable_component(value):
                raise TypeError(
                    f"Agent {field} contains non-serializable {type(value).__name__}."
                )
            dumped.append(value.dump_component().model_dump(exclude_none=True))
        return dumped

    @staticmethod
    def _type_ref(value: type[BaseModel] | None) -> str | None:
        from ..core.type_ref import type_ref

        return type_ref(value)

    @staticmethod
    def _load_type_ref(value: str | None) -> type[BaseModel] | None:
        from ..core.type_ref import load_type_ref

        return load_type_ref(value)

    # -------- LIFECYCLE -----------------------------------------------------------
    def build_registries(
        self,
        capabilities: t.Sequence[CapabilitiesT] | None = None,
        priority_tools: t.Sequence[str] | None = None,
        toolset: t.Sequence[CoreTool | t.Callable[..., t.Any]] | None = None,
    ) -> AgentCapabilities:
        return AgentCapabilities(
            capabilities=capabilities,
            priority_tools=priority_tools,
            toolset=toolset,
        )

    @classmethod
    def _collect_capabilities(cls, values: t.Iterable[t.Any]) -> list[CapabilitiesT]:
        """Collect capability objects without naming each concrete type."""
        collected: list[CapabilitiesT] = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, CoreAgentCapabilities):
                collected.append(value)
            elif isinstance(value, (list, tuple, set, frozenset)):
                collected.extend(cls._collect_capabilities(value))
        return collected

    async def prepare(self) -> None:
        """Hydrate async registries and render the prompt stack.

        Must be awaited once before any ``run*`` call (the engine does
        this automatically). Idempotent — calling twice is a no-op.

        Steps:
          1. Resolve async registries (loads skill metadata and
             revalidates the final tool catalog).
          2. For each layer in the prompt stack, ask the variables
             builder for the right inputs and render the layer.
          3. Store the rendered string keyed by layer type, so the
             client can later assemble the final system prompt in its
             provider-native order.
        """
        if self._prepared:
            return

        await self.registries.prepare()

        for layer in self.prompt_stack:
            await self._render_prompt_layer(layer)

        self._recalculate_prompt_tokens()
        self._prepared = True

    async def _render_prompt_layer(self, layer: "CoreLayer") -> None:
        variables = await self._variables_builder.collect(type(layer))
        try:
            rendered = layer.render(variables)
        except Exception as e:
            raise AgentError.layer_render_failed(
                layer_name=type(layer).__name__,
                error=e,
            ) from e
        self._rendered_layers[type(layer)] = rendered
        usage = PromptLayerUsage(
            layer_name=type(layer).__name__,
            chars=len(rendered),
            tokens=self._token_counter.count_text(rendered),
        )
        self._rendered_layer_usage[usage.layer_name] = usage

    def _recalculate_prompt_tokens(self) -> None:
        self._prompt_tokens = sum(
            usage.tokens for usage in self._rendered_layer_usage.values()
        )

    async def _refresh_dynamic_prompt_layers(self) -> None:
        dynamic_layer_names = {"MemoryLayer", "ContextLayer"}
        for layer in self.prompt_stack:
            if type(layer).__name__ in dynamic_layer_names:
                await self._render_prompt_layer(layer)
        self._recalculate_prompt_tokens()

    def _ensure_prepared(self) -> None:
        if not self._prepared:
            raise AgentError.not_prepared(agent_name=self.name)

    # -------- INTERNAL HELPERS -----------------------------------------------------------
    def _normalize_run_context(
        self,
        task: str | CoreMessage | list[CoreMessage] | None,
        run_context: RunContext | None,
    ) -> RunContext:
        ctx = run_context if run_context is not None else RunContext()
        if task is None:
            return ctx

        if isinstance(task, str):
            ctx.messages.append(UserMessage(source="user", content=task))
            return ctx

        if isinstance(task, CoreMessage):
            ctx.messages.append(task)
            return ctx

        ctx.messages.extend(task)
        return ctx

    def _build_prompt_ctx(self) -> PromptCtx:
        return PromptCtx(
            stack=self.prompt_stack,
            variables={},
            rendered_layers=self.rendered_layers,
            layer_usage=self.rendered_layer_usage,
            prompt_tokens=self.prompt_tokens,
        )

    def _compaction_event(
        self,
        *,
        phase: t.Literal["start", "end"],
        max_context_tokens: int,
        result: CompactionResult | None = None,
        total_token_count: int = 0,
        context_summary_persisted: bool = False,
        context_summary_session_id: str | None = None,
        prompt_tokens: int | None = None,
    ) -> CompactionEvent:
        result_changed = bool(result and result.changed)
        return CompactionEvent(
            source=self.name,
            phase=phase,
            strategy=type(self.compaction).__name__,
            changed=result_changed,
            old_message_count=len(result.old_messages) if result else 0,
            recent_message_count=len(result.recent_messages) if result else 0,
            old_token_count=result.old_token_count if result else 0,
            recent_token_count=result.recent_token_count if result else 0,
            total_token_count=result.total_token_count if result else total_token_count,
            live_message_threshold_tokens=live_message_threshold_tokens(
                max_context_tokens,
                max_output_tokens=client_max_output_tokens(self.client),
                prompt_tokens=prompt_tokens,
            ),
            live_message_budget_tokens=live_message_budget_tokens(
                live_message_capacity_tokens(
                    max_context_tokens,
                    max_output_tokens=client_max_output_tokens(self.client),
                    prompt_tokens=prompt_tokens,
                )
            ),
            summary=result.summary if result else None,
            context_summary_persisted=context_summary_persisted,
            context_summary_session_id=context_summary_session_id,
        )

    async def _apply_compaction(
        self,
        ctx: RunContext,
        prompts: PromptCtx,
    ) -> CompactionEvent | None:
        max_context_tokens = getattr(self.client.config, "max_context_window", 0) or 0
        if max_context_tokens <= 0:
            return None

        result = await self.compaction.compact(
            ctx=ctx,
            prompts=prompts,
            max_context_tokens=max_context_tokens,
            client=self.client,
        )
        # Strategies are pure w.r.t. ctx.messages — the caller applies.
        # Evicted messages are gone from the agent's state by design (the
        # model gets the summary); a UI that wants to keep showing them
        # must keep its own display transcript (see max_ai/ui/server.py).
        if result.changed:
            ctx.messages[:] = result.recent_messages
        (
            context_persisted,
            context_session_id,
        ) = await self._persist_context_after_compaction(
            ctx=ctx,
            result=result,
        )
        await self._maintain_memory_after_compaction(
            ctx=ctx, prompts=prompts, result=result
        )

        return self._compaction_event(
            phase="end",
            max_context_tokens=max_context_tokens,
            result=result,
            context_summary_persisted=context_persisted,
            context_summary_session_id=context_session_id,
            prompt_tokens=prompts.prompt_tokens or None,
        )

    async def _persist_context_after_compaction(
        self,
        *,
        ctx: RunContext,
        result: CompactionResult,
    ) -> tuple[bool, str | None]:
        logbook = self.registries.logbook
        session_id = ctx.session_id or getattr(logbook, "session_id", None)
        if logbook is None or not result.changed or not result.summary:
            return False, session_id

        upsert_summary = getattr(logbook, "upsert_summary", None)
        if not callable(upsert_summary):
            return False, session_id

        if not session_id:
            return False, None

        try:
            await logbook._ensure_connected()
            await upsert_summary(
                session_id=session_id,
                summary=result.summary,
                metadata={
                    "source": "compaction",
                    "strategy": type(self.compaction).__name__,
                    "old_message_count": len(result.old_messages),
                    "recent_message_count": len(result.recent_messages),
                    "old_token_count": result.old_token_count,
                    "recent_token_count": result.recent_token_count,
                    "total_token_count": result.total_token_count,
                },  # type: ignore[dict-item]
            )
            return True, session_id
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Context summary persistence during compaction failed",
                agent_name=self.name,
                session_id=session_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return False, session_id

    async def _maintain_memory_after_compaction(
        self,
        *,
        ctx: RunContext,
        prompts: PromptCtx,
        result: CompactionResult,
    ) -> None:
        memory = self.registries.memory
        has_compacted_content = bool(result.old_messages or result.summary)
        if memory is None or not has_compacted_content:
            return

        try:
            await memory._ensure_connected()
            facts = await memory.list_facts()
            task = self._memory_maintenance_task(
                facts=facts,
                messages=result.old_messages,
                summary=result.summary,
            )
            maintenance_ctx = RunContext(
                user_id=ctx.user_id,
                session_id=ctx.session_id,
                messages=[UserMessage(source="memory-maintenance", content=task)],
            )
            maintenance_prompts = PromptCtx.model_construct(
                stack=None,
                variables={},
                rendered_layers={},
                layer_usage={},
                prompt_tokens=0,
            )
            maintenance = await self.client.run(
                ctx=maintenance_ctx,
                prompts=maintenance_prompts,
                tools=None,
                output_format=MemoryMaintenanceOutput,
                stream=False,
                max_tokens=setting.compaction_summary_budget_tokens,
            )
            structured = maintenance.message.structured_output
            if not isinstance(structured, MemoryMaintenanceOutput):
                raise TypeError(
                    f"expected MemoryMaintenanceOutput, got {type(structured).__name__}"
                )

            applied_updates = 0
            for update in structured.updates:
                key = update.key.strip()
                category = update.category.strip()
                content = update.content.strip()
                if not key or not category or not content:
                    continue
                await memory.upsert(
                    MemoryRecord(
                        key=key, category=category, content=content, source="compaction"
                    )
                )
                applied_updates += 1
            if applied_updates:
                await self._refresh_memory_prompt_layer(prompts)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Memory maintenance during compaction failed",
                agent_name=self.name,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    async def _refresh_memory_prompt_layer(self, prompts: PromptCtx) -> None:
        for layer in self.prompt_stack:
            if type(layer).__name__ != "MemoryLayer":
                continue
            await self._render_prompt_layer(layer)
            prompts.rendered_layers[type(layer)] = self._rendered_layers[type(layer)]
            prompts.layer_usage = self.rendered_layer_usage
            prompts.prompt_tokens = self.prompt_tokens
            return

    def _memory_maintenance_task(
        self,
        *,
        facts: list[t.Any],
        messages: list[CoreMessage],
        summary: str | None,
    ) -> str:
        current_facts = (
            "\n".join(
                f"- key={getattr(fact, 'key', None) or getattr(fact, 'category', 'unknown')}; "
                f"category={getattr(fact, 'category', 'unknown')}; "
                f"content={getattr(fact, 'content', '')}"
                for fact in facts
            )
            or "No stored memory facts."
        )
        transcript = self._memory_messages_transcript(messages)
        compacted_summary = summary or "No compaction summary was produced."
        return (
            "You are maintaining durable user memory during context compaction.\n"
            "Review every stored memory fact, the compaction summary, and the "
            "conversation messages that are about to be compacted. Return only "
            "structured output.\n\n"
            "Rules:\n"
            "- Compare the compacted information against current memory facts before "
            "returning any update.\n"
            "- If the compacted information already matches memory, return an empty "
            "updates list and keep going.\n"
            "- Update existing facts when newer messages refine, correct, or add "
            "durable user-specific information.\n"
            "- Reuse an existing key whenever the new information belongs to that fact.\n"
            "- Create a new key only for durable information that does not fit an "
            "existing fact.\n"
            "- Do not include transient conversation details, tool mechanics, or assistant claims.\n"
            "- Do not delete facts; deletion requires explicit user approval elsewhere.\n"
            "- If nothing should change, return an empty updates list.\n\n"
            f"Current memory facts:\n{current_facts}\n\n"
            f"Compaction summary:\n{compacted_summary}\n\n"
            f"Messages to learn from:\n{transcript}"
        )

    def _memory_messages_transcript(self, messages: list[CoreMessage]) -> str:
        rows: list[str] = []
        for message in messages:
            role = getattr(message, "role", "message")
            source = getattr(message, "source", "unknown")
            text = (
                message.text()
                if callable(getattr(message, "text", None))
                else str(message)
            )
            if not text and getattr(message, "tool_calls", None):
                text = f"tool_calls={message.tool_calls}"
            rows.append(f"[{role}/{source}] {text}")
        return "\n".join(rows)

    def _build_reasoning(
        self,
        tool_executor: ToolExecutor,
    ) -> BaseReasoning:
        """Resolve config-time reasoning into a runtime-bound instance.

        Accepts None (default ReActLoopSelfDirected) or a user-provided
        BaseReasoning instance. The agent injects runtime dependencies via
        .bind().
        """
        reasoning = self.reasoning
        if reasoning is None:
            # Default to the self-directed ReAct loop: planning is native
            # but optional — the model calls update_plan only when it
            # decides a plan is worth having.
            reasoning = ReActLoopSelfDirected(
                max_loop_iterations=self.config.max_loop_iterations,
                max_connection_retries=self.config.max_connection_retries,
            )

        return reasoning.bind(
            name=self.name,
            client=self.client,
            tool_executor=tool_executor,
            middleware_chain=tool_executor.mw_chain,
            compaction=self.compaction,
            max_context_tokens=getattr(self.client.config, "max_context_window", 0)
            or 0,
        )

    def validate_compaction_object(
        self, compaction: CoreCompaction | None
    ) -> CoreCompaction:
        """Resolve Compaction Strategy"""
        if compaction is not None:
            return compaction
        return SlidingWindowCompaction()

    def validate_workspace_object(
        self, workspace: WorkSpaceRegistry | None
    ) -> WorkSpaceRegistry:
        """Resolve WorkSpace"""
        if workspace is not None:
            return workspace
        return Workspace()

    def validate_executor_object(self, executor: CoreExecutor | None) -> CoreExecutor:
        """Resolve the execution strategy.

        Defaults split execution by tool type:

        - No runtime tools → a plain ``LocalExecutor``; every tool runs
          in-process (developer-authored code, trusted accordingly).
        - Runtime tools present → a ``RoutingExecutor`` that keeps ordinary
          tools local and routes runtime tools (``CoreRuntimeTool``, i.e.
          ``bash`` and the skill scripts it runs) into a Docker sandbox.

        A user-supplied executor is always honored as-is.
        """
        if executor is not None:
            return executor

        local = LocalExecutor(default_timeout=self.config.tool_timeout)
        if not self.registries.requires_sandbox_executor:
            return local

        # Only agents with runtime tools pull in the Docker stack;
        # the global filesystem tools do not require Docker.
        from ..executor import DockerExecutor

        return RoutingExecutor(local=local, sandbox=DockerExecutor())

    def _validate_runtime_safety(self) -> None:
        """Validate that runtime tools have a sandboxed execution path.

        Runtime tools (``CoreRuntimeTool`` — e.g. ``bash``, and therefore
        the skill scripts the model runs through it) execute model-chosen
        commands and must not run in-process. They require a
        sandbox-capable executor. Ordinary tools are unaffected; they run
        locally regardless.

        Raises:
            AgentError: when skills are present but the resolved executor
                cannot isolate runtime tools.
        """
        if not self.registries.requires_sandbox_executor:
            return
        if not self._executor_can_sandbox(self.executor):
            raise AgentError.unsafe_local_executor_for_skills(self.name)

    @classmethod
    def _executor_can_sandbox(cls, executor: CoreExecutor) -> bool:
        """Return True if ``executor`` can isolate runtime tools.

        - ``RoutingExecutor`` qualifies iff it has a sandbox backend that
          itself qualifies.
        - A bare ``LocalExecutor`` cannot isolate model-driven commands.
        - Any other executor (Docker, a custom sandbox) is trusted to
          provide isolation.
        """
        if isinstance(executor, RoutingExecutor):
            return executor.sandbox is not None and cls._executor_can_sandbox(
                executor.sandbox
            )
        return not isinstance(executor, LocalExecutor)

    def _warn_if_capacity_starved(
        self, max_context_tokens: int, prompts: PromptCtx
    ) -> None:
        """Warn (once) when the config leaves almost no room for messages.

        window - prompt - max_output_tokens - safety margin is what the
        conversation actually lives in. When that is tiny, mid-loop
        compaction evicts the tool results the model is working from and
        it deterministically re-calls the same tools. This is a config
        bug, not a model bug — say so loudly instead of thrashing quietly.
        """
        if max_context_tokens <= 0 or getattr(self, "_capacity_warned", False):
            return
        reserved_output = client_max_output_tokens(self.client)
        capacity = live_message_capacity_tokens(
            max_context_tokens,
            max_output_tokens=reserved_output,
            prompt_tokens=prompts.prompt_tokens or None,
        )
        if capacity >= setting.compaction_capacity_warning_tokens:
            return
        self._capacity_warned = True
        log.warning(
            "Live-message capacity is critically low — compaction will "
            "evict the current turn's tool results and the model will "
            "re-call the same tools. Raise max_context_window or lower "
            "max_tokens.",
            agent_name=self.name,
            max_context_window=max_context_tokens,
            prompt_tokens=prompts.prompt_tokens or 0,
            reserved_output_tokens=reserved_output,
            live_message_capacity_tokens=capacity,
            recommended_minimum=setting.compaction_capacity_warning_tokens,
        )

    def _build_response(
        self,
        ctx: RunContext,
        loop_state: t.Any,
        start_time: float,
    ) -> AgentResponse:
        finish_reason = loop_state.finish_reason or "unknown"
        if finish_reason == "max_iterations_exceeded":
            finish_reason = "max_iterations"
        elif finish_reason == "tool_calls":
            finish_reason = "stop"
        elif finish_reason not in {
            "stop",
            "max_iterations",
            "approval_needed",
            "input_needed",
            "tool_direct_return",
            "no_result",
            "error",
            "cancelled",
        }:
            finish_reason = "error"

        usage = Usage(
            duration_ms=int((time.monotonic() - start_time) * 1000),
            llm_calls=loop_state.llm_calls,
            tool_calls=loop_state.tool_calls,
            attempts_to_call_api=loop_state.attempts_to_call_api,
            retries=loop_state.retries,
            tokens_input=loop_state.tokens_input,
            tokens_output=loop_state.tokens_output,
            tokens_cached=loop_state.tokens_cached,
        )
        return AgentResponse(
            context=ctx,
            source=self.name,
            usage=usage,
            finish_reason=t.cast(t.Any, finish_reason),
        )

    # -------- PUBLIC API — STREAMING ENGINE -----------------------------------------------------------
    async def run_stream_events(
        self, task: str | CoreMessage | list[CoreMessage] | None = None,
        run_context: RunContext | None = None,
        cancellation_token: CancellationToken | None = None,
        stream_tokens: bool = False, **kwargs: t.Any,
    ) -> t.AsyncGenerator[RunYield, None]:
        """Serialize this agent's mutable turn state and lease its environment."""
        response: AgentResponse | None = None
        async with self._turn_lock:
            if cancellation_token is not None and cancellation_token.is_cancelled():
                raise asyncio.CancelledError
            ctx = self._normalize_run_context(task, run_context)
            if not ctx.session_id:
                ctx.session_id = ctx.run_id

            async def run(env):
                async with aclosing(self._run_stream_events(
                    run_context=ctx, cancellation_token=cancellation_token,
                    stream_tokens=stream_tokens, _environment=env, **kwargs,
                )) as stream:
                    async for item in stream:
                        yield item

            if self.environment_factory is None:
                async with aclosing(run(None)) as stream:
                    async for item in stream:
                        if isinstance(item, AgentResponse):
                            response = item
                        else:
                            yield item
            else:
                async with self.environment_manager.acquire(
                    self.environment_factory, self.workspace, ctx.user_id, ctx.session_id,
                ) as env:
                    async with aclosing(run(env)) as stream:
                        async for item in stream:
                            if isinstance(item, AgentResponse):
                                response = item
                            else:
                                yield item
        # Release the lease before publishing the terminal response.
        if response is not None:
            yield response

    async def close(self) -> None:
        """Close owned environments. Shared managers are closed by the application."""
        if self._owns_environment_manager:
            await self.environment_manager.close()

    async def _run_stream_events(
        self,
        task: str | CoreMessage | list[CoreMessage] | None = None,
        run_context: RunContext | None = None,
        cancellation_token: CancellationToken | None = None,
        stream_tokens: bool = False,
        _environment: Environment | None = None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[RunYield, None]:
        """Drive one full run, yielding every CoreEvent then a terminal AgentResponse.

        This is the engine all other public methods consume. Yields
        each ``CoreEvent`` produced by the reasoning loop in real time
        (model events, reasoning events, tool events, errors), and
        ends with a single ``AgentResponse`` summarizing the run.

        Args:
            task: User input — string, a single ``CoreMessage``, a list
                of messages, or None to use only ``run_context``.
            run_context: Optional pre-existing context. If None, a fresh
                ``RunContext`` is created. The task (when provided) is
                appended to its ``messages``.
            cancellation_token: External cancellation signal.
            stream_tokens: If True, the underlying loop streams LLM
                output token-by-token via ``ModelStreamChunkEvent``.
                If False, only ``ModelResponseEvent`` is emitted per
                LLM call. Pass True when the consumer wants live text.
            **kwargs: Provider-specific overrides forwarded to the
                client.

        Yields:
            Zero or more ``CoreEvent`` instances, then exactly one
            ``AgentResponse`` as the final item.

        Raises:
            asyncio.CancelledError: External cancellation. The response
                is NOT yielded — the consumer cleans up.
        """
        await self.prepare()
        await self._refresh_dynamic_prompt_layers()
        self._validate_runtime_safety()

        ctx = self._normalize_run_context(task, run_context)
        # A caller may omit session_id. Tie that scope to the stable run_id
        # before checkpointing so a resumed run keeps the same directory.
        if not ctx.session_id:
            ctx.session_id = ctx.run_id

        # Workspace exists for every run, independently of registered tools.
        directory = self.workspace.materialize(
            ctx.user_id, ctx.session_id, skills=self.registries.skills
        )
        filesystem = self.workspace.get_filesystem()
        await self._checkpoint(ctx)
        deps: dict[str, t.Any] = {
            "runtime_root": str(directory.root),
            "conversation_dir": str(directory.conversation_dir),
            "skills_dir": str(directory.skill_dir),
            "filesystem_root": str(self.workspace.base_root),
            "workspace_filesystem": filesystem,
            # Existing explicit runtime tools still consume these path names.
            "tools_dir": str(directory.root / "tools"),
            "artifacts_dir": str(directory.conversation_dir),
        }
        if self.registries.requires_sandbox_executor:
            await self.executor.bind_to_workspace(self.workspace.base_root)
        if self.registries.has_skills:
            deps["skill_names"] = [
                skill.name for skill in self.registries.loaded_skill_blocks
            ]

        # Build Tool Executor
        tool_executor = ToolExecutor(
            tools=self.registries.all_tools,
            middlewares=self.middlewares,
            agent_name=self.name,
            runtime_executor=self.executor,
            max_concurrent_tools=self.config.tool_call_concurrency,
            runtime_deps=deps,
            environment=_environment,
        )

        # Inject Prompts and Reasoning Loop
        prompts = self._build_prompt_ctx()
        reasoning = self._build_reasoning(tool_executor)
        max_context_tokens = getattr(self.client.config, "max_context_window", 0) or 0
        self._warn_if_capacity_starved(max_context_tokens, prompts)
        live_message_tokens = self._token_counter.count_messages(ctx.messages)
        live_message_threshold = (
            live_message_threshold_tokens(
                max_context_tokens,
                max_output_tokens=client_max_output_tokens(self.client),
                prompt_tokens=prompts.prompt_tokens or None,
            )
            if max_context_tokens > 0
            else 0
        )
        compaction_started = (
            max_context_tokens > 0 and live_message_tokens > live_message_threshold
        )
        if compaction_started:
            yield self._compaction_event(
                phase="start",
                max_context_tokens=max_context_tokens,
                total_token_count=live_message_tokens,
                prompt_tokens=prompts.prompt_tokens or None,
            )
        compaction_event = await self._apply_compaction(ctx, prompts)
        if compaction_event is not None and (
            compaction_started or compaction_event.changed
        ):
            yield compaction_event

        loop_state = reasoning.LOOP_STATE_CLS()
        # A paused turn (approval / input_needed) stashes its counters on
        # the context; restore them so the resumed segment keeps the same
        # iteration budget and usage accounting. Popped so a *completed*
        # turn never leaks metrics into the next one.
        pending_metrics = ctx.runtime_state.shared_state.pop("loop_metrics", None)
        if isinstance(pending_metrics, dict):
            loop_state.apply_metrics(pending_metrics)
        start_time = time.monotonic()

        # Track the running loop for the duration of the turn. Cleared in
        # the finally so it never dangles past the turn.
        self._active_reasoning = reasoning

        try:
            async for event in reasoning.execute_reasoning_loop(
                ctx=ctx,
                prompts=prompts,
                loop_state=loop_state,
                stream_tokens=stream_tokens,
                cancellation_token=cancellation_token,
                output_format=self.output_format,
                **kwargs,
            ):
                yield event
                # A new iteration starts only after the previous round's
                # tool results are folded in — a safe point to checkpoint.
                if isinstance(event, ReasoningIterationEvent):
                    await self._checkpoint(ctx)

        except asyncio.CancelledError:
            # Caller-initiated cancellation — propagate without
            # yielding a response. The caller is unwinding.
            raise

        except Exception as e:
            # Loop-internal failure. Surface as ErrorEvent so the UI
            # can render it, then mark the response and continue.
            log.error(
                "Agent run failed",
                agent_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
            )
            yield ErrorEvent(
                source=self.name,
                error_message=str(e),
                error_type=type(e).__name__,
                is_recoverable=False,
            )
            loop_state.finish_reason = "error"

        finally:
            # The turn is over (or paused into a response): the loop is no
            # longer resumable from outside.
            self._active_reasoning = None

        # Paused turn → persist loop counters so resume continues the same
        # budget instead of restarting from zero.
        if loop_state.finish_reason in ("approval_needed", "input_needed"):
            ctx.runtime_state.shared_state["loop_metrics"] = (
                loop_state.metrics_snapshot()
            )

        # Terminal checkpoint: covers pauses (approval / input_needed,
        # including the loop-metrics stash above) and completed turns.
        await self._checkpoint(ctx)

        # Terminal yield — always last, always exactly one.
        yield self._build_response(ctx, loop_state, start_time)

    # -------- PUBLIC API — TEXT STREAM -----------------------------------------------------------
    async def run_stream(
        self,
        task: str | CoreMessage | list[CoreMessage] | None = None,
        run_context: RunContext | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[str, None]:
        """Yield assistant text chunks as they stream.

        Convenience over ``run_stream_events`` for CLI / chat use:
        consumes events, yields only the text chunks emitted by the
        LLM (no tool messages, no events, no final response).

        ``stream_tokens=True`` is set automatically — calling this
        method without streaming would defeat the point.

        Args:
            task: User input.
            run_context: Optional pre-existing context.
            cancellation_token: External cancellation signal.
            **kwargs: Forwarded to the client.

        Yields:
            ``str`` chunks of assistant content. Empty strings are
            filtered. Tool call arguments and final marker chunks
            are not yielded.
        """
        async for item in self.run_stream_events(
            task=task,
            run_context=run_context,
            cancellation_token=cancellation_token,
            stream_tokens=True,
            **kwargs,
        ):
            if (
                isinstance(item, ModelStreamChunkEvent)
                and not item.is_final
                and item.chunk
            ):
                yield item.chunk

    # -------- PUBLIC API — BLOCKING -----------------------------------------------------------
    async def run(
        self,
        task: str | CoreMessage | list[CoreMessage] | None = None,
        run_context: RunContext | None = None,
        cancellation_token: CancellationToken | None = None,
        stream_tokens: bool = False,
        **kwargs: t.Any,
    ) -> AgentResponse:
        """Run the agent to completion and return the final response.

        Convenience over ``run_stream_events`` for scripts and tests.
        Discards every intermediate event and returns the terminal
        ``AgentResponse``.

        Args:
            task: User input.
            run_context: Optional pre-existing context.
            cancellation_token: External cancellation signal.
            stream_tokens: Forwarded to the engine. Has no effect on
                the return value (events are discarded either way) but
                affects whether middleware sees streaming chunks.
            **kwargs: Forwarded to the client.

        Returns:
            ``AgentResponse`` with the run's final state.
        """
        response: AgentResponse | None = None
        async for item in self.run_stream_events(
            task=task,
            run_context=run_context,
            cancellation_token=cancellation_token,
            stream_tokens=stream_tokens,
            **kwargs,
        ):
            if isinstance(item, AgentResponse):
                response = item

        # The engine guarantees exactly one AgentResponse as the final
        # item; reaching here without one means the engine itself broke.
        assert response is not None, "run_stream_events did not yield an AgentResponse"
        return response

    # -------- PUBLIC API — RESUME -----------------------------------------------------------
    async def resume(
        self,
        run_context: RunContext,
        cancellation_token: CancellationToken | None = None,
        stream_tokens: bool = False,
        **kwargs: t.Any,
    ) -> AgentResponse:
        """Resume a paused run after the user resolved approvals / questions.

        Convenience over ``resume_stream_events``. Discards intermediate
        events and returns the terminal ``AgentResponse``.
        """
        response: AgentResponse | None = None
        async for item in self.resume_stream_events(
            run_context=run_context,
            cancellation_token=cancellation_token,
            stream_tokens=stream_tokens,
            **kwargs,
        ):
            if isinstance(item, AgentResponse):
                response = item

        assert response is not None, (
            "resume_stream_events did not yield an AgentResponse"
        )
        return response

    async def resume_stream(
        self,
        run_context: RunContext,
        cancellation_token: CancellationToken | None = None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[str, None]:
        """Resume and yield only assistant text chunks.

        Convenience for CLI / chat use. ``stream_tokens=True`` is set
        automatically.
        """
        async for item in self.resume_stream_events(
            run_context=run_context,
            cancellation_token=cancellation_token,
            stream_tokens=True,
            **kwargs,
        ):
            if (
                isinstance(item, ModelStreamChunkEvent)
                and not item.is_final
                and item.chunk
            ):
                yield item.chunk

    async def resume_stream_events(
        self,
        run_context: RunContext,
        cancellation_token: CancellationToken | None = None,
        stream_tokens: bool = False,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[RunYield, None]:
        """Resume a paused run, yielding events then a terminal response.

        Validates that the context is in a state that warrants resuming
        (pending approvals resolved, pending questions answered, no orphan
        executing records), applies the default stale-execution policy
        (fail), and then delegates to the same engine as
        ``run_stream_events`` with ``task=None``.

        Args:
            run_context: Context loaded from a store, with the user's
                approval decisions / answers already applied via
                ``ctx.tool_state.apply_approval(...)`` /
                ``ctx.tool_state.apply_user_answer(...)``.
            cancellation_token: External cancellation signal.
            stream_tokens: Forwarded to the loop.
            **kwargs: Forwarded to the client.

        Yields:
            ``CoreEvent`` instances followed by exactly one
            ``AgentResponse``.

        Raises:
            AgentError: If the context has unresolved pending
                approvals, or has no work left to do.
        """
        self._validate_resumable(run_context)
        self._handle_stale_executions(run_context)

        async for item in self.run_stream_events(
            task=None,
            run_context=run_context,
            cancellation_token=cancellation_token,
            stream_tokens=stream_tokens,
            **kwargs,
        ):
            yield item

    # -------- PERSISTENCE -----------------------------------------------------------
    async def _checkpoint(self, ctx: RunContext) -> None:
        """Persist the context to the configured store; never raises.

        Checkpointing is best-effort by design: losing a checkpoint is
        strictly better than killing a live run over a storage hiccup.
        """
        if self.store is None:
            return
        try:
            await self.store.save(ctx.run_id, ctx)
        except Exception as e:
            log.warning(
                "Run checkpoint failed",
                agent_name=self.name,
                run_id=ctx.run_id,
                error=str(e),
            )

    async def load_run(self, run_id: str) -> RunContext | None:
        """Load a checkpointed run from the configured store.

        Convenience for the crash-recovery flow: load, apply any pending
        decisions/answers via ``ctx.tool_state``, then ``resume(ctx)``.

        Raises:
            AgentError: If the agent has no store configured.
        """
        if self.store is None:
            raise AgentError(
                f"Agent {self.name!r} has no RunContextStore configured; "
                "pass store=... to the constructor to enable checkpointing."
            )
        return await self.store.load(run_id)

    # -------- RESUME HELPERS -----------------------------------------------------------
    def _validate_resumable(self, ctx: RunContext) -> None:
        """Raise if the context can't or shouldn't be resumed.

        Three failure modes:

        - Records still in ``PENDING_APPROVAL`` — the caller forgot to
          apply user decisions. Resuming would just pause again, which
          is technically idempotent but almost certainly a bug.
        - Records still in ``INPUT_NEEDED`` — the agent's question was
          never answered. Apply answers via
          ``ctx.tool_state.apply_user_answer()`` first.
        - Nothing actionable left — every record consumed and no
          assistant message awaiting follow-up. There is no work to do;
          calling resume here is a logic error in the caller.
        """
        pending = ctx.tool_state.pending_approvals
        if pending:
            ids = [r.id for r in pending]
            raise AgentError.unresolved_approvals(
                agent_name=self.name, tool_call_ids=ids
            )

        unanswered = ctx.tool_state.pending_user_input
        if unanswered:
            raise AgentError.unanswered_user_input(
                agent_name=self.name, tool_call_ids=[r.id for r in unanswered]
            )

        # If there are no actionable records and no stale executions,
        # the loop has nothing to do beyond what already happened. The
        # caller probably wants ``run(task=...)`` instead.
        actionable = ctx.tool_state.actionable_calls
        rejected = ctx.tool_state.rejected_calls
        stale = ctx.tool_state.stale_executions
        if not actionable and not rejected and not stale:
            raise AgentError.nothing_to_resume(agent_name=self.name)

    def _handle_stale_executions(self, ctx: RunContext) -> None:
        """Default stale-execution policy: mark them all as failed.

        Stale records are tool calls that were ``EXECUTING`` when the
        previous run died. We have no way to know whether they actually
        completed, so the safe default is to surface them as failures —
        the LLM sees the failure on the next turn and decides what to
        do (retry, ask, give up).

        Subclasses can override this method to implement a different
        policy (e.g. retry stale, ask the user, leave alone).
        """
        for record in list(ctx.tool_state.stale_executions):
            ctx.tool_state.fail_stale(
                record.id,
                reason="Tool execution did not complete in a previous run.",
            )

    @property
    def rendered_layers(self) -> dict[type["CoreLayer"], str]:
        """Rendered prompt layers keyed by concrete type.

        Available only after ``prepare()`` has run. The dict iteration
        order matches the underlying stack's insertion order, which the
        client can use as a default ordering hint when assembling the
        system prompt for its provider.
        """
        self._ensure_prepared()
        return dict(self._rendered_layers)

    @property
    def rendered_layer_usage(self) -> dict[str, PromptLayerUsage]:
        """Token and size stats for rendered prompt layers."""
        self._ensure_prepared()
        return dict(self._rendered_layer_usage)

    @property
    def prompt_tokens(self) -> int:
        """Total token count for all rendered prompt layers."""
        self._ensure_prepared()
        return self._prompt_tokens

    @property
    def is_prepared(self) -> bool:
        return self._prepared

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"prepared={self._prepared}, "
            f"registries={self.registries!r})"
        )
