"""The agent: where the harness decides how a run is done."""

from __future__ import annotations

import json
import typing as t
from collections.abc import Sequence

from pydantic import BaseModel

from ..base.clients import CoreChatCompletionClient
from ..base.completion_gate import CompletionBase
from ..base.component import ComponentBase
from ..base.knowledge import CoreKnowledgeRegistry
from ..base.layer import CoreLayer
from ..base.loop import CoreLoop
from ..base.middleware import CoreMiddleware
from ..base.sandbox import SandboxCore
from ..base.skills import SkillsCore
from ..base.tools import CoreTool
from ..capabilities.loop import ReactLoop
from ..capabilities.mcp._model import (
    HTTPServerConfig,
    MCPServerConfig,
    StdioMCPServerConfig,
    serialize_mcp_servers,
)
from ..capabilities.memory import Memory
from ..core.harness.rules import Harness
from ..core.model.agent import AgentSpec
from ..core.model.json_schema import model_from_schema, schema_of

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
        gates: Sequence[CompletionBase] | None = None,
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
        self.reasoning: AnyLoop = reasoning if reasoning is not None else ReactLoop()
        self.middlewares = list(middlewares or ())
        self.gates = list(gates or ())
        self.output_format = output_format
        self.prompt_layers = list(prompt_layers or ())
        self.sandbox = sandbox
        # 2. Refuse a piece that does not come from its Core class.
        self._validate()
        # 3. The rules of the harness; it also adds what is ours.
        self._run_harness_validation()

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
        self.middlewares = Harness.middleware_rules(self.middlewares)
        self.tools = Harness.tool_combination(self.toolset, self.sandbox, self.skills)

    def _validate(self) -> None:
        """Every piece must come from its Core class; the error names the one that does not."""
        # 1. The texts that identify the agent.
        for what in ("name", "description", "instructions"):
            if not isinstance(getattr(self, what), str):
                raise TypeError(f"{what} must be text")
        if not self.name.strip():
            raise ValueError("name cannot be empty")
        # 2. The pieces that are one object; only client and reasoning are required.
        self._check(self.client, CoreChatCompletionClient, "client")
        self._check(self.reasoning, CoreLoop, "reasoning")
        if self.memory is not None:
            self._check(self.memory, Memory, "memory")
        if self.skills is not None:
            self._check(self.skills, SkillsCore, "skills")
        if self.sandbox is not None:
            self._check(self.sandbox, SandboxCore, "sandbox")
        # 3. The pieces that are a list: each element is checked.
        lists: tuple[tuple[Sequence[object], type, str], ...] = (
            (self.toolset, CoreTool, "toolset"),
            (self.mcp, MCPServerConfig, "mcp"),
            (self.knowledge, CoreKnowledgeRegistry, "knowledge"),
            (self.middlewares, CoreMiddleware, "middlewares"),
            (self.gates, CompletionBase, "gates"),
            (self.prompt_layers, CoreLayer, "prompt_layers"),
        )
        for values, kind, what in lists:
            for value in values:
                self._check(value, kind, what)
        # 4. The answer's format is a class, not an object.
        form = t.cast("object", self.output_format)
        if form is not None and not (isinstance(form, type) and issubclass(form, BaseModel)):
            raise TypeError("output_format must be a pydantic BaseModel class")

    @staticmethod
    def _check(value: object, kind: type, what: str) -> None:
        """Raise when ``value`` is not a ``kind``, saying what was given."""
        if not isinstance(value, kind):
            given = type(value).__name__
            raise TypeError(f"{what} must be a {kind.__name__}; got {given}")

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
            gates=[dump(gate, "gate") for gate in self.gates],
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
            gates=[load(gate) for gate in config.gates],
            output_format=model_from_schema(form) if form is not None else None,
            prompt_layers=[load(layer) for layer in config.prompt_layers],
            sandbox=load(config.sandbox),
        )
