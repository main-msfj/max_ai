"""
Component configuration System
"""

from __future__ import annotations

import asyncio
import importlib
import os
import typing as t
import warnings

from pydantic import BaseModel

ComponentType = t.Union[
    t.Literal[
        "model",
        "agent",
        "tool",
        "termination",
        "orchestrator",
        "step",
        "workflow",
        "memory",
        "store",
        "knowledge",
        "prompts",
        "skills",
        "context",
    ],
    str,
]

ConfigT = t.TypeVar("ConfigT", bound=BaseModel)
ExpectedT = t.TypeVar("ExpectedT")


class ComponentModel(BaseModel):
    """Serialized representation of a component."""

    provider: str
    """Import path of the component class."""

    component_type: ComponentType | None = None
    """Logical category of the component."""

    version: int | None = None
    """Schema version for compatibility."""

    component_version: int | None = None
    """Implementation version for migrations."""

    description: str | None = None
    """Human-readable description."""

    label: str | None = None
    """UI label for display."""

    config: dict[str, t.Any]
    """Configuration payload for instantiation."""


def _type_to_provider_str(cls: type) -> str:
    """
    Return the importable provider path for a component class.

    Returns
    -------
    str
        The resulting text value.
    """
    return f"{cls.__module__}.{cls.__qualname__}"


KNOWN_PROVIDERS: dict[str, str] = {
    "ollama": "max_ai.capabilities.clients.ollama.client.OllamaChatCompletionClient",
    "OllamaChatCompletionClient": "max_ai.capabilities.clients.ollama.client.OllamaChatCompletionClient",
    "maxai.llm.OllamaChatCompletionClient": "max_ai.capabilities.clients.ollama.client.OllamaChatCompletionClient",
    "OpenAIChatCompletionClient": "max_ai.capabilities.clients.openai.client.OpenAIChatCompletionClient",
    "maxai.llm.OpenAIChatCompletionClient": "max_ai.capabilities.clients.openai.client.OpenAIChatCompletionClient",
    "OpenRouterChatCompletionClient": "max_ai.capabilities.clients.openrouter.client.OpenRouterChatCompletionClient",
    "maxai.llm.OpenRouterChatCompletionClient": "max_ai.capabilities.clients.openrouter.client.OpenRouterChatCompletionClient",
    "maxai.stacks.AgentPolicyLayer": "max_ai.capabilities.stacks.agent_policy_layer.AgentPolicyLayer",
    "maxai.stacks.TaskAnalysisLayer": "max_ai.capabilities.stacks.task_analysis_layer.TaskAnalysisLayer",
    "maxai.stacks.RenderingLayer": "max_ai.capabilities.stacks.rendering_layer.RenderingLayer",
    "maxai.stacks.SkillsLayer": "max_ai.capabilities.stacks.skills_layer.SkillsLayer",
    "maxai.stacks.KnowledgeLayer": "max_ai.capabilities.stacks.knowledge_layer.KnowledgeLayer",
    "maxai.stacks.ContextLayer": "max_ai.capabilities.stacks.context_layer.ContextLayer",
    "maxai.stacks.MemoryLayer": "max_ai.capabilities.stacks.memory_layer.MemoryLayer",
    "maxai.stacks.SessionStateLayer": "max_ai.capabilities.stacks.session_state_layer.SessionStateLayer",
    "maxai.agents.Agent": "max_ai.agents.agent.Agent",
    "maxai.completion.RuntimeCompletionGate": "max_ai.capabilities.completion_gate.gate.RuntimeCompletionGate",
    "maxai.reasoning.ReactLoop": "max_ai.capabilities.reasoning.react.loop.ReactLoop",
    "maxai.guards.SchemaRetryGuard": "max_ai.capabilities.reasoning.guards.SchemaRetryGuard",
    "maxai.guards.RepetitionGuard": "max_ai.capabilities.reasoning.guards.RepetitionGuard",
    "maxai.guards.BudgetGuard": "max_ai.capabilities.reasoning.guards.BudgetGuard",
    "maxai.guards.NoProgressGuard": "max_ai.capabilities.reasoning.guards.NoProgressGuard",
    "maxai.guards.PlanCompletionGuard": "max_ai.capabilities.reasoning.guards.PlanCompletionGuard",
    "maxai.session_store.LocalSessionStore": "max_ai.capabilities.session_store.local._store.LocalSessionStore",
    "maxai.compaction.SummaryCompaction": "max_ai.capabilities.compaction.summary._strategy.SummaryCompaction",
    "maxai.compaction.SlidingWindowCompaction": "max_ai.capabilities.compaction.window._strategy.SlidingWindowCompaction",
}


# Provider allowlist: deserialize imports the class a config names, and an
# import runs code, so only trusted packages load. Built-ins always do.
_ALLOWED_PREFIXES: set[str] = {"max_ai."}


def allow_providers(*prefixes: str) -> None:
    """Let ``deserialize`` load components from these packages, e.g.
    ``allow_providers("my_company.", "third_party_pkg.")``. ``"*"`` allows
    any importable class (local development only). The env var
    ``MAXAI_ALLOWED_PROVIDERS`` (comma-separated) does the same without code.
    """
    _ALLOWED_PREFIXES.update(prefix.strip() for prefix in prefixes if prefix.strip())


def _check_allowed(provider: str) -> None:
    """
    Ensure a component provider is permitted by the trust configuration.

    Parameters
    ----------
    provider : str
        Import path or provider name to check.
    """
    env = os.getenv("MAXAI_ALLOWED_PROVIDERS", "")
    allowed = _ALLOWED_PREFIXES | {p.strip() for p in env.split(",") if p.strip()}
    if "*" in allowed or any(provider.startswith(prefix) for prefix in allowed):
        return
    raise PermissionError(
        f"Component provider {provider!r} is not allowed. Trust its package with "
        f"allow_providers({provider.split('.')[0] + '.'!r}) or MAXAI_ALLOWED_PROVIDERS."
    )


class ComponentBase(t.Generic[ConfigT]):
    """Enables serialization into config models."""

    component_type: t.ClassVar[ComponentType]
    """Component category."""

    component_schema: t.ClassVar[type[BaseModel]]
    """Component Schema"""

    component_version: t.ClassVar[int] = 1
    """Schema version."""

    component_provider_override: t.ClassVar[str | None] = None
    """Optional override for import path."""

    component_description: t.ClassVar[str | None] = None
    """Optional description override."""

    component_label: t.ClassVar[str | None] = None
    """Optional UI label override."""

    def _to_config(self) -> ConfigT:
        """
        Return the component settings that should be serialized.

        Returns
        -------
        ConfigT
            The component configuration model.
        """
        raise NotImplementedError("This component does not support dumping to config")

    @classmethod
    def _from_config(cls, config: ConfigT) -> t.Self:
        """
        Construct a component from its validated configuration.

        Parameters
        ----------
        config : ConfigT
            Model or component configuration.

        Returns
        -------
        t.Self
            The component reconstructed from its configuration.
        """
        raise NotImplementedError("This component does not support loading from config")

    @classmethod
    def _from_config_past_version(
        cls,
        config: dict[str, t.Any],
        version: int,
    ) -> t.Self:
        """
        Migrate an older configuration and construct the component.

        Parameters
        ----------
        config : dict[str, t.Any]
            Model or component configuration.
        version : int
            Version number of the saved configuration.

        Returns
        -------
        t.Self
            The component reconstructed from the migrated configuration.
        """
        raise NotImplementedError(
            "This component does not support loading from past versions"
        )

    def serialize(self) -> ComponentModel:
        """This component as storable config: its provider plus a config with
        no secrets (only the names of the env vars that hold them). Store it
        with ``.model_dump_json()`` and rebuild it with ``deserialize``."""
        provider = self.component_provider_override or _type_to_provider_str(
            self.__class__
        )

        if "<locals>" in provider:
            raise TypeError("Cannot dump component with local class")

        if "._" in provider:
            warnings.warn(
                "Internal module path used in provider string. "
                "Set component_provider_override to avoid this.",
                stacklevel=2,
            )

        if not hasattr(self, "component_type"):
            raise AttributeError("component_type not defined")

        description = self.component_description
        if description is None and self.__class__.__doc__:
            docstring = self.__class__.__doc__.strip()
            for marker in ["\n\nArgs:", "\n\nParameters:", "\n\nAttributes:", "\n\n"]:
                docstring = docstring.split(marker)[0]
            description = docstring.strip()

        config = self._to_config().model_dump(exclude_none=True)

        return ComponentModel(
            provider=provider,
            component_type=self.component_type,
            version=self.component_version,
            component_version=self.component_version,
            description=description,
            label=self.component_label or self.__class__.__name__,
            config=config,
        )

    @classmethod
    def deserialize(
        cls,
        data: ComponentModel | dict[str, t.Any] | str | bytes,
        expected: type[ExpectedT] | None = None,
    ) -> t.Self | ExpectedT:
        """Rebuild a component from ``serialize()`` output: the model, its
        dict or its JSON text (e.g. straight from a database row)."""
        if isinstance(data, (str, bytes)):
            loaded_model = ComponentModel.model_validate_json(data)
        elif isinstance(data, dict):
            loaded_model = ComponentModel(**data)
        else:
            loaded_model = data

        provider = KNOWN_PROVIDERS.get(
            loaded_model.provider,
            loaded_model.provider,
        )

        parts = provider.rsplit(".", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid provider path: {provider!r}")
        _check_allowed(provider)

        module_path, class_name = parts
        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)

        if not is_component_class(component_class):
            raise TypeError(f"Invalid component class: {component_class!r}")

        loaded_version = (
            loaded_model.component_version
            or loaded_model.version
            or component_class.component_version
        )

        if loaded_version < component_class.component_version:
            instance = component_class._from_config_past_version(
                loaded_model.config,
                loaded_version,
            )
        else:
            schema = component_class.component_schema
            validated_config = schema.model_validate(loaded_model.config)
            instance = component_class._from_config(validated_config)

        if expected is not None:
            if not isinstance(instance, expected):
                raise TypeError(
                    f"Expected {expected.__name__}, got {type(instance).__name__}"
                )
            return instance

        if not isinstance(instance, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(instance).__name__}")

        return instance

    @staticmethod
    def require_type(value: t.Any, expected: type[ExpectedT], field: str) -> ExpectedT:
        """
        Validate a value against an expected Python type.

        Parameters
        ----------
        value : t.Any
            Value to validate.
        expected : type[ExpectedT]
            Expected Python type.
        field : str
            Field name used to describe a validation error.

        Returns
        -------
        ExpectedT
            The validated value.
        """
        if not isinstance(value, expected):
            raise TypeError(
                f"{field} must be {expected.__name__}, got {type(value).__name__}"
            )
        return value


class CoreLifecycleComponent(ComponentBase[ConfigT]):
    """Serializable component with an async connection lifecycle."""

    def __init__(self) -> None:
        """
        Initialize the lifecycle component state.
        """
        self._connected: bool = False
        self._connect_lock: asyncio.Lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open backend resources needed by this component."""
        return None

    async def disconnect(self) -> None:
        """Close backend resources opened by connect()."""
        return None

    async def _ensure_connected(self) -> None:
        """Connect once, safely, even under concurrent calls."""
        if not self._connected:
            async with self._connect_lock:
                if not self._connected:
                    await self.connect()
                    self._connected = True

    async def __aenter__(self) -> t.Self:
        """
        Connect the component and return it for an async context.

        Returns
        -------
        t.Self
            This lifecycle component, connected and ready for use.
        """
        await self._ensure_connected()
        return self

    async def __aexit__(self, *exc: t.Any) -> None:
        """
        Disconnect the component when its async context ends.

        Parameters
        ----------
        exc : t.Any
            Exception information passed by the async context manager.
        """
        if self._connected:
            await self.disconnect()
            self._connected = False


class Component(t.Generic[ConfigT]):
    """Marker mixin for concrete serializable components.

    Concrete classes usually combine this with a framework base that
    already inherits ``ComponentBase``:

        class MyLayer(Component[MyConfig], CoreLayer): ...
    """

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        """
        Register subclass metadata and validate its component declaration.

        Parameters
        ----------
        kwargs : t.Any
            Subclass-specific initialization options.
        """
        super().__init_subclass__(**kwargs)
        if not is_component_class(cls):
            warnings.warn(
                (
                    f"Component class {cls.__name__!r} should also inherit from "
                    "ComponentBase through one of the core framework bases."
                ),
                stacklevel=2,
            )


def is_component_class(value: t.Any) -> t.TypeGuard[type[ComponentBase[t.Any]]]:
    """
    Check whether a value is a concrete ComponentBase subclass.

    Parameters
    ----------
    value : t.Any
        Value to validate.

    Returns
    -------
    t.TypeGuard[type[ComponentBase[t.Any]]]
        Whether the value is a ComponentBase subclass.
    """
    if not isinstance(value, type):
        return False

    try:
        if not issubclass(value, ComponentBase):
            return False
    except TypeError:
        return False

    component_cls = t.cast(type[ComponentBase[t.Any]], value)

    return (
        getattr(component_cls, "component_schema", None) is not None
        and getattr(component_cls, "component_type", None) is not None
        and callable(getattr(component_cls, "_to_config", None))
        and callable(getattr(component_cls, "_from_config", None))
    )
