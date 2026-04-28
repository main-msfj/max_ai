"""
Component system for LLM agents.

Provides a unified serialization framework that converts Python components
into structured JSON configurations and restores them at runtime.

Enables:
- workflow persistence
- UI-driven configuration
- cross-service portability

Inspired by AutoGen's component architecture.
"""

from __future__ import annotations

import warnings
import importlib
import typing as t

from pydantic import BaseModel
from typing_extensions import Self, TypeGuard, TypeVar

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
        "prompts"
    ],
    str,
]

ConfigT = TypeVar("ConfigT", bound=BaseModel)
FromConfigT = TypeVar("FromConfigT", bound=BaseModel, contravariant=True)
ToConfigT = TypeVar("ToConfigT", bound=BaseModel, covariant=True)
T = TypeVar("T")


# Core model
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


def _type_to_provider_str(t: type) -> str:
    """Convert a class into its import path."""
    return f"{t.__module__}.{t.__qualname__}"


# Provider registry
KNOWN_PROVIDER = {
    "openai_chat_completion_client": "maxais.llm.OpenAIChatCompletionClient",
    "OpenAIChatCompletionClient": "maxais.llm.OpenAIChatCompletionClient",
    "client": "maxais.llm.OpenAIChatCompletionClient",
    "agent": "maxais.agents.Agent",
    "Agent": "maxais.agents.Agent",
    "list_memory": "maxais.memory.ListMemory",
    "ListMemory": "maxais.memory.ListMemory",
    "file_memory": "maxais.memory.FileMemory",
    "FileMemory": "maxais.memory.FileMemory",
    "memory": "maxais.memory.ListMemory",
    "max_message_termination": "maxais.termination.MaxMessageTermination",
    "text_mention_termination": "maxais.termination.TextMentionTermination",
    "composite_termination": "maxais.termination.CompositeTermination",
    "termination": "maxais.termination.MaxMessageTermination",
    "round_robin_orchestrator": "maxais.orchestration.RoundRobinOrchestrator",
    "AIOrchestrator": "maxais.orchestration.AIOrchestrator",
    "orchestrator": "maxais.orchestration.RoundRobinOrchestrator",
    "workflow": "maxais.workflow.Workflow",
    "Workflow": "maxais.workflow.Workflow",
}


# From config
class ComponentFromConfig(t.Generic[FromConfigT]):
    """Allows instantiation from configuration objects."""

    @classmethod
    def _from_config(cls, config: FromConfigT) -> Self:
        """Create instance from config."""
        raise NotImplementedError()

    @classmethod
    def from_config_past_version(cls, config: dict[str, t.Any], version: int) -> Self:
        """Load older config versions."""
        raise NotImplementedError()


# To config
class ComponentToConfig(t.Generic[ToConfigT]):
    """Enables serialization into config models."""

    component_type: t.ClassVar[ComponentType]
    """Component category."""

    component_version: t.ClassVar[int] = 1
    """Schema version."""

    component_provider_override: t.ClassVar[str | None] = None
    """Optional override for import path."""

    component_description: t.ClassVar[str | None] = None
    """Optional description override."""

    component_label: t.ClassVar[str | None] = None
    """Optional UI label override."""

    def _to_config(self) -> ToConfigT:
        """Convert instance to config."""
        raise NotImplementedError()

    def dump_component(self) -> ComponentModel:
        """Serialize component to portable model."""

        provider = (
            self.component_provider_override
            if self.component_provider_override
            else _type_to_provider_str(self.__class__)
        )

        if "<locals>" in provider:
            raise TypeError("Local classes cannot be serialized")

        if "._" in provider:
            warnings.warn(
                "Internal module path used in provider string.",
                stacklevel=2,
            )

        description = self.component_description or (
            self.__class__.__doc__.strip().split("\n\n")[0]
            if self.__class__.__doc__
            else None
        )

        return ComponentModel(
            provider=provider,
            component_type=self.component_type,
            version=self.component_version,
            component_version=self.component_version,
            description=description,
            label=self.component_label or self.__class__.__name__,
            config=self._to_config().model_dump(exclude_none=True),
        )


# Loader
ExpectedType = TypeVar("ExpectedType")


class ComponentLoader:
    """Loads components from serialized models."""

    @classmethod
    def load_component(
        cls,
        model: ComponentModel | dict[str, t.Any],
        expected: type[ExpectedType] | None = None,
    ) -> Self | ExpectedType:

        if isinstance(model, dict):
            model = ComponentModel(**model)

        if model.provider in KNOWN_PROVIDER:
            model.provider = KNOWN_PROVIDER[model.provider]

        module_path, class_name = model.provider.rsplit(".", 1)
        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)

        if not is_component_class(component_class):
            raise TypeError("Invalid component class")

        schema = component_class.component_config_schema

        if model.component_version < component_class.component_version:
            instance = component_class.from_config_past_version(
                model.config, model.component_version or 0
            )
        else:
            validated = schema.model_validate(model.config)
            instance = component_class._from_config(validated)

        if expected and not isinstance(instance, expected):
            raise TypeError("Type mismatch")

        return instance


# ─────────────────────────────
# Schema
# ─────────────────────────────


class ComponentSchemaType(t.Generic[ConfigT]):
    """Declares required configuration schema."""

    component_config_schema: type[ConfigT]


# Base
class ComponentBase(
    ComponentToConfig[ConfigT],
    ComponentLoader,
    t.Generic[ConfigT],
):
    """Base class combining serialization and loading."""
    @staticmethod
    def require_type(value: t.Any, expected: type[T], field: str) -> T:
        """
        Type guard for constructor arguments.

        Raises ``TypeError`` if ``value`` is not an instance of ``expected``.
        Returns ``value`` unchanged when valid, so it can be used inline:

            self.name = self.require_type(name, str, "name")

        Args:
            value: The value to validate.
            expected: The expected type.
            field: Field name used in the error message.

        Returns:
            The original value, typed as ``expected``.
        """
        if not isinstance(value, expected):
            raise TypeError(
                f"{field} must be {expected.__name__}, "
                f"got {type(value).__name__}"
            )
        return value


class Component(
    ComponentFromConfig[ConfigT],
    ComponentSchemaType[ConfigT],
    t.Generic[ConfigT],
):
    """Base class for all serializable components."""


class _ConcreteComponent(
    ComponentFromConfig[ConfigT],
    ComponentSchemaType[ConfigT],
    ComponentToConfig[ConfigT],
    ComponentLoader,
    t.Generic[ConfigT],
):
    pass


# Helpers
def is_component_instance(cls: t.Any) -> TypeGuard[_ConcreteComponent[BaseModel]]:
    """Check if instance is a valid component."""
    return (
        isinstance(cls, ComponentFromConfig)
        and isinstance(cls, ComponentToConfig)
        and isinstance(cls, ComponentSchemaType)
        and isinstance(cls, ComponentLoader)
    )


def is_component_class(cls: type) -> TypeGuard[t.Type[_ConcreteComponent[BaseModel]]]:
    """Check if class is a valid component type."""
    return (
        issubclass(cls, ComponentFromConfig)
        and issubclass(cls, ComponentToConfig)
        and issubclass(cls, ComponentSchemaType)
        and issubclass(cls, ComponentLoader)
    )
