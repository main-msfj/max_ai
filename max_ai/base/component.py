"""
Component configuration System
"""

from __future__ import annotations

import asyncio
import importlib
import warnings
import typing as t

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
        "routines",
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
    return f"{cls.__module__}.{cls.__qualname__}"


KNOWN_PROVIDERS: dict[str, str] = {
    "ollama": "max_ai.clients.ollama.client.OllamaChatCompletionClient",
    "OllamaChatCompletionClient": "max_ai.clients.ollama.client.OllamaChatCompletionClient",
}


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
        raise NotImplementedError("This component does not support dumping to config")

    @classmethod
    def _from_config(cls, config: ConfigT) -> t.Self:
        raise NotImplementedError("This component does not support loading from config")

    @classmethod
    def _from_config_past_version(
        cls,
        config: dict[str, t.Any],
        version: int,
    ) -> t.Self:
        raise NotImplementedError(
            "This component does not support loading from past versions"
        )

    def dump_component(self) -> ComponentModel:
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
    def load_component(
        cls,
        model: ComponentModel | dict[str, t.Any],
        expected: type[ExpectedT] | None = None,
    ) -> t.Self | ExpectedT:
        loaded_model = ComponentModel(**model) if isinstance(model, dict) else model

        provider = KNOWN_PROVIDERS.get(
            loaded_model.provider,
            loaded_model.provider,
        )

        parts = provider.rsplit(".", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid provider path: {provider!r}")

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
        if not isinstance(value, expected):
            raise TypeError(
                f"{field} must be {expected.__name__}, got {type(value).__name__}"
            )
        return value


class CoreLifecycleComponent(ComponentBase[ConfigT]):
    """Serializable component with an async connection lifecycle."""

    def __init__(self) -> None:
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
        await self._ensure_connected()
        return self

    async def __aexit__(self, *exc: t.Any) -> None:
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
