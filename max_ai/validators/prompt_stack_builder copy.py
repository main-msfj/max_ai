"""Builder for the default PromptStack with user overrides."""

from __future__ import annotations

import logging
import typing as t

from ..base.layer import CoreLayer
from ..stacks.memory_layer import MemoryLayer
from ..stacks.skills_layer import SkillsLayer
from ..stacks.routine_layer import RoutineLayer
from ..stacks.context_layer import ContextLayer
from ..stacks.rendering_layer import RenderingLayer
from ..stacks.knowledge_layer import KnowledgeLayer
from ..stacks.agent_policy_layer import AgentPolicyLayer
from ..stacks.task_analysis_layer import TaskAnalysisLayer
from ..stacks.priority_tools_layer import PriorityToolsLayer

from ..loggers import ScopedLogger

# The canonical set of framework layers. Order here is the insertion
# order into the stack — it is NOT the order in the final prompt
# (that is decided by each provider client's ``format_messages``).
DEFAULT_FRAMEWORK_LAYERS: tuple[type[CoreLayer], ...] = (
    AgentPolicyLayer,
    TaskAnalysisLayer,
    RenderingLayer,
    PriorityToolsLayer,
    SkillsLayer,
    RoutineLayer,
    KnowledgeLayer,
    ContextLayer,
    MemoryLayer,
)

LayerT = t.TypeVar("LayerT", bound=CoreLayer)

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="PromptStackBuilder")


class PromptStack:
    """
    Validated container of ``CoreLayer`` layers.

    Responsibilities:
      - Hold a set of layers keyed by concrete type (one instance per type).
      - Fail fast at construction if any layer is malformed — the caller
        finds out about broken templates immediately, not at runtime.
      - Expose type-safe access (``get``, ``find``, ``__contains__``) so
        provider clients can pick layers in the order their API expects.

    What this class does NOT do:
      - It does not compose or order layers into a final prompt string.
        Each provider client (Anthropic, OpenAI, etc.) performs its own
        assembly via ``format_messages``, because the target message
        shape differs between providers.
    """

    def __init__(self, layers: t.Sequence[CoreLayer]) -> None:
        if not layers:
            raise ValueError("PromptStack requires at least one layer.")

        self._check_no_duplicates(layers)

        # Revalidate every layer up front. Each CoreLayer already validates
        # itself in __init__, but calling it here makes the guarantee
        # explicit: a PromptStack only exists if every layer is healthy.
        for layer in layers:
            layer._validate_placeholders()

        self._layers: dict[type, CoreLayer] = {type(layer): layer for layer in layers}

    # -------- INTROSPECTION -----------------------------------------------------------
    def __iter__(self) -> t.Iterator[CoreLayer]:
        """Iterate over layers in insertion order."""
        return iter(self._layers.values())

    def __len__(self) -> int:
        return len(self._layers)

    def __contains__(self, layer_type: type) -> bool:
        return layer_type in self._layers

    def as_list(self) -> list[CoreLayer]:
        """Return a shallow copy of the layers in insertion order."""
        return list(self._layers.values())

    def get(self, layer_type: type[LayerT]) -> LayerT:
        """Return the layer of the given type. Raises KeyError if absent."""
        try:
            return t.cast(LayerT, self._layers[layer_type])
        except KeyError:
            raise KeyError(f"No layer of type {layer_type.__name__} in stack.")

    def find(self, layer_type: type[LayerT]) -> LayerT | None:
        """Return the layer of the given type, or None if absent."""
        layer = self._layers.get(layer_type)
        return t.cast(LayerT, layer) if layer is not None else None

    # -------- MUTATION -----------------------------------------------------------
    def replace(self, layer: CoreLayer) -> None:
        """Replace the layer of the same concrete type with ``layer``."""
        layer_type = type(layer)
        if layer_type not in self._layers:
            raise KeyError(f"No layer of type {layer_type.__name__} to replace.")
        layer._validate_placeholders()
        self._layers[layer_type] = layer

    def append(self, layer: CoreLayer) -> None:
        """Add a new layer type to the stack."""
        layer_type = type(layer)
        if layer_type in self._layers:
            raise ValueError(
                f"Stack already has a layer of type {layer_type.__name__}."
            )
        layer._validate_placeholders()
        self._layers[layer_type] = layer

    def remove(self, layer_type: type) -> None:
        """Remove the layer of the given type."""
        if layer_type not in self._layers:
            raise KeyError(f"No layer of type {layer_type.__name__} to remove.")
        del self._layers[layer_type]

    # -------- INTERNALS -----------------------------------------------------------
    @staticmethod
    def _check_no_duplicates(layers: t.Sequence[CoreLayer]) -> None:
        seen: set[type] = set()
        for layer in layers:
            layer_type = type(layer)
            if layer_type in seen:
                raise ValueError(
                    f"Duplicate layer type in stack: {layer_type.__name__}"
                )
            seen.add(layer_type)

    def __repr__(self) -> str:
        names = ", ".join(type(l).__name__ for l in self._layers.values())
        return f"PromptStack([{names}])"


def build_default_stack(
    overrides: t.Sequence[CoreLayer] | None = None,
) -> PromptStack:
    """
    Build the default ``PromptStack`` with any user-provided overrides.

    Every layer is validated during stack construction. If a user passes
    a broken override (bad template, contract mismatch), this function
    raises — the caller learns about the problem before the agent ever
    runs.

    Overrides:
      - An override whose type matches a default replaces the default.
      - An override whose type is unknown is appended to the stack
        (framework users can extend with their own layers).

    Args:
        overrides: Iterable of custom layer instances.

    Returns:
        A fully validated ``PromptStack``.

    Raises:
        StackError: If any layer (default or override) fails validation.
        ValueError: If overrides contain two layers of the same type.

    Example:
        stack = build_default_stack(overrides=[
            AgentPolicyLayer(template=my_custom_template),
        ])
    """
    layers: list[CoreLayer] = [cls() for cls in DEFAULT_FRAMEWORK_LAYERS]

    if not overrides:
        return PromptStack(layers)

    override_by_type: dict[type, CoreLayer] = {}
    for override in overrides:
        override_type = type(override)
        if override_type in override_by_type:
            raise ValueError(
                f"Duplicate override for layer type {override_type.__name__}."
            )
        override_by_type[override_type] = override

    # Replace defaults in place. Whatever overrides remain afterwards
    # didn't match any default — those are user extensions and get
    # appended to the end.
    for i, layer in enumerate(layers):
        if type(layer) in override_by_type:
            layers[i] = override_by_type.pop(type(layer))

    extras: list[CoreLayer] = []
    for override_type, override in override_by_type.items():
        log.info("Adding custom layer to stack: %s", override_type.__name__)
        extras.append(override)

    return PromptStack(layers + extras)
