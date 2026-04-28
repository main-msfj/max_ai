"""Prompt Stack — validated container of prompt layers.

The stack does NOT compose the final prompt. It holds validated layers
and exposes them for provider clients to assemble in their own native
message format (Anthropic vs OpenAI vs others order/group differently).

Validation happens at construction time: any layer whose template is
malformed or whose declared contract doesn't match its placeholders
raises immediately, BEFORE the stack is ever used at runtime.
"""

from __future__ import annotations

import typing as t

from ..base.layer import CoreLayer

LayerT = t.TypeVar("LayerT", bound=CoreLayer)


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
