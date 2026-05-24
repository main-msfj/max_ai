from __future__ import annotations

import typing as t
from pydantic import BaseModel, Field
from ..base.layer import CoreLayer
from ..manager.stacks import LayerContainer


class PromptLayerUsage(BaseModel):
    """Token and size stats for one rendered prompt layer."""

    layer_name: str = Field(..., description="Concrete prompt layer class name.")
    chars: int = Field(default=0, description="Rendered character count.")
    tokens: int = Field(default=0, description="Rendered token count.")


# -------- PROMPT CONTEXT -----------------------------------------------------------
class PromptCtx(BaseModel):

    """
    Everything the client needs to assemble the prompt for a single turn.

    Built by ``Agent`` after ``prepare()`` has finished rendering
    every layer of the stack. The client reads ``rendered_layers`` to
    compose its provider-native system prompt and uses ``stack`` /
    ``variables`` only when it needs to inspect the source layers
    (debugging, telemetry, conditional re-rendering).
    """

    model_config = {"arbitrary_types_allowed": True}

    stack: LayerContainer = Field(
        ...,
        description="Validated container of prompt layers (source of truth).",
    )
    variables: dict[str, t.Any] = Field(
        default_factory=dict,
        description="Variables that were used to render the layers.",
    )
    rendered_layers: dict[type[CoreLayer], str] = Field(  # type: ignore
        default_factory=dict,
        description=(
            "Output of each layer in the stack, keyed by the concrete "
            "CoreLayer subclass. Populated by agent.prepare(). Iteration "
            "order matches the underlying stack's insertion order."
        ),
    )
    layer_usage: dict[str, PromptLayerUsage] = Field(
        default_factory=dict,
        description="Token and size stats for each rendered layer.",
    )
    prompt_tokens: int = Field(
        default=0,
        description="Total token count for all rendered prompt layers.",
    )
