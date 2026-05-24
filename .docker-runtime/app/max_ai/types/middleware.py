import typing as t
from pydantic import BaseModel, Field

if t.TYPE_CHECKING:
    from .run_context import RunContext


# -------- MIDDLE WARE -----------------------------------------------------------
class MiddlewareCtx(BaseModel):
    """Runtime context passed through the middleware pipeline."""

    ctx: "RunContext" = Field(description="Active Running context")
    data: t.Any = Field(description="Input payload for the current action")
    action: str = Field(description="Execution stage identifier")
    metadata: dict[str, t.Any] = Field(default_factory=dict)

from .run_context import RunContext  # noqa: E402
MiddlewareCtx.model_rebuild()