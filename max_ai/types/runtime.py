import typing as t
from ..core.blocks import RunTimeBlock


# -------- RUNTIME STATE -----------------------------------------------------------
class RuntimeState(RunTimeBlock):
    """Execution context. Extends the block with operations and loaders."""

    @classmethod
    def load_from(cls, source: dict[str, t.Any] | None) -> t.Self:
        """Create an RuntimeState from a dict (e.g. from JSON/DB)."""
        return cls.model_validate(source or {})
