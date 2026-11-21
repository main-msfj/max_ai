"""Layer that shows the user's memory kept as files. Only for agents with memory."""
import typing as t
from pathlib import Path

from ...base.component import Component
from ...base.layer import CoreLayer
from ...core.model.stacks import StackConfig


class MemoryFilesLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer with the memory kept as files in the sandbox. Renders
    nothing when no ``memory`` is given, so it can always be in the stack.

    Template Variables:
        memory (optional): what ``Memory.block(sandbox, index)`` gives: where
            the memory is, how to use it and one line per file.
    """

    component_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "max_ai.capabilities.stacks.MemoryFilesLayer"

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        super().__init__(
            name="MemoryFilesLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables,
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")

    def _optional_variables(self) -> set[str]:
        return {"memory"}
