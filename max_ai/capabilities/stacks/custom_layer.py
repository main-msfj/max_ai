"""Layer with the text the user writes for their agent."""
import typing as t
from pathlib import Path

from ...base.component import Component
from ...base.layer import CoreLayer
from ...core.model.stacks import StackConfig


class CustomLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer with text the user writes: rules of the business, tone,
    domain knowledge. The text is shown as written, never run as a template.

    Template Variables:
        content: the user's text; given when the layer is created.
    """

    component_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "max_ai.capabilities.stacks.CustomLayer"

    def __init__(
        self,
        content: str = "",
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        super().__init__(
            name="CustomLayer",
            template=template,
            load_from=load_from,
            extra_variables={"content": content, **(extra_variables or {})},
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")
