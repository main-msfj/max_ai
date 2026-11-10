"""Layer that lists the deferred tools and how to run them."""

import typing as t
from pathlib import Path

from ...base.component import Component
from ...base.layer import CoreLayer
from ...core.model.stacks import StackConfig


class ToolsLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer that lists the tools not sent to the model and explains
    how to reach them: ``search_tools`` for their parameters, ``call_tool``
    to run them. Renders nothing when there are no deferred tools.

    Template Variables:
        deferred_tools (required): ``{"name", "description"}`` per tool.
        search_tool, call_tool (required): the two tools' names.
    """

    component_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "maxai.stacks.ToolsLayer"

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        super().__init__(
            name="ToolsLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables,
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")

    def _required_variables(self) -> set[str]:
        return {"deferred_tools", "search_tool", "call_tool"}
