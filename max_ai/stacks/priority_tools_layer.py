"""
Layer to Config High-Priority Tools the Agent Should Prefer
"""
import typing as t
from pathlib import Path

from ..core.models import StackConfig
from ..base.component import Component

from ..base.layer import CoreLayer


class PriorityToolsLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer that marks a subset of the agent's tools as high-priority.

    Semantics:
        "Priority" does NOT mean "always call". It means the agent should
        strongly prefer these tools over general knowledge WHEN the user's
        query is relevant to them. Relevance is still the deciding factor.

    Template Variables:
        priority_tools (optional): List of tool names (strings) flagged as
            priority. When omitted or empty, the whole layer renders as an
            empty string — no priority-tools block is added to the prompt.

    Example:
        layer = PriorityToolsLayer()

        prompt = layer.render({
            "priority_tools": ["search_inventory", "check_order_status"],
        })

        # No priority tools — layer renders nothing
        prompt = layer.render({"priority_tools": []})
    """

    component_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "maxai.stacks.PriorityToolsLayer"

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        """
        Args:
            template: Inline Jinja2 template string referencing
                `priority_tools`. If None, the default is used.
            load_from: Path to a template file (.j2). If None, the default
                is used.
        """
        super().__init__(
            name="PriorityToolsLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables,
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")

    def _optional_variables(self) -> set[str]:
        return {"priority_tools"}