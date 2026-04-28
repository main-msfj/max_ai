"""
Layer to Config Agent Task Analysis and Planning
"""
import typing as t
from pathlib import Path

from ..core.models import StackConfig
from ..base.component_config import Component

from ..base.layer import CoreLayer


class TaskAnalysisLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer that tells the agent how to analyze a task before acting:
    when to proceed directly, when to ask clarification, and when to plan.

    This layer has no template variables — the content is fully static
    instructions. It is always included in the prompt as-is.
    """

    component_config_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "maxai.stacks.TaskAnalysisLayer"

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        """
        Args:
            template: Inline Jinja2 template string. If None, the default
                is used. No variables are required.
            load_from: Path to a template file (.j2). If None, the default
                is used.
        """
        super().__init__(
            name="TaskAnalysisLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables,
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")