"""
Layer to Config Agent Output Rendering Rules
"""
import typing as t
from pathlib import Path

from ..core.models import StackConfig
from ..base.component_config import Component

from ..base.layer import CoreLayer


class RenderingLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer that defines how the agent must format its final output
    so it renders correctly in the UI.

    Covers plain text, Markdown links, fenced code blocks with language
    tags, tables, lists, ASCII/Mermaid diagrams, math, emphasis, quotes,
    and headings — plus what to avoid.

    This layer has no template variables — the content is fully static
    instructions. It is always included in the prompt as-is.
    """

    component_config_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "maxai.stacks.RenderingLayer"

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
            name="RenderingLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables,
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")