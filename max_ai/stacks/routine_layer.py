"""
Layer to Config Agent Routines (Search-Based)
"""
import typing as t 
from pathlib import Path

from ..core.models import StackConfig
from ..base.component import Component

from ..base.layer import CoreLayer


class RoutineLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer that tells the agent how to discover and load routines
    through tools, without embedding any routine content in the prompt.

    The agent does NOT see the routine catalog in the prompt. Instead,
    it is told that:
      1. A semantic search tool exists over the routine catalog.
      2. A fetch tool exists to load a specific routine's full instructions.

    This scales to hundreds or thousands of routines at zero prompt-token
    cost — the agent only loads what's needed, when it's needed.

    Template Variables:
        routine_search_tool (optional): Name of the tool used to search
            the routine catalog (returns candidate routines with
            `name + description`). When omitted, the entire layer renders
            as an empty string — the agent has no access to routines.
        routine_fetch_tool (optional): Name of the tool used to fetch the
            full instructions of a specific routine by name. If omitted,
            the workflow assumes the search tool returns full content
            directly (single-tool design).

    Example:
        layer = RoutineLayer()

        # Two-tool design (recommended for scale)
        prompt = layer.render({
            "routine_search_tool": "search_routines",
            "routine_fetch_tool": "get_routine",
        })

        # Single-tool design (search returns full content)
        prompt = layer.render({
            "routine_search_tool": "search_routines",
        })

        # No routines — layer disappears from the prompt
        prompt = layer.render({})
    """

    component_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "maxai.stacks.RoutineLayer"

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        """
        Args:
            template: Inline Jinja2 template string referencing
                `routine_search_tool` and optionally `routine_fetch_tool`.
                If None, the default is used.
            load_from: Path to a template file (.j2). If None, the default
                is used.
        """
        super().__init__(
            name="RoutineLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables,
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")

    def _optional_variables(self) -> set[str]:
        return {"routine_search_tool", "routine_fetch_tool"}