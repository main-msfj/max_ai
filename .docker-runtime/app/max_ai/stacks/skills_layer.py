"""
Layer to inject loaded skills into the agent's system prompt.
"""
import typing as t
from pathlib import Path

from ..core.models import StackConfig
from ..base.component import Component

from ..base.layer import CoreLayer


class SkillsLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer that renders the lightweight catalog of available
    skills.

    Skills are self-contained packages (instructions, scripts, reference
    files) resolved from a ``CoreSkillRegistry`` during
    ``agent.prepare()``. This layer receives ``SkillBlock`` entries with
    only ``name`` and ``description`` populated. Full instructions,
    references, scripts, and assets stay inside the materialized skill
    directory and are handled by the skill runtime.

    If the loaded-skills list is empty, the layer renders to an empty
    string — it always lives in the stack, but contributes nothing
    when there's nothing to advertise.

    Template Variables:
        loaded_skills (required): List of lightweight ``SkillBlock``
            objects from ``AgentCapabilities.loaded_skill_blocks``.

    Example:
        layer = SkillsLayer()

        # With skills loaded
        prompt = layer.render({"loaded_skills": registry.loaded_skill_blocks})

        # No skills — layer disappears from the prompt
        prompt = layer.render({"loaded_skills": []})
    """

    component_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "maxai.stacks.SkillsLayer"

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        """
        Args:
            template: Inline Jinja2 template string referencing
                ``loaded_skills``. If None, the default is used.
            load_from: Path to a template file (.j2). If None, the
                default is used.
        """
        super().__init__(
            name="SkillsLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")

    def _required_variables(self) -> set[str]:
        return {"loaded_skills"}
