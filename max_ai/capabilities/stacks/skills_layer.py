"""Layer that lists the skills and how to read them."""
import typing as t
from pathlib import Path

from ...base.component import Component
from ...base.layer import CoreLayer
from ...core.model.stacks import StackConfig


class SkillsLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer that lists the skills and says how to read one. Renders
    nothing when there are no skills.

    Template Variables:
        skills (required): ``{"name", "description"}`` per skill.
        skill_tool (required): name of the tool that reads a skill.
        bash_tool (optional): name of the tool that runs commands.
    """

    component_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "max_ai.capabilities.stacks.SkillsLayer"

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        super().__init__(
            name="SkillsLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")

    def _required_variables(self) -> set[str]:
        return {"skills", "skill_tool"}

    def _optional_variables(self) -> set[str]:
        return {"bash_tool"}
