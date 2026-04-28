"""
Layer to Config Agent Policy and Behaviours
"""
import typing as t
from pathlib import Path

from ..core.models import StackConfig
from ..base.component_config import Component

from ..base.layer import CoreLayer


class AgentPolicyLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer for agent identity and behavior.

    Template Variables:
        name: The agent's display name.
        description: The agent's specific role or area of expertise.
        instructions: Core behavioral guidelines and constraints.

    Example:
        Typical template structure (Jinja2, saved as .j2):

        .. code-block:: jinja

            <identity>
                <name>{{ name }}</name>
                <description>{{ description }}</description>
                <instructions>
                {% if instructions is string %}
                    {{ instructions }}
                {% else %}
                    {% for item in instructions %}
                    - {{ item }}
                    {% endfor %}
                {% endif %}
                </instructions>
            </identity>
    """
    component_config_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "maxai.stacks.AgentPolicyLayer"
    

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        """
        Args:
            template: Inline Jinja2 template string referencing `name`,
                `description`, `instructions`. If None, the default is used.
            load_from: Path to a template file (.j2). If None, the default
                is used.
        """
        super().__init__(
            name="AgentPolicyLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables,
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")

    def _required_variables(self) -> set[str]:
        return {"name", "description", "instructions"}
