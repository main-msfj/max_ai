"""Layer with what changes while a conversation lives. Always the last one."""
import typing as t
from pathlib import Path

from ...base.component import Component
from ...base.layer import CoreLayer
from ...core.model.stacks import StackConfig


class SessionLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer with the date and the summary of the conversation. It goes
    last: when one of them changes, only the end of the prompt changes.

    Template Variables:
        current_date (optional): today's date.
        summary (optional): the summary a compaction left.
    """

    component_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "maxai.stacks.SessionLayer"

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        super().__init__(
            name="SessionLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables,
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")

    def _optional_variables(self) -> set[str]:
        return {"current_date", "summary"}
