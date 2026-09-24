"""Layer with the session state the transcript may no longer hold."""

import typing as t
from pathlib import Path

from ...base.component import Component
from ...base.layer import CoreLayer
from ...core.model.stacks import StackConfig


class SessionStateLayer(Component[StackConfig], CoreLayer):
    """Renders what must survive compaction: the compaction summary and the
    current plan. The reasoning loop refreshes these variables before every
    model call, so the layer follows the session within a turn.

    Template Variables:
        compaction_summary (optional): ``CoreCompaction.render(state)`` output.
        current_plan (optional): the plan steps as text.
    """

    component_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "maxai.stacks.SessionStateLayer"

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        """Initialize ``SessionStateLayer``.

Parameters
----------
template : str | None
    Value supplied for ``template``.
load_from : str | Path | None
    Value supplied for ``load_from``.
extra_variables : dict[str, t.Any] | None
    Value supplied for ``extra_variables``."""
        super().__init__(
            name="SessionStateLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables,
        )

    def _default_template(self) -> str:
        """Perform the internal ``default template`` operation for ``SessionStateLayer``."""
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")

    def _optional_variables(self) -> set[str]:
        """Perform the internal ``optional variables`` operation for ``SessionStateLayer``."""
        return {"compaction_summary", "current_plan"}
