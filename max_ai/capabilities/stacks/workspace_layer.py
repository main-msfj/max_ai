"""Layer that tells the model where it works. Only for agents with a sandbox."""
import typing as t
from pathlib import Path

from ...base.component import Component
from ...base.layer import CoreLayer
from ...core.model.stacks import StackConfig


class WorkspaceLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer with the folders of the sandbox. Renders nothing when no
    ``workspace`` is given, so it can always be in the stack.

    Template Variables (``sandbox.session_paths(session_id)`` gives both):
        workspace (optional): the session's working directory.
        tmp (optional): the session's folder for temporary files.
    """

    component_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "maxai.stacks.WorkspaceLayer"

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        super().__init__(
            name="WorkspaceLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables,
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")

    def _optional_variables(self) -> set[str]:
        return {"workspace", "tmp"}
