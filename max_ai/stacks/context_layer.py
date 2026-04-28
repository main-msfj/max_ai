"""
Layer that injects the current session's summary into the system prompt.
"""

import typing as t
from pathlib import Path

from ..core.models import StackConfig
from ..base.component_config import Component

from ..base.layer import CoreLayer


class ContextLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer that surfaces continuity context for the active session.

    On session resume, the framework asks the configured
    ``CoreContextRegistry`` for the stored summary of the current session
    (via ``get_current_session_summary()``) and feeds it into this layer
    as ``current_session_summary``. The result is rendered as a small
    background block at the top of the system prompt so the agent can
    pick up where it left off without re-asking what was already covered.

    If no summary exists (fresh session, or a session that hasn't been
    summarized yet), the layer renders to an empty string.

    Cross-session recall (e.g. "remember when we talked about X last
    week") is NOT handled here — that goes through the ``search_context``
    tool exposed by the registry. This layer only carries the current
    session's continuity.

    Template Variables:
        current_session_summary (optional): Stored summary of the active
            session, returned by ``CoreContextRegistry.get_current_session_summary()``.
            ``None`` or an empty string causes the layer to render empty.
    """
    
    component_config_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "maxai.stacks.ContextLayer"

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        """
        Args:
            template: Inline Jinja2 template string referencing
                ``current_session_summary``. If None, the default is used.
            load_from: Path to a template file (.j2). If None, the default
                is used.
            extra_variables: Additional variables to merge into every
                render() call. Useful when overriding the template with
                one that requires custom variables.
        """
        super().__init__(
            name="ContextLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables,
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")

    def _optional_variables(self) -> set[str]:
        return {"current_session_summary"}