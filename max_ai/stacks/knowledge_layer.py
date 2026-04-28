"""
Layer to Config Agent Retrieval-Augmented Behaviour
"""
import typing as t
from pathlib import Path

from ..core.models import StackConfig
from ..base.component_config import Component

from ..base.layer import CoreLayer


class KnowledgeLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer for retrieval-augmented generation (RAG).

    Lists the names of the retrieval tools the agent can call. The
    actual descriptions of each tool — when to use it, what it covers —
    live on the tool itself (each ``CoreKnowledgeRegistry`` injects its
    own description into the tool the LLM sees). The layer just tells
    the model these tools exist and gives it general guidance on how to
    integrate retrieved content into responses.

    Retrieved data itself is NOT rendered here. Search results are
    per-turn artefacts and belong in the message stream, not in the
    static system prompt.

    If no retrieval tools are configured, the entire layer renders to
    an empty string — the agent has no retrieval surface, so the
    instructions would be noise.

    Template Variables:
        retrieval_tools (optional): List of tool names the agent uses to
            perform retrieval searches. When absent or empty, the whole
            layer is rendered as an empty string.

    Example:
        layer = KnowledgeLayer()

        # Multiple knowledge sources
        prompt = layer.render({
            "retrieval_tools": ["search_docs", "search_tickets"],
        })

        # No retrieval — layer disappears
        prompt = layer.render({"retrieval_tools": []})
    """

    component_config_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "maxai.stacks.KnowledgeLayer"

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        """
        Args:
            template: Inline Jinja2 template string referencing
                ``retrieval_tools``. If None, the default is used.
            load_from: Path to a template file (.j2). If None, the
                default is used.
            extra_variables: Additional variables to merge into every
                render() call. Useful when overriding the template with
                one that requires custom variables.
        """
        super().__init__(
            name="KnowledgeLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables,
        )

    def _default_template(self) -> str:
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")

    def _optional_variables(self) -> set[str]:
        return {"retrieval_tools"}