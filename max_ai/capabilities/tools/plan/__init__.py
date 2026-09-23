"""Plan data model and the native update_plan tool.

``AgentPlan``/``PlanStep`` are re-exported eagerly: a pure Pydantic model
that ``core.event_type``, ``types.run_context`` and ``base.reasoning``
import directly, so keep this eager import free of anything heavier.

``AgentUpdatePlanTool`` is exported lazily: it imports ``core.event_type``,
which imports this package, so loading it eagerly would close the cycle.
It reads/writes ``RunContext.plan`` through
``ToolContext.deps["run_context"]`` and emits ``PlanningEvent`` itself.
"""

import importlib
import typing as t

from ._model import AgentPlan, PlanStep

__all__ = ["AgentPlan", "PlanStep", "AgentUpdatePlanTool"]

if t.TYPE_CHECKING:  # static analyzers see the real symbols
    from ._agent_tool import AgentUpdatePlanTool


def __getattr__(name: str) -> t.Any:
    if name == "AgentUpdatePlanTool":
        return importlib.import_module(f"{__name__}._agent_tool").AgentUpdatePlanTool
    raise AttributeError(name)
