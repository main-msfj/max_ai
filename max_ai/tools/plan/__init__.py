"""Plan data model and the native update_plan tool.

``AgentPlan``/``PlanStep`` are re-exported eagerly here: they're a pure
Pydantic data model with no dependency on ``base``/``core``, and
``core.event_type``, ``types.run_context`` and ``base.reasoning`` import
``AgentPlan`` directly (see ``max_ai.tools.__init__`` for the same
constraint on this package). Keep this eager import free of anything
heavier than the model itself.

``UpdatePlanTool`` and ``AgentUpdatePlanTool`` are exported lazily (resolved
on first attribute access) because both pull in ``core.event_type`` and/or
``base.reasoning``/``base.tools``. Importing either eagerly here would close
the cycle: ``core.event_type`` → ``tools.plan`` → (either tool module) →
``core.event_type`` (and, for ``UpdatePlanTool``, via ``base.reasoning``
too).

``UpdatePlanTool`` (``_tool.py``) is built for ``ReActLoopSelfDirected``/
``react.py`` — stages the plan on ``loop_state.plan_draft``, the loop syncs
it to ``ctx.plan``. ``AgentUpdatePlanTool`` (``_agent_tool.py``) is the
``Agent``-stack variant — no ``loop_state`` there, so it reads/writes
``RunContext.plan`` directly via ``ToolContext.deps["run_context"]`` and
emits ``PlanningEvent`` itself. Deliberately separate classes (Option B),
not a shared rewrite — see ``_agent_tool.py``'s module docstring for why.
"""

import importlib
import typing as t

from ._model import AgentPlan, PlanStep

__all__ = ["AgentPlan", "PlanStep", "UpdatePlanTool", "AgentUpdatePlanTool"]

if t.TYPE_CHECKING:  # static analyzers see the real symbols
    from ._tool import UpdatePlanTool
    from ._agent_tool import AgentUpdatePlanTool


def __getattr__(name: str) -> t.Any:
    if name == "UpdatePlanTool":
        return importlib.import_module(f"{__name__}._tool").UpdatePlanTool
    if name == "AgentUpdatePlanTool":
        return importlib.import_module(f"{__name__}._agent_tool").AgentUpdatePlanTool
    raise AttributeError(name)
