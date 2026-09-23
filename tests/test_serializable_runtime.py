"""Reasoning, guards and the runtime gate are data; providers are allowlisted."""

from __future__ import annotations

import pytest

from max_ai.base import component
from max_ai.base.reasoning import BaseReasoning, ReasoningConfig
from max_ai.capabilities.completion_gate import RuntimeCompletionGate, RuntimeGateConfig
from max_ai.capabilities.reasoning.guards import (
    BudgetGuard,
    GuardConfig,
    LoopGuard,
    RepetitionGuard,
)
from max_ai.capabilities.reasoning.react import ReactLoop
from max_ai.capabilities.tools.plan import AgentPlan
from max_ai.core.messages import AssistantMessage
from max_ai.types.run_context import RunContext


@pytest.fixture(autouse=True)
def fresh_allowlist(monkeypatch):
    monkeypatch.setattr(component, "_ALLOWED_PREFIXES", {"max_ai."})
    monkeypatch.delenv("MAXAI_ALLOWED_PROVIDERS", raising=False)


# -------- reasoning & guards -----------------------------------------------------------
def test_default_loop_keeps_following_the_defaults():
    config = ReactLoop(max_loop_iterations=20).serialize()
    assert config.provider == "maxai.reasoning.ReactLoop"
    assert config.config == {"max_connection_retries": 3, "max_loop_iterations": 20}
    loop = ReactLoop.deserialize(config.model_dump_json())
    assert loop.max_loop_iterations == 20
    assert [type(g).__name__ for g in loop.guards] == ["SchemaRetryGuard", "RepetitionGuard", "BudgetGuard"]


def test_custom_guards_round_trip_with_their_settings():
    loop = ReactLoop(max_loop_iterations=7, guards=[RepetitionGuard(max_repeats=4), BudgetGuard(0.5)])
    back = BaseReasoning.deserialize(loop.serialize().model_dump_json())
    assert isinstance(back, ReactLoop) and back.max_loop_iterations == 7
    assert (back.guards[0].max_repeats, back.guards[1].threshold) == (4, 0.5)
    assert ReactLoop.deserialize(ReactLoop(guards=[]).serialize()).guards == []


class MyGuardConfig(GuardConfig):
    word: str = "stop"


class MyGuard(LoopGuard):
    component_schema = MyGuardConfig

    def __init__(self, word: str = "stop") -> None:
        self.word = word


class MyLoopConfig(ReasoningConfig):
    depth: int = 2


class MyLoop(BaseReasoning):
    """A third-party reasoning strategy."""

    component_schema = MyLoopConfig

    def __init__(self, depth: int = 2, max_connection_retries: int = 3) -> None:
        super().__init__(max_connection_retries=max_connection_retries)
        self.depth = depth

    def _to_config(self) -> MyLoopConfig:
        return MyLoopConfig(depth=self.depth, max_connection_retries=self.max_connection_retries)

    @classmethod
    def _from_config(cls, config: MyLoopConfig) -> "MyLoop":
        return cls(**config.model_dump())

    async def execute_reasoning_loop(self, **kwargs):  # pragma: no cover - not run here
        yield None


def test_third_party_components_need_to_be_allowed():
    loop_json = MyLoop(depth=5).serialize().model_dump_json()
    guard_json = MyGuard("halt").serialize().model_dump_json()
    with pytest.raises(PermissionError, match="allow_providers"):
        BaseReasoning.deserialize(loop_json)

    component.allow_providers("tests.")
    assert BaseReasoning.deserialize(loop_json).depth == 5
    assert LoopGuard.deserialize(guard_json).word == "halt"


def test_allowlist_from_env_and_wildcard(monkeypatch):
    loop_json = MyLoop().serialize().model_dump_json()
    monkeypatch.setenv("MAXAI_ALLOWED_PROVIDERS", "other_pkg., tests.")
    assert isinstance(BaseReasoning.deserialize(loop_json), MyLoop)
    monkeypatch.delenv("MAXAI_ALLOWED_PROVIDERS")
    component.allow_providers("*")
    assert isinstance(BaseReasoning.deserialize(loop_json), MyLoop)


def test_a_forged_provider_is_refused_before_importing_anything():
    forged = {"provider": "os.system", "component_type": "reasoning", "config": {}}
    with pytest.raises(PermissionError, match="not allowed"):
        BaseReasoning.deserialize(forged)


# -------- runtime gate options ---------------------------------------------------------
def closing_with_open_plan() -> RunContext:
    ctx = RunContext(messages=[AssistantMessage(source="a", content="listo")])
    ctx.plan = AgentPlan(rationale="r", steps=[{"id": 1, "description": "x", "status": "active"}])
    return ctx


def test_gate_options_turn_checks_off():
    strict = RuntimeCompletionGate(workspace=None)
    assert strict.on_final_response(closing_with_open_plan()).status == "incomplete"

    relaxed = RuntimeCompletionGate(workspace=None, config=RuntimeGateConfig(plan_must_close=False))
    assert relaxed.on_final_response(closing_with_open_plan()).status == "completed"

    off = RuntimeCompletionGate(workspace=None, config=RuntimeGateConfig(enabled=False))
    empty = RunContext(messages=[AssistantMessage(source="a", content="")])
    assert off.on_final_response(empty).status == "completed"


def test_gate_options_are_plain_config():
    gate = RuntimeCompletionGate(workspace=None, config=RuntimeGateConfig(check_bash_outputs=False))
    assert gate.serialize().config == {
        "enabled": True, "plan_must_close": True,
        "check_bash_outputs": False, "nudge_bash_failures": True,
    }
