"""
Smoke test for Agent.

Goal: prove the agent can be constructed with a real-ish set of
capabilities, prepare() runs end-to-end, and rendered_layers contains
non-empty content for layers that should produce output.

The chat completion client is stubbed — no LLM call happens. We only
exercise __init__ + prepare(), which is what's been built so far.
"""

from __future__ import annotations

import json
import textwrap
import typing as t
from pathlib import Path

import pytest

from max_ai.base.clients import CoreChatCompletionClient
from max_ai.base.agent import Agent
from max_ai.core.models import ModelConfig
from max_ai.errors.agent import AgentError
from max_ai.types.completions import Usage
from max_ai.types.run_context import RunContext

from max_ai.stacks.memory_layer import MemoryLayer
from max_ai.stacks.skills_layer import SkillsLayer
from max_ai.stacks.context_layer import ContextLayer
from max_ai.stacks.routine_layer import RoutineLayer
from max_ai.stacks.knowledge_layer import KnowledgeLayer
from max_ai.stacks.agent_policy_layer import AgentPolicyLayer
from max_ai.stacks.priority_tools_layer import PriorityToolsLayer

from max_ai.capabilities.memory import LocalMemoryRegistry
from max_ai.capabilities.context import LocalContextRegistry
from max_ai.capabilities.knowledge import LocalKnowledgeRegistry
from max_ai.capabilities.routines import LocalRoutineRegistry
from max_ai.capabilities.skills.local import LocalSkillRegistry
from max_ai.capabilities.workspace import WorkspaceLocal


# =====================================================================
# Test client stub
# =====================================================================
class StubClient(CoreChatCompletionClient):
    """Minimal client that satisfies the ABC. Never makes an API call."""

    def __init__(self) -> None:
        super().__init__(model="stub-model", api_key=None, config=ModelConfig())

    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        return Usage()

    def build_api_messages(self, messages):
        return []

    def format_messages(self, ctx, prompts):
        return []

    def adapt_schema_for_provider(self, schema):
        return schema

    def build_tool_schema(self, tools):
        return []

    async def complete(self, messages, tools, output_format, **kwargs):
        raise NotImplementedError("StubClient does not actually call any LLM.")

    async def stream(self, messages, tools, output_format, **kwargs):
        raise NotImplementedError("StubClient does not actually call any LLM.")
        yield  # makes this a generator


# =====================================================================
# Concrete agent for testing (Agent is abstract — needs subclass)
# =====================================================================
class _DummyAgent(Agent):
    """Minimal subclass to allow instantiation. run() is unused here."""

    async def run(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


# =====================================================================
# Helpers — build source dirs for the various local registries
# =====================================================================
def _setup_skill_source(tmp_path: Path) -> Path:
    skill_root = tmp_path / "skills"
    skill_dir = skill_root / "demo_skill"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "references").mkdir(parents=True)

    (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: demo_skill
        description: A minimal skill for the smoke test.
        ---

        Use this skill to demo things.
        """))

    (skill_dir / "scripts" / "actions.py").write_text(textwrap.dedent("""\
        def do_thing(x: str) -> str:
            \"\"\"Do a thing.

            Args:
                x: Input.
            \"\"\"
            return x
        """))
    return skill_root


def _setup_routine_source(tmp_path: Path) -> Path:
    routine_root = tmp_path / "routines_root"
    routines_dir = routine_root / "routines"
    routines_dir.mkdir(parents=True)
    (routines_dir / "client_followup.json").write_text(json.dumps({
        "name": "client_followup",
        "description": "Follow up with a client.",
        "instructions": "1. Reach out. 2. Restate value.",
    }))
    return routine_root


def _setup_knowledge_source(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "docs.json").write_text(json.dumps([
        {
            "content": "FastAPI is a Python framework.",
            "score": None,
            "tokens": 6,
            "metadata": {},
        },
    ]))


def _setup_memory_source(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "u1.json").write_text(json.dumps({
        "user_identity": {
            "category": "user_identity",
            "content": "Software engineer in Buenos Aires.",
            "last_updated": "2026-04-25T10:00:00+00:00",
        },
    }))


def _setup_context_source(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "u1.json").write_text(json.dumps({
        "session_001": {
            "summary": "We discussed spaceship designs.",
            "vector": [],
            "timestamp": "2026-04-20T15:00:00Z",
            "metadata": {},
        },
    }))


# =====================================================================
# THE TEST
# =====================================================================
@pytest.mark.asyncio
async def test_core_agent_prepares_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Build a Agent with all capabilities, prepare it, validate
    rendered_layers and tools."""

    # 1. Setup all source dirs.
    skill_root = _setup_skill_source(tmp_path)
    routine_root = _setup_routine_source(tmp_path)
    _setup_knowledge_source(tmp_path)
    _setup_memory_source(tmp_path)
    _setup_context_source(tmp_path)
    monkeypatch.setenv("SKILLS_CACHE_DIR", str(tmp_path / "skills-cache"))
    monkeypatch.setenv("SERVER_DIR", str(tmp_path / "server"))

    # 2. Build registries.
    memory = LocalMemoryRegistry(user_id="u1", base_path=tmp_path)
    context = LocalContextRegistry(
        user_id="u1", session_id="session_001", base_path=tmp_path
    )
    knowledge = LocalKnowledgeRegistry(
        name="docs", description="Internal docs.", base_path=tmp_path
    )
    routines = LocalRoutineRegistry(
        source_path=routine_root,
        routines=["client_followup"],
    )
    skills = LocalSkillRegistry(
        source=skill_root,
        skills=["demo_skill"],
    )
    workspace = WorkspaceLocal(root=tmp_path / "server")

    # 3. Build the agent.
    agent = _DummyAgent(
        name="smoke_agent",
        description="Agent used to smoke-test prepare().",
        instructions="Be helpful and accurate.",
        client=StubClient(),
        memory=memory,
        skills=skills,
        logbook=context,
        routines=routines,
        knowledge=[knowledge],
        workspace=workspace,
    )

    # Sanity: not yet prepared.
    assert agent.is_prepared is False

    # 4. Prepare.
    await agent.prepare()
    assert agent.is_prepared is True

    # 5. Validate rendered_layers.
    rendered = agent.rendered_layers

    # Agent policy layer must contain agent metadata.
    policy = rendered[AgentPolicyLayer]
    assert "smoke_agent" in policy
    assert "Be helpful and accurate." in policy

    # Memory layer must contain the seeded memory.
    memory_out = rendered[MemoryLayer]
    assert "Software engineer in Buenos Aires" in memory_out
    assert "[user_identity]" in memory_out

    # Context layer must contain the session summary.
    context_out = rendered[ContextLayer]
    assert "spaceship designs" in context_out

    # Knowledge layer must mention the search tool.
    knowledge_out = rendered[KnowledgeLayer]
    assert "search_docs" in knowledge_out

    # Routine layer must mention the routine tools.
    routine_out = rendered[RoutineLayer]
    assert "search_routines" in routine_out
    assert "get_routine" in routine_out

    # Skills layer must mention the loaded skill.
    skills_out = rendered[SkillsLayer]
    assert "demo_skill" in skills_out
    assert "A minimal skill for the smoke test." in skills_out
    assert "search_skills" not in skills_out
    assert "Call the `bash` tool to inspect" in skills_out
    assert "The `bash` tool is the ONLY way to engage a skill" in skills_out
    assert "`demo_skill`" not in skills_out
    assert "$SKILLS_DIR" not in skills_out
    assert "SKILL.md" not in skills_out

    # PriorityTools layer renders empty (no priority tools given).
    assert rendered[PriorityToolsLayer].strip() == ""

    # 6. Validate tools available to the LLM.
    tool_names = {t.name for t in agent.capabilities.all_tools}
    assert "list_memories" in tool_names
    assert "update_memory" in tool_names
    assert "delete_memory" in tool_names
    assert "search_docs" in tool_names
    assert "search_routines" in tool_names
    assert "get_routine" in tool_names
    assert "search_context" in tool_names
    assert "workspace" in tool_names
    assert "bash" in tool_names
    assert "search_skills" not in tool_names
    assert "skill_bash" not in tool_names
    assert "do_thing" not in tool_names
    assert "read_skill_resource" not in tool_names

    # 7. prepare() is idempotent.
    await agent.prepare()  # must not raise
    assert agent.is_prepared is True

    # 8. Skills cannot run on LocalExecutor because they require a sandbox.
    ctx = RunContext(user_id="u1")
    with pytest.raises(AgentError, match="skills cannot run with LocalExecutor"):
        async for _ in agent.run_stream_events("hello", run_context=ctx):
            pass
