"""
Smoke tests for AgentCapabilities.

The registry's job is to validate and unify the agent's runtime
components. These tests exercise:
  - Sync construction: explicit tools, memory/knowledge/routines/context,
    priority_tools coherence, name uniqueness.
  - Async prepare(): skill loading, post-load tool revalidation, idempotency.
  - Tool aggregation: every capability exposes its tools through the
    expected property and they all flow into all_tools.
  - Failure modes: duplicate tool names, missing priority_tools, accessing
    all_tools before prepare().
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from max_ai.manager.registry import AgentCapabilities
from max_ai.errors.capabilities import CapabilityError

from max_ai.tools import FunctionAsTool

from max_ai.base.memory import MemoryToolMode
from max_ai.base.context import LogBookToolMode
from max_ai.base.routines import RoutineToolMode

from max_ai.capabilities.memory import LocalMemoryRegistry
from max_ai.capabilities.context import LocalContextRegistry
from max_ai.capabilities.knowledge import LocalKnowledgeRegistry
from max_ai.capabilities.routines import LocalRoutineRegistry
from max_ai.capabilities.skills import LocalSkillRegistry  


# =====================================================================
# Helpers
# =====================================================================
def _make_tool(name: str = "ping") -> FunctionAsTool:
    """Build a trivial FunctionAsTool for tests."""

    async def fn(message: str) -> str:
        """Echo the message back.

        Args:
            message: Anything.
        """
        return message

    # Override the function name via FunctionAsTool's name kwarg if the
    # default uses fn.__name__. We assume FunctionAsTool accepts a name
    # override; if not, rename fn directly.
    fn.__name__ = name
    return FunctionAsTool(fn)


def _build_routine_source(tmp_path: Path) -> Path:
    """Create a routines source dir with two authorized routines."""
    routines_dir = tmp_path / "routines"
    routines_dir.mkdir(parents=True)
    for name in ("client_followup", "email_reply"):
        (routines_dir / f"{name}.json").write_text(json.dumps({
            "name": name,
            "description": f"{name} routine.",
            "instructions": "1. Do X. 2. Do Y.",
        }))
    return tmp_path


def _build_skill_source(tmp_path: Path, skill_name: str = "demo_skill") -> Path:
    """Create a skill source dir with one minimal skill."""
    skill_dir = tmp_path / skill_name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "references").mkdir(parents=True)

    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {skill_name}
        description: A minimal skill for testing.
        ---

        Use this skill when testing.
        """))

    (skill_dir / "scripts" / "actions.py").write_text(textwrap.dedent("""\
        def do_thing(x: str) -> str:
            \"\"\"Do a thing.

            Args:
                x: Input value.
            \"\"\"
            return x
        """))
    return tmp_path


# =====================================================================
# Construction & sync validation
# =====================================================================
def test_empty_registry_constructs():
    """No arguments — the registry is valid and empty."""
    cr = AgentCapabilities()
    assert cr.has_memory is False
    assert cr.has_knowledge is False
    assert cr.has_routines is False
    assert cr.has_skills is False
    assert cr.has_context is False
    assert cr.has_tools is False
    assert cr._all_tools_sync() == []


def test_explicit_tools_normalized():
    """Plain callables get wrapped, CoreTool instances pass through."""
    tool = _make_tool("ping")

    async def raw_callable(x: str) -> str:
        """Echo.

        Args:
            x: Anything.
        """
        return x
    raw_callable.__name__ = "echo"

    cr = AgentCapabilities(tools=[tool, raw_callable])
    assert len(cr.tools) == 2
    assert {t.name for t in cr.tools} == {"ping", "echo"}


def test_invalid_tool_entry_rejected():
    """Non-callable, non-CoreTool entries fail loud."""
    with pytest.raises(CapabilityError):
        AgentCapabilities(tools=[42])  # type: ignore[list-item]


def test_priority_tools_must_reference_real_tool():
    """priority_tools entries that don't match any tool name fail at construction."""
    tool = _make_tool("ping")
    with pytest.raises(CapabilityError):
        AgentCapabilities(
            tools=[tool],
            priority_tools=["does_not_exist"],
        )


def test_priority_tools_accepted_when_present():
    """priority_tools matching a real tool passes."""
    tool = _make_tool("ping")
    cr = AgentCapabilities(tools=[tool], priority_tools=["ping"])
    assert cr.priority_tools == ["ping"]


def test_duplicate_tool_names_rejected():
    """Two tools with the same name fail at construction."""
    a = _make_tool("ping")
    b = _make_tool("ping")
    with pytest.raises(CapabilityError):
        AgentCapabilities(tools=[a, b])


# =====================================================================
# Capability-derived tools
# =====================================================================
@pytest.mark.asyncio
async def test_memory_contributes_tools(tmp_path: Path):
    mem = LocalMemoryRegistry(
        user_id="u1",
        base_path=tmp_path,
        tool_mode=MemoryToolMode.FULL,
    )
    cr = AgentCapabilities(memory=mem)
    names = {t.name for t in cr.memory_tools}
    # FULL mode exposes list/update/delete.
    assert {"list_memories", "update_memory", "delete_memory"} <= names


@pytest.mark.asyncio
async def test_knowledge_contributes_tools_per_source(tmp_path: Path):
    """Each knowledge source contributes its own search_{name} tool."""
    kb1 = LocalKnowledgeRegistry(
        name="docs", description="Docs.", base_path=tmp_path
    )
    kb2 = LocalKnowledgeRegistry(
        name="tickets", description="Tickets.", base_path=tmp_path
    )
    cr = AgentCapabilities(knowledge=[kb1, kb2])
    names = {t.name for t in cr.knowledge_tools}
    assert names == {"search_docs", "search_tickets"}


@pytest.mark.asyncio
async def test_routines_contribute_tools(tmp_path: Path):
    """Routine registry exposes search_routines + get_routine in FULL mode."""
    _build_routine_source(tmp_path)
    routines = LocalRoutineRegistry(
        source_path=tmp_path,
        routines=["client_followup", "email_reply"],
        tool_mode=RoutineToolMode.FULL,
    )
    cr = AgentCapabilities(routines=routines)
    names = {t.name for t in cr.routine_tools}
    assert names == {"search_routines", "get_routine"}


@pytest.mark.asyncio
async def test_context_contributes_tools(tmp_path: Path):
    ctx = LocalContextRegistry(
        user_id="u1",
        session_id="s1",
        base_path=tmp_path,
        tool_mode=LogBookToolMode.READ_ONLY,
    )
    cr = AgentCapabilities(context=ctx)
    names = {t.name for t in cr.context_tools}
    assert names == {"search_context"}


# =====================================================================
# Prepare lifecycle
# =====================================================================
@pytest.mark.asyncio
async def test_all_tools_requires_prepare(tmp_path: Path):
    """Accessing all_tools before prepare() raises."""
    cr = AgentCapabilities(tools=[_make_tool("ping")])
    with pytest.raises(CapabilityError):
        _ = cr.all_tools


@pytest.mark.asyncio
async def test_prepare_loads_skills_and_adds_tools(tmp_path: Path):
    """After prepare(), skill tools and read_skill_resource appear in all_tools."""
    _build_skill_source(tmp_path, "demo_skill")

    skills = LocalSkillRegistry(
        name="local_skills",
        source_path=tmp_path,
        skills=["demo_skill"],
    )

    cr = AgentCapabilities(skills=skills)

    # Before prepare(): no skill tools yet.
    assert cr.skill_tools == []
    assert cr._loaded_skills == []

    await cr.prepare()

    # After prepare(): skill function + read_skill_resource present.
    skill_tool_names = {t.name for t in cr.skill_tools}
    assert "do_thing" in skill_tool_names
    assert "read_skill_resource" in skill_tool_names

    # all_tools is now accessible.
    all_names = {t.name for t in cr.all_tools}
    assert "do_thing" in all_names
    assert "read_skill_resource" in all_names

    # Loaded skill is exposed via the property.
    assert len(cr.loaded_skills) == 1
    assert cr.loaded_skills[0].block.name == "demo_skill"


@pytest.mark.asyncio
async def test_prepare_is_idempotent(tmp_path: Path):
    """Calling prepare() twice is a no-op."""
    _build_skill_source(tmp_path, "demo_skill")
    skills = LocalSkillRegistry(
        name="local_skills",
        source_path=tmp_path,
        skills=["demo_skill"],
    )
    cr = AgentCapabilities(skills=skills)

    await cr.prepare()
    first_count = len(cr.loaded_skills)

    await cr.prepare()  # second call must not re-load or duplicate
    assert len(cr.loaded_skills) == first_count


# =====================================================================
# Lookups
# =====================================================================
@pytest.mark.asyncio
async def test_find_and_get_tool(tmp_path: Path):
    tool = _make_tool("ping")
    cr = AgentCapabilities(tools=[tool])
    await cr.prepare()  # required to access all_tools

    assert cr.find_tool("ping") is not None
    assert cr.find_tool("nope") is None

    assert cr.get_tool("ping").name == "ping"
    with pytest.raises(KeyError):
        cr.get_tool("nope")


# =====================================================================
# All capabilities together
# =====================================================================
@pytest.mark.asyncio
async def test_full_stack_aggregates_all_tools(tmp_path: Path):
    """Memory + knowledge + routines + context + skills + explicit tools
    all show up in all_tools after prepare()."""
    # Routines source
    routines_root = tmp_path / "routines_root"
    _build_routine_source(routines_root)

    # Skills source
    skills_root = tmp_path / "skills_root"
    skills_root.mkdir()
    _build_skill_source(skills_root, "demo_skill")

    # Knowledge dir is just inside tmp_path
    kb = LocalKnowledgeRegistry(
        name="docs", description="Docs.", base_path=tmp_path
    )
    mem = LocalMemoryRegistry(user_id="u1", base_path=tmp_path)
    ctx = LocalContextRegistry(
        user_id="u1", session_id="s1", base_path=tmp_path
    )
    routines = LocalRoutineRegistry(
        source_path=routines_root,
        routines=["client_followup", "email_reply"],
    )
    skills = LocalSkillRegistry(
        name="local_skills",
        source_path=skills_root,
        skills=["demo_skill"],
    )

    cr = AgentCapabilities(
        memory=mem,
        knowledge=[kb],
        routines=routines,
        context=ctx,
        skills=skills,
        tools=[_make_tool("ping")],
        priority_tools=["ping"],
    )

    await cr.prepare()

    names = {t.name for t in cr.all_tools}
    # Explicit
    assert "ping" in names
    # Memory (FULL by default)
    assert {"list_memories", "update_memory", "delete_memory"} <= names
    # Knowledge
    assert "search_docs" in names
    # Routines
    assert {"search_routines", "get_routine"} <= names
    # Context
    assert "search_context" in names
    # Skills
    assert "do_thing" in names
    assert "read_skill_resource" in names