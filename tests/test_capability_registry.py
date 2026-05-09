"""Smoke tests for AgentCapabilities."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from max_ai.capabilities.context import LocalContextRegistry
from max_ai.capabilities.knowledge import LocalKnowledgeRegistry
from max_ai.capabilities.memory import LocalMemoryRegistry
from max_ai.capabilities.routines import LocalRoutineRegistry
from max_ai.capabilities.skills import LocalSkillRegistry
from max_ai.errors.capabilities import CapabilityError
from max_ai.manager import AgentCapabilities
from max_ai.tools import FunctionAsTool


def _make_tool(name: str = "ping") -> FunctionAsTool:
    async def fn(message: str) -> str:
        """Echo the message back.

        Args:
            message: Anything.
        """
        return message

    fn.__name__ = name
    return FunctionAsTool(fn)


def _build_routine_source(tmp_path: Path) -> Path:
    routines_dir = tmp_path / "routines"
    routines_dir.mkdir(parents=True)
    for name in ("client_followup", "email_reply"):
        (routines_dir / f"{name}.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "description": f"{name} routine.",
                    "instructions": "1. Do X. 2. Do Y.",
                }
            )
        )
    return tmp_path


def _build_skill_source(tmp_path: Path, skill_name: str = "demo_skill") -> Path:
    skill_dir = tmp_path / skill_name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {skill_name}
            description: A minimal skill for testing.
            ---

            Use this skill when testing.
            """
        )
    )
    return tmp_path


def test_empty_registry_constructs():
    cr = AgentCapabilities()
    assert cr.has_memory is False
    assert cr.has_knowledge is False
    assert cr.has_routines is False
    assert cr.has_skills is False
    assert cr.has_logbook is False
    assert cr.has_tools is False
    assert {tool.name for tool in cr._all_tools_sync()} == {"workspace"}


def test_explicit_tools_normalized():
    tool = _make_tool("ping")

    async def raw_callable(x: str) -> str:
        """Echo.

        Args:
            x: Anything.
        """
        return x

    raw_callable.__name__ = "echo"

    cr = AgentCapabilities(toolset=[tool, raw_callable])
    assert {t.name for t in cr.toolset} == {"ping", "echo"}


def test_invalid_tool_entry_rejected():
    with pytest.raises(CapabilityError):
        AgentCapabilities(toolset=[42])  # type: ignore[list-item]


def test_priority_tools_must_reference_real_tool():
    with pytest.raises(CapabilityError):
        AgentCapabilities(
            toolset=[_make_tool("ping")],
            priority_tools=["does_not_exist"],
        )


def test_priority_tools_accepted_when_present():
    cr = AgentCapabilities(toolset=[_make_tool("ping")], priority_tools=["ping"])
    assert cr.priority_tools == ["ping"]


def test_duplicate_tool_names_rejected():
    with pytest.raises(CapabilityError):
        AgentCapabilities(toolset=[_make_tool("ping"), _make_tool("ping")])


def test_capability_sources_contribute_tools(tmp_path: Path):
    _build_routine_source(tmp_path)
    memory = LocalMemoryRegistry(user_id="u1", base_path=tmp_path)
    logbook = LocalContextRegistry(user_id="u1", session_id="s1", base_path=tmp_path)
    knowledge = LocalKnowledgeRegistry(name="docs", description="Docs.", base_path=tmp_path)
    routines = LocalRoutineRegistry(
        source_path=tmp_path,
        routines=["client_followup", "email_reply"],
    )

    cr = AgentCapabilities(
        memory=memory,
        logbook=logbook,
        knowledge=[knowledge],
        routines=routines,
    )
    names = {tool.name for tool in cr._all_tools_sync()}

    assert "workspace" in names
    assert {"list_memories", "update_memory", "delete_memory"} <= names
    assert "search_context" in names
    assert "search_docs" in names
    assert {"search_routines", "get_routine"} <= names


@pytest.mark.asyncio
async def test_all_tools_requires_prepare():
    cr = AgentCapabilities(toolset=[_make_tool("ping")])
    with pytest.raises(CapabilityError):
        _ = cr.all_tools


@pytest.mark.asyncio
async def test_prepare_loads_skills_and_adds_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SKILLS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SERVER_DIR", str(tmp_path / "server"))
    _build_skill_source(tmp_path, "demo_skill")

    skills = LocalSkillRegistry(source=tmp_path, skills=["demo_skill"])
    cr = AgentCapabilities(skills=skills)

    assert cr.loaded_skill_blocks == []
    await cr.prepare()

    assert {tool.name for tool in cr.skill_tools} == {"search_skills", "skill_bash"}
    assert {"workspace", "search_skills", "skill_bash"} <= {tool.name for tool in cr.all_tools}
    assert len(cr.loaded_skill_blocks) == 1
    assert cr.loaded_skill_blocks[0].name == "demo_skill"


@pytest.mark.asyncio
async def test_prepare_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SKILLS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SERVER_DIR", str(tmp_path / "server"))
    _build_skill_source(tmp_path, "demo_skill")
    cr = AgentCapabilities(skills=LocalSkillRegistry(source=tmp_path, skills=["demo_skill"]))

    await cr.prepare()
    first_count = len(cr.loaded_skill_blocks)
    await cr.prepare()
    assert len(cr.loaded_skill_blocks) == first_count


@pytest.mark.asyncio
async def test_find_and_get_tool():
    cr = AgentCapabilities(toolset=[_make_tool("ping")])
    await cr.prepare()

    assert cr.find_tool("ping") is not None
    assert cr.find_tool("workspace") is not None
    assert cr.find_tool("nope") is None
    assert cr.get_tool("ping").name == "ping"
    with pytest.raises(KeyError):
        cr.get_tool("nope")


@pytest.mark.asyncio
async def test_full_stack_aggregates_all_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SKILLS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SERVER_DIR", str(tmp_path / "server"))
    routines_root = tmp_path / "routines_root"
    _build_routine_source(routines_root)
    skills_root = tmp_path / "skills_root"
    skills_root.mkdir()
    _build_skill_source(skills_root, "demo_skill")

    cr = AgentCapabilities(
        memory=LocalMemoryRegistry(user_id="u1", base_path=tmp_path),
        knowledge=[LocalKnowledgeRegistry(name="docs", description="Docs.", base_path=tmp_path)],
        routines=LocalRoutineRegistry(
            source_path=routines_root,
            routines=["client_followup", "email_reply"],
        ),
        logbook=LocalContextRegistry(user_id="u1", session_id="s1", base_path=tmp_path),
        skills=LocalSkillRegistry(source=skills_root, skills=["demo_skill"]),
        toolset=[_make_tool("ping")],
        priority_tools=["ping"],
    )

    await cr.prepare()
    names = {tool.name for tool in cr.all_tools}

    assert "workspace" in names
    assert "ping" in names
    assert {"list_memories", "update_memory", "delete_memory"} <= names
    assert "search_docs" in names
    assert {"search_routines", "get_routine"} <= names
    assert "search_context" in names
    assert {"search_skills", "skill_bash"} <= names
