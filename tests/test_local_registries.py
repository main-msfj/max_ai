"""
Smoke tests for every filesystem-local registry.

One test per registry. Each test:
  1. Builds a temporary directory layout for the registry.
  2. Constructs the registry pointing at that directory.
  3. Exercises its primary operations.
  4. Asserts the results match what we put on disk.

These are integration tests against the real filesystem (using
pytest's tmp_path fixture) — no mocks. If a registry's contract
drifts, the test fails immediately.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path

import pytest

from max_ai.base.context import LogBookToolMode
from max_ai.base.knowledge import KnowledgeToolMode
from max_ai.base.memory import MemoryToolMode
from max_ai.capabilities.context import LocalContextRegistry
from max_ai.capabilities.knowledge import LocalKnowledgeRegistry
from max_ai.capabilities.memory import LocalMemoryRegistry
from max_ai.capabilities.skills.local import LocalSkillRegistry
from max_ai.errors.memory import MemoryError
from max_ai.types.workspace import WorkspaceDirectory


# =====================================================================
# MEMORY
# =====================================================================
@pytest.mark.asyncio
async def test_local_memory_registry_round_trip(tmp_path: Path):
    """create_or_update → get_context returns what we wrote, per category."""
    mem = LocalMemoryRegistry(base_path=tmp_path, tool_mode=MemoryToolMode.FULL).bind("u1", "s1")

    async with mem:
        assert await mem.get_context() == []  # a new user starts empty

        await mem.create_or_update("identity", "Software engineer in Buenos Aires")
        await mem.create_or_update("language", "Spanish, prefers technical English")
        by_category = {r.category: r for r in await mem.get_context()}
        assert by_category["identity"].memory == "Software engineer in Buenos Aires"
        assert isinstance(by_category["language"].updated, datetime)

        # Updating a category replaces it.
        await mem.create_or_update("language", "English only")
        by_category = {r.category: r for r in await mem.get_context()}
        assert len(by_category) == 2 and by_category["language"].memory == "English only"

        assert await mem.delete_memory("language") == "Memory deleted: language"
        assert [r.category for r in await mem.get_context()] == ["identity"]
        assert "not found" in await mem.delete_memory("never_existed")

    assert (tmp_path / "memory" / "u1" / "s1.json").is_file()


@pytest.mark.asyncio
async def test_local_memory_rejects_empty_categories(tmp_path: Path):
    mem = LocalMemoryRegistry(base_path=tmp_path).bind("u1", "s1")
    async with mem:
        for category in ("", "   "):
            with pytest.raises(MemoryError):
                await mem.create_or_update(category, "value")


# =====================================================================
# CONTEXT
# =====================================================================
@pytest.mark.asyncio
async def test_local_context_registry_summary_and_search(tmp_path: Path):
    """Pre-populated file: summary for current session, search across all."""
    # Build the user file by hand — registry is read-only.
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True)
    user_file = context_dir / "u1.json"
    user_file.write_text(json.dumps({
        "session_001": {
            "summary": "Discussed spaceship designs and rocket fuel options.",
            "vector": [],
            "timestamp": "2026-04-20T15:00:00Z",
            "metadata": {"topic": "engineering"},
        },
        "session_002": {
            "summary": "Conversation about bicycle frame materials.",
            "vector": [],
            "timestamp": "2026-04-22T10:00:00Z",
            "metadata": {"topic": "cycling"},
        },
    }))

    ctx = LocalContextRegistry(
        user_id="u1",
        session_id="session_002",  # current session is 002
        base_path=tmp_path,
        tool_mode=LogBookToolMode.READ_ONLY,
    )

    async with ctx:
        # Current session summary.
        summary = await ctx.get_current_session_summary()
        assert summary == "Conversation about bicycle frame materials."

        # Semantic search ranks the rocket/spaceship session on top.
        results = await ctx.search("rocket spaceship")
        assert results
        assert results[0].session_id == "session_001"
        assert "spaceship" in results[0].content
        assert results[0].score is not None
        assert results[0].score > 0
        # The on-topic match must clearly outscore the off-topic one.
        scores = {r.session_id: r.score for r in results}
        if "session_002" in scores:
            assert scores["session_001"] > scores["session_002"]


@pytest.mark.asyncio
async def test_local_context_registry_missing_file(tmp_path: Path):
    """No file on disk yet → summary is None, search is []."""
    ctx = LocalContextRegistry(
        user_id="u_unknown",
        session_id="session_x",
        base_path=tmp_path,
    )
    async with ctx:
        assert await ctx.get_current_session_summary() is None
        assert await ctx.search("anything") == []


# =====================================================================
# KNOWLEDGE
# =====================================================================
@pytest.mark.asyncio
async def test_local_knowledge_registry_search(tmp_path: Path):
    """Pre-populated source file: search returns ranked blocks."""
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True)
    source_file = knowledge_dir / "docs.json"
    source_file.write_text(json.dumps([
        {
            "content": "FastAPI is a modern Python web framework for APIs.",
            "tokens": 12,
            "metadata": {"source": "fastapi.md"},
        },
        {
            "content": "Pydantic provides data validation using type annotations.",
            "tokens": 10,
            "metadata": {"source": "pydantic.md"},
        },
        {
            "content": "Cooking pasta requires boiling salted water.",
            "tokens": 7,
            "metadata": {"source": "irrelevant.md"},
        },
    ]))

    kb = LocalKnowledgeRegistry(
        name="docs",
        description="Internal documentation.",
        base_path=tmp_path,
        tool_mode=KnowledgeToolMode.FULL,
    )

    async with kb:
        # Query that matches the python framework blocks.
        results = await kb.search("Python framework", limit=5)
        assert len(results) >= 1
        # Top result should be the FastAPI block (closest semantically).
        # Ranking happens internally (cosine similarity); the score itself
        # is never attached to the returned blocks — see KnowledgeBlock.
        assert "FastAPI" in results[0].content
        # The unrelated cooking block, if it clears the relevance bar at
        # all, must rank below the on-topic blocks.
        contents = [b.content for b in results]
        if any("pasta" in c for c in contents):
            assert contents.index(next(c for c in contents if "pasta" in c)) > 0


@pytest.mark.asyncio
async def test_local_knowledge_registry_missing_file(tmp_path: Path):
    """Source file doesn't exist → search returns []."""
    kb = LocalKnowledgeRegistry(
        name="empty_source",
        description="Doesn't exist.",
        base_path=tmp_path,
    )
    async with kb:
        assert await kb.search("anything") == []


# =====================================================================
# SKILL
# =====================================================================
@pytest.mark.asyncio
async def test_local_skill_registry_loads_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a minimal skill on disk and expose it via the new registry flow."""
    source_root = tmp_path / "source"
    cache_root = tmp_path / "cache"
    server_root = tmp_path / "server"
    monkeypatch.setenv("SKILLS_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("SERVER_DIR", str(server_root))

    skill_dir = source_root / "demo_skill"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "references").mkdir(parents=True)

    # SKILL.md
    skill_md = textwrap.dedent("""\
        ---
        name: demo_skill
        description: A minimal skill for testing.
        resources:
          notes.md: Some reference notes.
        ---

        # Demo Skill

        When asked to demo, use the `say_hello` tool.
        """)
    (skill_dir / "SKILL.md").write_text(skill_md)

    # A simple script with one public function.
    script = textwrap.dedent("""\
        def say_hello(name: str) -> str:
            \"\"\"Return a greeting.

            Args:
                name: Who to greet.
            \"\"\"
            return f"Hello, {name}!"
        """)
    (skill_dir / "scripts" / "greetings.py").write_text(script)

    # The reference file declared in frontmatter.
    (skill_dir / "references" / "notes.md").write_text("# Demo notes\n\nNothing important.")

    # Construct registry pointing at the parent dir.
    reg = LocalSkillRegistry(
        source=source_root,
        skills=["demo_skill"],
    )

    async with reg:
        blocks = await reg.get_skills()
        assert len(blocks) == 1
        block = blocks[0]

        # Prompt block is intentionally lightweight.
        assert block.name == "demo_skill"
        assert block.description == "A minimal skill for testing."

        # Skill package was cached and can be materialized for a user.
        cached_skill = reg._registry_cache_dir / "demo_skill"
        assert (cached_skill / "SKILL.md").is_file()
        assert (cached_skill / "references" / "notes.md").is_file()
        assert (cached_skill / "scripts" / "greetings.py").is_file()

        user_root = server_root.resolve() / "tmp" / "u1"
        directory = WorkspaceDirectory(
            root=user_root,
            workspace_dir=user_root / "workspace",
            skill_dir=user_root / "skills",
            artifacts_dir=user_root / "workspace",
        )
        session_skills = reg.materialize(directory)
        assert session_skills == directory.skill_dir
        assert (session_skills / "demo_skill" / "SKILL.md").is_file()
        assert (session_skills / "demo_skill" / "references" / "notes.md").is_file()
