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

from max_ai.base.memory import MemoryToolMode
from max_ai.base.context import LogBookToolMode
from max_ai.base.knowledge import KnowledgeToolMode
from max_ai.base.routines import RoutineToolMode

from max_ai.capabilities.memory import LocalMemoryRegistry
from max_ai.capabilities.context import LocalContextRegistry
from max_ai.capabilities.knowledge import LocalKnowledgeRegistry
from max_ai.capabilities.routines import LocalRoutineRegistry
from max_ai.capabilities.skills.local import LocalSkillRegistry
from max_ai.types.workspace import WorkspaceDirectory
from max_ai.errors.memory import MemoryError


# =====================================================================
# MEMORY
# =====================================================================
@pytest.mark.asyncio
async def test_local_memory_registry_round_trip(tmp_path: Path):
    """update_fact → get_context returns what we wrote, in correct shape."""
    mem = LocalMemoryRegistry(
        user_id="u1",
        base_path=tmp_path,
        tool_mode=MemoryToolMode.FULL,
    )

    async with mem:
        # First read on a new user is empty, not an error.
        initial = await mem.get_context()
        assert initial == []

        # Write two facts.
        await mem.update_fact("user_identity", "Software engineer in Buenos Aires")
        await mem.update_fact("language", "Spanish, prefers technical English")

        # Read back.
        all_facts = await mem.get_context()
        assert len(all_facts) == 2

        by_key = {f.key: f for f in all_facts}
        assert by_key["user_identity"].category == "general"
        assert by_key["user_identity"].content == "Software engineer in Buenos Aires"
        assert by_key["language"].content == "Spanish, prefers technical English"
        assert isinstance(by_key["language"].last_updated, datetime)

        # Update existing key overwrites.
        await mem.update_fact("language", "English only")
        updated = await mem.get_context()
        assert len(updated) == 2  # still two
        by_key = {f.key: f for f in updated}
        assert by_key["language"].content == "English only"

        # Delete removes it.
        await mem.delete_fact("language")
        after_delete = await mem.get_context()
        assert len(after_delete) == 1
        assert after_delete[0].key == "user_identity"

        # Delete missing key is silent no-op.
        await mem.delete_fact("never_existed")  # must not raise

    # File should exist on disk after writes.
    user_file = tmp_path / "memory" / "u1.json"
    assert user_file.is_file()


@pytest.mark.asyncio
async def test_local_memory_rejects_invalid_keys(tmp_path: Path):
    mem = LocalMemoryRegistry(user_id="u1", base_path=tmp_path)
    async with mem:
        with pytest.raises(MemoryError):
            await mem.update_fact("", "value")
        with pytest.raises(MemoryError):
            await mem.update_fact("   ", "value")


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
            "score": None,
            "tokens": 12,
            "metadata": {"source": "fastapi.md"},
        },
        {
            "content": "Pydantic provides data validation using type annotations.",
            "score": None,
            "tokens": 10,
            "metadata": {"source": "pydantic.md"},
        },
        {
            "content": "Cooking pasta requires boiling salted water.",
            "score": None,
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
        # All returned blocks must have a fresh score, not None.
        for block in results:
            assert block.score is not None
            assert block.score > 0
        # Top result should be the FastAPI block (closest semantically).
        assert "FastAPI" in results[0].content
        # The unrelated cooking block must rank below the on-topic blocks.
        cooking = next((b for b in results if "pasta" in b.content), None)
        if cooking is not None:
            assert cooking.score < results[0].score


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
# ROUTINE
# =====================================================================
@pytest.mark.asyncio
async def test_local_routine_registry_only_authorized(tmp_path: Path):
    """Repo has 3 routines on disk; only 2 are authorized."""
    routines_dir = tmp_path / "routines"
    routines_dir.mkdir(parents=True)

    def write_routine(name: str, description: str, instructions: str):
        (routines_dir / f"{name}.json").write_text(json.dumps({
            "name": name,
            "description": description,
            "instructions": instructions,
        }))

    write_routine("client_followup", "Follow up with a client.", "1. ... 2. ...")
    write_routine("email_reply", "Draft an email reply.", "Steps...")
    write_routine("internal_secret", "Internal procedure.", "Don't expose.")

    reg = LocalRoutineRegistry(
        source_path=tmp_path,
        routines=["client_followup", "email_reply"],  # only 2 of 3
        tool_mode=RoutineToolMode.FULL,
    )

    async with reg:
        # Catalog returns the 2 authorized, ignores the 3rd on disk.
        catalog = await reg.get_catalog()
        assert {r.name for r in catalog} == {"client_followup", "email_reply"}

        # Search ranks the email routine on top, and never leaks the
        # unauthorized routine regardless of semantic score.
        results = await reg.search("email")
        assert results
        assert results[0].name == "email_reply"
        assert "internal_secret" not in {r.name for r in results}

        # Fetch authorized routine succeeds.
        block = await reg.fetch("client_followup")
        assert block.name == "client_followup"
        assert "1." in block.instructions

        # Fetch unauthorized routine fails with informative message.
        with pytest.raises(ValueError) as exc_info:
            await reg.fetch("internal_secret")
        msg = str(exc_info.value)
        assert "internal_secret" in msg
        assert "client_followup" in msg  # available list mentioned
        assert "email_reply" in msg


@pytest.mark.asyncio
async def test_local_routine_registry_missing_authorized(tmp_path: Path):
    """Authorized routine that doesn't exist on disk → fails loud at connect."""
    (tmp_path / "routines").mkdir(parents=True)
    # Don't create any files — just an empty routines dir.

    reg = LocalRoutineRegistry(
        source_path=tmp_path,
        routines=["does_not_exist"],
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        async with reg:
            pass
    assert "does_not_exist" in str(exc_info.value)


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

        directory = WorkspaceDirectory(
            root=server_root.resolve() / "tmp" / "u1",
            tool_dir=server_root.resolve() / "tmp" / "u1" / "tools",
            skill_dir=server_root.resolve() / "tmp" / "u1" / "skills",
            artifacts_dir=server_root.resolve() / "tmp" / "u1" / "artifacts",
        )
        session_skills = reg.materialize(directory)
        assert session_skills == directory.skill_dir
        assert (session_skills / "demo_skill" / "SKILL.md").is_file()
        assert (session_skills / "demo_skill" / "references" / "notes.md").is_file()
