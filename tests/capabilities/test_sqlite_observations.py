"""Tests for ObservationRecord persistence in SQLiteContextRegistry."""

from __future__ import annotations

import pytest
from pathlib import Path

from max_ai.base.context import LogBookToolMode
from max_ai.base.observation import ObservationRecord
from max_ai.capabilities.context import SQLiteContextRegistry


# -------- FIXTURES ------------------------------------------------------------
@pytest.fixture
def registry(tmp_path: Path) -> SQLiteContextRegistry:
    return SQLiteContextRegistry(
        user_id="user_001",
        session_id="session_abc",
        base_path=tmp_path,
        tool_mode=LogBookToolMode.READ_WRITE,
    )


@pytest.fixture
def registry_read_only(tmp_path: Path) -> SQLiteContextRegistry:
    return SQLiteContextRegistry(
        user_id="user_001",
        session_id="session_abc",
        base_path=tmp_path,
        tool_mode=LogBookToolMode.READ_ONLY,
    )


# -------- write_observation TESTS ---------------------------------------------
@pytest.mark.asyncio
async def test_write_observation_returns_id(registry):
    """write_observation returns a non-empty id string."""
    await registry.connect()
    obs_id = await registry.write_observation(content="Found competitor A has no public API")
    assert isinstance(obs_id, str)
    assert len(obs_id) > 0
    await registry.disconnect()


@pytest.mark.asyncio
async def test_write_observation_persists(registry):
    """Written observation is retrievable via get_observations."""
    await registry.connect()
    await registry.write_observation(
        content="Strategy X failed with 403 error",
        observation_type="error",
        tags=["pricing", "competitors"],
    )
    records = await registry.get_observations()
    assert len(records) == 1
    assert records[0].content == "Strategy X failed with 403 error"
    assert records[0].observation_type == "error"
    assert "pricing" in records[0].tags
    assert records[0].session_id == "session_abc"
    await registry.disconnect()


@pytest.mark.asyncio
async def test_write_multiple_observations(registry):
    """Multiple observations are stored and returned most-recent-first."""
    await registry.connect()
    await registry.write_observation(content="First finding", observation_type="finding")
    await registry.write_observation(content="Second finding", observation_type="decision")
    await registry.write_observation(content="Third finding", observation_type="hypothesis")

    records = await registry.get_observations()
    assert len(records) == 3
    # most recent first
    assert records[0].content == "Third finding"
    assert records[2].content == "First finding"
    await registry.disconnect()


@pytest.mark.asyncio
async def test_get_observations_limit(registry):
    """get_observations respects the limit parameter."""
    await registry.connect()
    for i in range(5):
        await registry.write_observation(content=f"Observation {i}")

    records = await registry.get_observations(limit=3)
    assert len(records) == 3
    await registry.disconnect()


@pytest.mark.asyncio
async def test_get_observations_filter_by_tags(registry):
    """get_observations filters by tags when provided."""
    await registry.connect()
    await registry.write_observation(content="About pricing", tags=["pricing"])
    await registry.write_observation(content="About competitors", tags=["competitors"])
    await registry.write_observation(content="About both", tags=["pricing", "competitors"])

    pricing_records = await registry.get_observations(tags=["pricing"])
    assert len(pricing_records) == 2
    assert all("pricing" in r.tags for r in pricing_records)
    await registry.disconnect()


@pytest.mark.asyncio
async def test_get_observations_empty(registry):
    """get_observations returns empty list when no observations exist."""
    await registry.connect()
    records = await registry.get_observations()
    assert records == []
    await registry.disconnect()


@pytest.mark.asyncio
async def test_observations_scoped_to_user(tmp_path):
    """Observations from different users are isolated."""
    registry_a = SQLiteContextRegistry(
        user_id="user_a", session_id="s1",
        base_path=tmp_path, tool_mode=LogBookToolMode.READ_WRITE,
    )
    registry_b = SQLiteContextRegistry(
        user_id="user_b", session_id="s1",
        base_path=tmp_path, tool_mode=LogBookToolMode.READ_WRITE,
    )
    await registry_a.connect()
    await registry_b.connect()

    await registry_a.write_observation(content="User A finding")
    await registry_b.write_observation(content="User B finding")

    records_a = await registry_a.get_observations()
    records_b = await registry_b.get_observations()

    assert len(records_a) == 1
    assert records_a[0].content == "User A finding"
    assert len(records_b) == 1
    assert records_b[0].content == "User B finding"

    await registry_a.disconnect()
    await registry_b.disconnect()


# -------- TOOL EXPOSURE TESTS -------------------------------------------------
@pytest.mark.asyncio
async def test_read_write_exposes_two_tools(registry):
    """READ_WRITE mode exposes search_context and write_observation tools."""
    tools = registry.as_tools()
    tool_names = {t.name for t in tools}
    assert "search_context" in tool_names
    assert "write_observation" in tool_names


@pytest.mark.asyncio
async def test_read_only_does_not_expose_write_tool(registry_read_only):
    """READ_ONLY mode does not expose write_observation."""
    tools = registry_read_only.as_tools()
    tool_names = {t.name for t in tools}
    assert "search_context" in tool_names
    assert "write_observation" not in tool_names


@pytest.mark.asyncio
async def test_write_observation_tool_works(registry):
    """The write_observation tool callable actually persists data."""
    await registry.connect()
    tools = registry.as_tools()
    write_tool = next(t for t in tools if t.name == "write_observation")

    from max_ai.types.tool_call import ToolCallRecord
    request = ToolCallRecord(
        id="test_call",
        tool_name="write_observation",
        parameters={"content": "Found via tool call", "observation_type": "finding", "tags": ["test"]},
        session_id="session_abc",
    )
    result = await write_tool.execute(request)
    assert result.success is True

    records = await registry.get_observations()
    assert len(records) == 1
    assert records[0].content == "Found via tool call"
    await registry.disconnect()


@pytest.mark.asyncio
async def test_observation_record_fields(registry):
    """ObservationRecord has all expected fields populated."""
    await registry.connect()
    obs_id = await registry.write_observation(
        content="Test observation",
        observation_type="decision",
        tags=["tag1", "tag2"],
    )
    records = await registry.get_observations()
    r = records[0]

    assert r.id == obs_id
    assert r.session_id == "session_abc"
    assert r.content == "Test observation"
    assert r.observation_type == "decision"
    assert r.tags == ["tag1", "tag2"]
    assert r.created_at is not None
    await registry.disconnect()
