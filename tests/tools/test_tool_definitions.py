"""A tool checks itself when it is built: a snake_case name and a description.
An MCP tool keeps the name its server gave it."""

import pytest

from max_ai.capabilities.mcp.tool import MCPTool
from max_ai.capabilities.tools.ask_user import AskUserTool
from max_ai.capabilities.tools.bash import BashTool
from max_ai.capabilities.tools.call_tool import CallToolTool
from max_ai.capabilities.tools.decorator import tool
from max_ai.capabilities.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
from max_ai.capabilities.tools.search_tool import SearchToolsTool
from max_ai.capabilities.tools.skill import ReadSkillTool
from max_ai.capabilities.tools.todos import WriteTodosTool
from max_ai.core.tool.registry import ToolRegistry
from max_ai.errors.tools import ToolDefinitionError
from max_ai.types.tools import ToolKind


def test_every_max_ai_tool_builds():
    for made in (AskUserTool, BashTool, ReadFileTool, WriteFileTool, EditFileTool, ReadSkillTool, WriteTodosTool):
        made()
    SearchToolsTool(ToolRegistry())
    CallToolTool(ToolRegistry())


def test_a_name_that_is_not_snake_case_is_refused():
    def Add(a: int, b: int) -> int:
        """Add two numbers together."""

    with pytest.raises(ValueError, match="snake_case"):
        tool(Add)


def test_a_tool_without_a_description_is_refused():
    def add(a: int, b: int) -> int:
        return a + b

    with pytest.raises(ToolDefinitionError, match="no description"):
        tool(add, description="   ")


def test_a_tool_is_checked_when_it_is_built_not_when_an_agent_starts():
    def Add(a: int, b: int) -> int:
        """Add two numbers together."""

    # No agent, registry or setting involved: building the tool is enough.
    with pytest.raises(ToolDefinitionError):
        tool(Add)
    assert tool(Add, name="add").kind is ToolKind.FUNCTION


def test_an_mcp_tool_keeps_the_name_its_server_gave_it():
    built = MCPTool("Search-Web", "Web.", {"type": "object"}, client_manager=None, server_id="exa")
    assert built.description == "Web." and built.kind is ToolKind.MCP
    with pytest.raises(ToolDefinitionError, match="no description"):
        MCPTool("Search-Web", "", {"type": "object"}, client_manager=None, server_id="exa")
