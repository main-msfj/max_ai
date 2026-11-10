"""A tool must show the model a snake_case name and a description of 5 to 150
tokens; MCP tools are not checked, their servers write them."""

import pytest

from max_ai.capabilities.mcp.tool import MCPTool
from max_ai.capabilities.tools.ask_user import AskUserTool
from max_ai.capabilities.tools.bash import BashTool
from max_ai.capabilities.tools.decorator import tool
from max_ai.capabilities.tools.file_system import FileSystemTools
from max_ai.capabilities.tools.plan import AgentUpdatePlanTool
from max_ai.capabilities.tools.search_tool import SearchToolsTool
from max_ai.core.tool.registry import ToolRegistry


def test_every_max_ai_tool_builds(tmp_path):
    AskUserTool()
    AgentUpdatePlanTool()
    BashTool()
    SearchToolsTool(ToolRegistry())
    FileSystemTools(tmp_path).tools


def test_a_name_that_is_not_snake_case_is_refused():
    def Add(a: int, b: int) -> int:
        """Add two numbers together."""

    with pytest.raises(ValueError, match="snake_case"):
        tool(Add)


def test_a_description_too_short_or_too_long_is_refused():
    def add(a: int, b: int) -> int:
        """Add."""

    with pytest.raises(ValueError, match="at least 5"):
        tool(add)
    with pytest.raises(ValueError, match="at most 150; move how to use it"):
        tool(add, description="Add two numbers. " * 60)


def test_mcp_tools_are_not_checked():
    built = MCPTool("Search-Web", "Web.", {"type": "object"}, client_manager=None, server_id="exa")
    assert built.description == "Web."
