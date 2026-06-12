from __future__ import annotations

import typing as t

import pytest
from mcp.types import (
    CallToolResult,
    ListResourceTemplatesResult,
    ListResourcesResult,
    ListToolsResult,
    Resource,
    ResourceTemplate,
    TextContent,
    TextResourceContents,
    Tool,
    ToolAnnotations,
)
from pydantic import AnyUrl

from max_ai.base.tools import CoreTool
from max_ai.mcp import (
    HTTPServerConfig,
    MCPClientManager,
    MCPResourceTool,
    MCPTool,
    StdioMCPServerConfig,
    create_mcp_tools,
)
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode


class FakeManager:
    def __init__(self, tool_result: t.Any = None, resource_result: t.Any = None) -> None:
        self.tool_result = tool_result
        self.resource_result = resource_result
        self.calls: list[tuple[str, str, dict[str, t.Any], float]] = []
        self.resource_calls: list[tuple[str, t.Any]] = []

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, t.Any],
        timeout_seconds: float,
    ) -> t.Any:
        self.calls.append((server_id, tool_name, arguments, timeout_seconds))
        return self.tool_result

    async def read_resource(self, server_id: str, uri: t.Any) -> t.Any:
        self.resource_calls.append((server_id, uri))
        return self.resource_result


@pytest.mark.asyncio
async def test_mcp_tool_executes_remote_tool_and_returns_core_result() -> None:
    manager = FakeManager(
        tool_result=CallToolResult(
            content=[TextContent(type="text", text="hello from mcp")],
            isError=False,
        )
    )
    tool = MCPTool(
        mcp_tool_name="search",
        mcp_tool_description="Search things",
        mcp_tool_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
            "additionalProperties": False,
        },
        client_manager=t.cast(t.Any, manager),
        server_id="docs",
        approval_mode=ToolApprovalMode.AUTO_APPROVED,
        timeout_seconds=12,
    )

    result = await tool.execute(
        ToolCallRecord(tool_name=tool.name, parameters={"q": "abc"})
    )

    assert result.success is True
    assert result.result == "hello from mcp"
    assert result.metadata["server_id"] == "docs"
    assert result.metadata["mcp_tool_name"] == "search"
    assert manager.calls == [("docs", "search", {"q": "abc"}, 12)]


def test_http_server_config_adds_bearer_token_header() -> None:
    config = HTTPServerConfig(
        server_id="github",
        url="http://localhost:3000/mcp",
        token="secret",
    )

    assert config.headers["Authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_mcp_tool_returns_invalid_parameters_for_schema_errors() -> None:
    manager = FakeManager()
    tool = MCPTool(
        mcp_tool_name="search",
        mcp_tool_description="Search things",
        mcp_tool_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
            "additionalProperties": False,
        },
        client_manager=t.cast(t.Any, manager),
        server_id="docs",
    )

    result = await tool.execute(ToolCallRecord(tool_name=tool.name, parameters={}))

    assert result.success is False
    assert "Invalid parameters" in (result.error or "")
    assert manager.calls == []


@pytest.mark.asyncio
async def test_mcp_resource_tool_reads_text_resource() -> None:
    manager = FakeManager(
        resource_result=t.cast(
            t.Any,
            type(
                "ReadResult",
                (),
                {
                    "contents": [
                        TextResourceContents(
                            uri=AnyUrl("file:///tmp/readme.md"),
                            text="resource body",
                        )
                    ]
                },
            )(),
        )
    )
    tool = MCPResourceTool(
        client_manager=t.cast(t.Any, manager),
        server_id="docs",
        available_resources=[
            Resource(name="readme", uri=AnyUrl("file:///tmp/readme.md"))
        ],
    )

    result = await tool.execute(
        ToolCallRecord(
            tool_name=tool.name,
            parameters={"uri": "file:///tmp/readme.md"},
        )
    )

    assert result.success is True
    assert result.result == "resource body"
    assert result.metadata["resource_uri"] == "file:///tmp/readme.md"
    assert manager.resource_calls[0][0] == "docs"


@pytest.mark.asyncio
async def test_create_mcp_tools_can_register_without_connecting() -> None:
    manager, tools = await create_mcp_tools(
        [StdioMCPServerConfig(server_id="fs", command="npx", args=["server"])],
        auto_connect=False,
    )

    assert isinstance(manager, MCPClientManager)
    assert manager.server_ids == ["fs"]
    assert tools == []


@pytest.mark.asyncio
async def test_manager_connect_discovers_tools_and_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    import max_ai.mcp.client_manager as client_manager_module

    class FakeTransport:
        read = object()
        write = object()
        closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeClientSession:
        def __init__(self, read: object, write: object) -> None:
            self.read = read
            self.write = write
            self.exited = False

        async def __aenter__(self) -> "FakeClientSession":
            return self

        async def __aexit__(self, *args: object) -> None:
            self.exited = True

        async def initialize(self) -> object:
            return object()

        async def list_tools(self, cursor: str | None = None) -> ListToolsResult:
            return ListToolsResult(
                tools=[
                    Tool(
                        name="search",
                        description="Search docs",
                        inputSchema={
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                        annotations=ToolAnnotations(readOnlyHint=True),
                    ),
                    Tool(
                        name="delete_doc",
                        description="Delete docs",
                        inputSchema={
                            "type": "object",
                            "properties": {"id": {"type": "string"}},
                        },
                        annotations=ToolAnnotations(destructiveHint=True),
                    ),
                    Tool(
                        name="forced_safe",
                        description="Override me",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                ]
            )

        async def list_resources(self, cursor: str | None = None) -> ListResourcesResult:
            return ListResourcesResult(
                resources=[Resource(name="readme", uri=AnyUrl("file:///readme.md"))]
            )

        async def list_resource_templates(
            self,
            cursor: str | None = None,
        ) -> ListResourceTemplatesResult:
            return ListResourceTemplatesResult(
                resourceTemplates=[
                    ResourceTemplate(name="doc", uriTemplate="docs://{name}")
                ]
            )

    async def fake_connect(config: object) -> FakeTransport:
        return FakeTransport()

    monkeypatch.setattr(client_manager_module, "ClientSession", FakeClientSession)
    monkeypatch.setattr(client_manager_module, "connect_to_mcp_server", fake_connect)

    manager = MCPClientManager()
    manager.add_server(
        HTTPServerConfig(
            server_id="docs",
            url="http://localhost:3000/mcp",
            tool_approval_modes={"forced_safe": ToolApprovalMode.AUTO_APPROVED},
        )
    )

    await manager.connect("docs")
    tools = manager.get_tools()

    assert len(tools) == 4
    assert all(isinstance(tool, CoreTool) for tool in tools)
    assert tools[0].name == "docs_search"
    assert tools[0].approval_mode == ToolApprovalMode.AUTO_APPROVED
    assert tools[1].name == "docs_delete_doc"
    assert tools[1].approval_mode == ToolApprovalMode.ASK_APPROVED
    assert tools[2].name == "docs_forced_safe"
    assert tools[2].approval_mode == ToolApprovalMode.AUTO_APPROVED
    assert tools[3].name == "docs_read_resource"
