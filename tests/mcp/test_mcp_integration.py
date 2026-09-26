from __future__ import annotations

import typing as t

import pytest
from mcp.types import (
    CallToolResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    Resource,
    ResourceTemplate,
    TextContent,
    TextResourceContents,
    Tool,
    ToolAnnotations,
)

from max_ai.base.tools import CoreTool
from max_ai.capabilities.mcp import (
    HTTPServerConfig,
    MCPClientManager,
    MCPResourceTool,
    MCPTool,
    StdioMCPServerConfig,
    create_mcp_tools,
    deserialize_mcp_servers,
    serialize_mcp_servers,
)
from max_ai.errors.mcp import MCPServerConfigError
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

    assert config.request_headers["Authorization"] == "Bearer secret"


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
                            uri="file:///tmp/readme.md",
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
            Resource(name="readme", uri="file:///tmp/readme.md")
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
    import max_ai.capabilities.mcp.client_manager as client_manager_module

    class FakeClient:
        def __init__(self) -> None:
            self.closed = False
            self.calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.closed = True

        async def list_tools(self, *, cursor=None):
            return ListToolsResult(tools=[
                Tool(name="search", description="Search docs",
                     input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
                     annotations=ToolAnnotations(read_only_hint=True)),
                Tool(name="delete_doc", description="Delete docs",
                     input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
                     annotations=ToolAnnotations(destructive_hint=True)),
                Tool(name="forced_safe", description="Override me",
                     input_schema={"type": "object", "properties": {}}),
            ])

        async def list_resources(self, *, cursor=None):
            return ListResourcesResult(resources=[Resource(name="readme", uri="file:///readme.md")])

        async def list_resource_templates(self, *, cursor=None):
            return ListResourceTemplatesResult(resource_templates=[
                ResourceTemplate(name="doc", uri_template="docs://{name}")
            ])

        async def call_tool(self, name, arguments, read_timeout_seconds=None):
            self.calls.append((name, arguments, read_timeout_seconds))
            return CallToolResult(content=[TextContent(type="text", text="done")])

    client = FakeClient()
    monkeypatch.setattr(client_manager_module, "create_mcp_client", lambda config: client)
    manager = MCPClientManager()
    manager.add_server(HTTPServerConfig(
        server_id="docs", url="http://localhost:3000/mcp",
        tool_approval_modes={"forced_safe": ToolApprovalMode.AUTO_APPROVED},
    ))

    await manager.connect("docs")
    tools = manager.get_tools()
    assert len(tools) == 4
    assert all(isinstance(tool, CoreTool) for tool in tools)
    assert [tool.name for tool in tools] == [
        "docs_search", "docs_delete_doc", "docs_forced_safe", "docs_read_resource"
    ]
    assert tools[0].approval_mode == ToolApprovalMode.AUTO_APPROVED
    assert tools[1].approval_mode == ToolApprovalMode.ASK_APPROVED
    assert tools[2].approval_mode == ToolApprovalMode.AUTO_APPROVED
    assert await manager.call_tool("docs", "search", {"q": "x"}, 12) == CallToolResult(
        content=[TextContent(type="text", text="done")]
    )
    assert client.calls == [("search", {"q": "x"}, 12)]
    await manager.disconnect_all()
    assert client.closed


def test_server_configs_round_trip_as_json(monkeypatch) -> None:
    monkeypatch.setenv("DOCS_TOKEN", "secret")
    monkeypatch.setenv("DOCS_KEY", "key-123")
    monkeypatch.setenv("CRM_KEY", "crm-456")
    stdio = StdioMCPServerConfig(
        server_id="local", command="python", args=["-m", "server"],
        env={"MODE": "test"}, env_from={"API_KEY": "CRM_KEY"},
        tool_approval_modes={"search": "auto_approval"},
    )
    http = HTTPServerConfig(
        server_id="remote", url="https://example.com/mcp", token_env="DOCS_TOKEN",
        headers={"X-Test": "yes"}, headers_env={"X-Api-Key": "DOCS_KEY"},
    )
    assert StdioMCPServerConfig.model_validate_json(stdio.model_dump_json()) == stdio
    assert HTTPServerConfig.model_validate_json(http.model_dump_json()) == http
    payload = serialize_mcp_servers([stdio, http])
    assert deserialize_mcp_servers(payload) == [stdio, http]
    # Stored config names the env vars; secrets appear only once resolved.
    assert not any(secret in payload for secret in ("secret", "key-123", "crm-456"))
    assert http.request_headers == {
        "X-Test": "yes", "X-Api-Key": "key-123", "Authorization": "Bearer secret",
    }
    assert stdio.process_env == {"MODE": "test", "API_KEY": "crm-456"}


def test_a_literal_token_works_in_code_but_is_never_stored() -> None:
    http = HTTPServerConfig(server_id="remote", url="https://example.com/mcp", token="secret")
    assert http.request_headers["Authorization"] == "Bearer secret"
    assert "secret" not in http.model_dump_json() and "secret" not in repr(http)
    with pytest.raises(MCPServerConfigError, match="use token_env"):
        serialize_mcp_servers([http])


def test_a_missing_env_var_fails_when_connecting(monkeypatch) -> None:
    monkeypatch.delenv("NOPE_TOKEN", raising=False)
    http = HTTPServerConfig(server_id="remote", url="https://example.com/mcp", token_env="NOPE_TOKEN")
    with pytest.raises(MCPServerConfigError, match="needs env var NOPE_TOKEN"):
        http.request_headers

async def test_a_cancellation_inside_the_client_is_a_connection_error(monkeypatch) -> None:
    import asyncio

    from max_ai.capabilities.mcp import client_manager as client_manager_module
    from max_ai.capabilities.mcp.client_manager import MCPClientManager

    class DroppingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def list_tools(self, *, cursor=None):
            return ListToolsResult(tools=[])

        async def list_resources(self, *, cursor=None):
            return ListResourcesResult(resources=[])

        async def list_resource_templates(self, *, cursor=None):
            return ListResourceTemplatesResult(resource_templates=[])

        async def call_tool(self, name, arguments, read_timeout_seconds=None):
            raise asyncio.CancelledError  # what the transport does when the stream breaks

    monkeypatch.setattr(client_manager_module, "create_mcp_client", lambda config: DroppingClient())
    manager = MCPClientManager()
    manager.add_server(HTTPServerConfig(server_id="web", url="http://localhost:3000/mcp"))
    await manager.connect("web")
    with pytest.raises(RuntimeError, match="interrupted the request"):
        await manager.call_tool("web", "search", {}, 5)

    await manager.disconnect_all()
