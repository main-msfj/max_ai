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
from pydantic import ValidationError

from max_ai.base.tools import CoreTool
from max_ai.capabilities.mcp import (
    HTTPSecret,
    HTTPServerConfig,
    MCPClientManager,
    MCPResourceTool,
    MCPTool,
    StdioMCPServerConfig,
    StdioSecret,
    deserialize_mcp_servers,
    serialize_mcp_servers,
)
from max_ai.errors.mcp import MCPConnectionLost, MCPServerConfigError
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode


class FakeManager:
    def __init__(self, tool_result: t.Any = None, resource_result: t.Any = None) -> None:
        self.tool_result = tool_result
        self.resource_result = resource_result
        self.calls: list[tuple[str, str, dict[str, t.Any]]] = []
        self.resource_calls: list[tuple[str, t.Any]] = []

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, t.Any],
        *,
        read_only: bool = False,
        on_progress: t.Any = None,
    ) -> t.Any:
        self.calls.append((server_id, tool_name, arguments))
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
    )

    result = await tool.execute(
        ToolCallRecord(tool_name=tool.name, parameters={"q": "abc"})
    )

    assert result.success is True
    assert result.result == "hello from mcp"
    assert result.metadata["server_id"] == "docs"
    assert result.metadata["mcp_tool_name"] == "search"
    assert manager.calls == [("docs", "search", {"q": "abc"})]


def test_http_server_config_adds_bearer_token_header() -> None:
    config = HTTPServerConfig(
        server_id="github",
        url="http://localhost:3000/mcp",
        secret=HTTPSecret(type="bearer", keys={"GITHUB_MCP_TOKEN": "secret"}),
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

        async def call_tool(self, name, arguments, read_timeout_seconds=None, progress_callback=None):
            self.calls.append((name, arguments, read_timeout_seconds))
            return CallToolResult(content=[TextContent(type="text", text="done")])

    client = FakeClient()
    monkeypatch.setattr(client_manager_module, "create_mcp_client", lambda config, extensions=None: client)
    manager = MCPClientManager()
    manager.add_server(HTTPServerConfig(
        server_id="docs", url="http://localhost:3000/mcp", trusted=True,
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
    assert await manager.call_tool("docs", "search", {"q": "x"}) == CallToolResult(
        content=[TextContent(type="text", text="done")]
    )
    assert client.calls == [("search", {"q": "x"}, manager.max_timeout)]
    await manager.disconnect_all()
    assert client.closed


def test_server_configs_round_trip_as_json(monkeypatch) -> None:
    monkeypatch.setenv("DOCS_TOKEN", "tok-123")
    monkeypatch.setenv("DOCS_KEY", "key-123")
    monkeypatch.setenv("CRM_KEY", "crm-456")
    monkeypatch.setenv("CRM_ID", "id-789")
    stdio = StdioMCPServerConfig(
        server_id="local", command="python", args=["-m", "server"],
        secret=StdioSecret(keys={"CRM_KEY": "crm-456", "CRM_ID": "id-789"}),
        tool_approval_modes={"search": "auto_approval"},
    )
    http = HTTPServerConfig(
        server_id="remote", url="https://example.com/mcp",
        secret=HTTPSecret(type="bearer", keys={"DOCS_TOKEN": "tok-123"}),
    )
    keyed = HTTPServerConfig(
        server_id="keyed", url="https://example.com/mcp",
        secret=HTTPSecret(type="api_key", header="X-Api-Key", keys={"DOCS_KEY": "key-123"}),
    )
    assert StdioMCPServerConfig.model_validate_json(stdio.model_dump_json()) == stdio
    assert HTTPServerConfig.model_validate_json(http.model_dump_json()) == http
    payload = serialize_mcp_servers([stdio, http, keyed])
    # Stored: only the names. Loaded: the values come from the environment again.
    assert '"keys":["DOCS_TOKEN"]' in payload and '"keys":["CRM_KEY","CRM_ID"]' in payload
    assert not any(value in payload for value in ("tok-123", "key-123", "crm-456", "id-789"))
    assert deserialize_mcp_servers(payload) == [stdio, http, keyed]
    assert http.request_headers == {"Authorization": "Bearer tok-123"}
    assert keyed.request_headers == {"X-Api-Key": "key-123"}
    assert stdio.process_env == {"CRM_KEY": "crm-456", "CRM_ID": "id-789"}


@pytest.mark.parametrize("build", [
    # A name the config does not know: before, the key was silently not sent.
    lambda: HTTPServerConfig(server_id="x", url="https://e.com/mcp", headers_env={"x-api-key": "K"}),
    lambda: HTTPServerConfig(server_id="x", url="https://e.com/mcp", env={"MODE": "test"}),
    # There is nowhere to write a value that would be stored: only ``secret``.
    lambda: HTTPServerConfig(server_id="x", url="https://e.com/mcp", headers={"Authorization": "Bearer t"}),
    lambda: StdioMCPServerConfig(server_id="x", command="python", env={"CRM_TOKEN": "t"}),
])
def test_a_config_refuses_a_name_it_does_not_know(build) -> None:
    with pytest.raises(ValidationError):
        build()


def test_a_secret_shows_its_names_and_never_its_values() -> None:
    http = HTTPServerConfig(
        server_id="remote", url="https://example.com/mcp",
        secret=HTTPSecret(type="bearer", keys={"DOCS_TOKEN": "sk-value"}),
    )
    assert http.request_headers["Authorization"] == "Bearer sk-value"
    for shown in (http.model_dump_json(), repr(http), str(http.secret)):
        assert "sk-value" not in shown and "DOCS_TOKEN" in shown


def test_a_missing_env_var_fails_when_connecting_and_is_named(monkeypatch) -> None:
    monkeypatch.delenv("NOPE_TOKEN", raising=False)
    stored = '{"server_id": "remote", "url": "https://example.com/mcp", '
    stored += '"secret": {"type": "bearer", "keys": ["NOPE_TOKEN"]}}'
    http = HTTPServerConfig.model_validate_json(stored)  # loading does not fail
    with pytest.raises(MCPServerConfigError, match="needs the env var.*NOPE_TOKEN"):
        http.request_headers


@pytest.mark.parametrize(
    "secret",
    [
        {"type": "bearer", "keys": {"A": "1", "B": "2"}},  # one value, one header
        {"type": "bearer", "header": "x-api-key", "keys": {"A": "1"}},
        {"type": "api_key", "keys": {"A": "1"}},  # which header?
        {"keys": {"A": "1"}},  # no type
        {"type": "bearer", "keys": {}},
    ],
)
def test_an_http_secret_that_does_not_say_one_clear_way_is_refused(secret) -> None:
    with pytest.raises((MCPServerConfigError, ValueError)):
        HTTPSecret(**secret)


def test_each_kind_of_server_only_takes_its_own_secret() -> None:
    with pytest.raises(ValueError):
        StdioMCPServerConfig(
            server_id="local", command="python",
            secret=HTTPSecret(type="bearer", keys={"A": "1"}),
        )

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

        async def call_tool(self, name, arguments, read_timeout_seconds=None, progress_callback=None):
            raise asyncio.CancelledError  # what the transport does when the stream breaks

    monkeypatch.setattr(client_manager_module, "create_mcp_client", lambda config, extensions=None: DroppingClient())
    manager = MCPClientManager()
    manager.add_server(HTTPServerConfig(server_id="web", url="http://localhost:3000/mcp"))
    await manager.connect("web")
    with pytest.raises(MCPConnectionLost, match="interrupted the request"):
        await manager.call_tool("web", "search", {})

    await manager.disconnect_all()


async def test_only_read_only_calls_are_sent_again_after_a_drop(monkeypatch):
    import asyncio

    from max_ai.capabilities.mcp import client_manager as client_manager_module
    from max_ai.capabilities.mcp.client_manager import MCPClientManager
    from max_ai.core.retry import RetryPolicy

    calls: list[str] = []

    class DropsOnce:
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

        async def call_tool(self, name, arguments, read_timeout_seconds=None, progress_callback=None):
            calls.append(name)
            if len(calls) % 2:
                raise asyncio.CancelledError  # the stream broke mid-call
            return CallToolResult(content=[TextContent(type="text", text="done")])

    monkeypatch.setattr(client_manager_module, "create_mcp_client", lambda config, extensions=None: DropsOnce())
    manager = MCPClientManager(retry=RetryPolicy(base_delay=0))
    manager.add_server(HTTPServerConfig(server_id="web", url="http://localhost:3000/mcp"))
    await manager.connect("web")

    result = await manager.call_tool("web", "search", {}, read_only=True)
    assert result.content[0].text == "done" and calls == ["search", "search"]
    with pytest.raises(MCPConnectionLost):
        await manager.call_tool("web", "send_email", {})
    assert calls[2:] == ["send_email"]  # it may have run: never sent twice

    await manager.disconnect_all()
