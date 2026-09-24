# Max AI

Max AI is a Python framework for building tool-using LLM agents. It separates agent orchestration, prompt composition, model providers, tools, runtime execution, skills, memory, knowledge, and UI delivery into small components with clear contracts.

The default behavior is a ReAct-style agent loop:

1. Build and validate the agent configuration.
2. Load capabilities such as tools, skills, memory, routines, knowledge, and context.
3. Render a layered system prompt.
4. Call the LLM through a provider adapter.
5. Execute requested tools through a tool executor.
6. Feed tool results back into the conversation.
7. Repeat until the model produces a final answer, requests approval, errors, or reaches the configured iteration limit.

## Architecture

```text
                                  USER / APP / CLI / UI
                                           |
                                           v
                               +----------------------+
                               |  Agent.run*() APIs   |
                               |----------------------|
                               | run()                |
                               | run_stream()         |
                               | run_stream_events()  |
                               +----------+-----------+
                                          |
                                          v
+--------------------------------------------------------------------------------+
|                                  Agent                                         |
|--------------------------------------------------------------------------------|
| Owns configuration, lifecycle, prompt stack, capabilities, executor, client,    |
| reasoning loop, compaction strategy, middleware, and final AgentResponse.       |
+-----+----------------+------------------+------------------+-------------------+
      |                |                  |                  |
      | prepare()      | runtime setup    | prompt ctx       | reasoning
      v                v                  v                  v
+-------------+  +-------------+  +----------------+  +--------------------------+
|Capabilities |  | Workspace   |  | Prompt Stack   |  | Reasoning Loop           |
|Manager      |  | Registry    |  | LayerContainer |  | ReActLoop by default     |
+------+------+  +------+------+  +-------+--------+  +------------+-------------+
       |                |                 |                        |
       | collects       | materializes    | renders                | calls
       v                v                 v                        v
+-------------+  +-------------+  +----------------+  +--------------------------+
| Tools       |  | user runtime|  | PromptCtx       |  | Chat Client              |
| Memory      |  | tools/      |  | rendered layers |  | OpenAI, OpenRouter,      |
| Skills      |  | skills/     |  | layer usage     |  | Ollama adapters included |
| Knowledge   |  | artifacts/  |  | token counts    |  +------------+-------------+
| Workspace   |  +-------------+  +----------------+               |
+------+------+                                                   |
       |                                                          |
       | exposes tools / metadata                                 |
       v                                                          |
+-----------------------------+                                   |
| ToolExecutor                |<----------------------------------+
|-----------------------------|       LLM returns tool calls
| resolves tool names         |
| validates parameters        |
| handles approval states     |
| runs middleware             |
| dispatches execution        |
+-------------+---------------+
              |
              v
   +----------------------+       +----------------------+
   | LocalExecutor        |       | DockerExecutor       |
   | in-process tools     |       | sandboxed runtime    |
   +----------------------+       +----------+-----------+
                                             |
                                             v
                                  +----------------------+
                                  | /mnt/tools           |
                                  | /mnt/skills          |
                                  | /mnt/artifacts       |
                                  +----------------------+
```

## How the Components Talk

The `Agent` is the orchestration root. It receives a model client, optional tools, optional capability registries, optional custom prompt layers, an executor, and an optional reasoning loop. It normalizes everything during construction, prepares async resources before execution, then drives one run through `run_stream_events()`.

The agent manages capabilities (memory, skills, knowledge) and tool registration. It validates tool-name uniqueness, loads skill metadata during `prepare()`, and exposes the final tool list to the dispatcher.

The `LayerContainer` stores the prompt stack. Default layers (always included or conditionally added based on configuration):

```text
AgentPolicyLayer
TaskAnalysisLayer
RenderingLayer
SkillsLayer         (if skills are configured)
KnowledgeLayer      (if knowledge is configured)
MemoryLayer         (if memory is configured)
SessionStateLayer
```

Each layer is a Jinja2-backed `CoreLayer`. Layers validate their template variables at construction time, so broken prompt contracts fail before the agent runs. During `Agent.prepare()`, the layers are rendered with variables collected from the agent's capabilities into a `PromptCtx`.

The `CoreChatCompletionClient` is the provider boundary. The agent and reasoning loop do not know provider wire formats. A concrete client decides how to combine rendered layers and chat history into provider messages, convert `CoreMessage` objects to API payloads, convert `CoreTool` objects to provider tool schemas, normalize token usage, and perform streaming or non-streaming completions.

Included clients: `OpenAIChatCompletionClient` (OpenAI API), `OpenRouterChatCompletionClient` (OpenRouter), and `OllamaChatCompletionClient` (local Ollama). Each supports tool calling, streaming, and structured output.

The `ReActLoop` is the default reasoning engine. On each iteration it calls the client, appends the assistant message to `RunContext.messages`, registers any tool calls into `ctx.tool_state`, sends executable records to `ToolExecutor`, appends returned `ToolMessage` objects, and repeats. It stops when there are no tool calls, approval is needed, cancellation happens, no model result is returned, or the configured loop limit is reached.

The `ToolExecutor` is the tool-call pipeline. It receives `ToolCallRecord` objects from the reasoning loop, resolves tool names, validates parameters against each tool's JSON schema, emits approval events when needed, runs middleware around the execution step, dispatches execution to an executor, and returns both observability events and `ToolMessage` results for the LLM.

The executor controls where side effects happen. `LocalExecutor` runs trusted tools in the same Python process. `DockerExecutor` runs tools in a sandboxed Docker runtime and bind-mounts a per-user workspace as `/mnt`, with `tools`, `skills`, and `artifacts` directories.

The workspace registry creates per-user runtime directories. When tools or skills require runtime files, the agent binds the executor to the workspace, materializes the user's runtime directory, copies selected skills into it, and passes paths such as `runtime_root`, `tools_dir`, `skills_dir`, and `artifacts_dir` to runtime-aware tools.

## Runtime Flow

```text
User task
   |
   v
Agent.run_stream_events(task)
   |
   +--> prepare()
   |      |
   |      +--> capabilities.prepare()
   |      |      +--> load skill metadata
   |      |      +--> validate final tool catalog
   |      |
   |      +--> PromptVariablesBuilder.collect(layer)
   |      +--> CoreLayer.render(...)
   |      +--> store rendered_layers + prompt token usage
   |
   +--> normalize task into RunContext.messages
   |
   +--> materialize workspace if tools or skills need runtime files
   |
   +--> build ToolExecutor(all_tools, middlewares, executor)
   |
   +--> optionally compact old context
   |
   +--> ReActLoop.execute_reasoning_loop(...)
          |
          +--> client.run(ctx, prompts, tools)
          |
          +--> assistant message
          |
          +--> no tool calls?
          |      |
          |      +--> final AgentResponse
          |
          +--> register tool calls in ctx.tool_state
          |
          +--> ToolExecutor.execute_tool_call(...)
                 |
                 +--> approval check
                 +--> parameter validation
                 +--> middleware chain
                 +--> LocalExecutor or DockerExecutor
                 +--> ToolMessage returned to ReActLoop
```

## Key Concepts

### Agent

`max_ai.agents.Agent` is the main public abstraction. It exposes three APIs:

- `run_stream_events()` yields every `CoreEvent` and then one final `AgentResponse`.
- `run_stream()` yields assistant text chunks only.
- `run()` consumes the event stream and returns the final `AgentResponse`.

Use `run_stream_events()` for UIs, tracing, approvals, and debugging. Use `run_stream()` for chat-like text streaming. Use `run()` for scripts and tests.

### RunContext

`RunContext` carries per-run state: messages, user/session identifiers, run identifiers, and tool state. The reasoning loop mutates it as the turn progresses. Tool results are appended as `ToolMessage` instances so the model can reason over the outcome on the next iteration.

### Prompt Layers

Prompt layers are small, typed prompt fragments. Each layer declares required and optional variables, validates its Jinja2 template, and renders once during `prepare()`. Provider clients decide how rendered layers are assembled into the final model request.

Override default layers by passing `prompt_layers` to `Agent`. An override of the same concrete type replaces the default. A new layer type is appended.

### Capabilities

Capabilities are registries that expose tools or prompt metadata:

- memory: persistent user facts and memory tools;
- skills: selected `SKILL.md` packages and runtime files;
- logbook/context: session summaries and conversation context;
- routines: reusable procedures exposed through search/fetch tools;
- knowledge: retrieval backends and their tools;
- workspace: per-user runtime filesystem.

The agent collects these automatically from constructor arguments.

### Tools

MCP servers are configured separately from `toolset`. Pass one or more
serializable server configurations through `mcp`:

```python
from max_ai.agents import Agent
from max_ai.capabilities.mcp import (
    HTTPServerConfig, StdioMCPServerConfig,
    serialize_mcp_servers, deserialize_mcp_servers,
)

servers = [
    HTTPServerConfig(server_id="docs", url="https://example.com/mcp"),
    StdioMCPServerConfig(
        server_id="local", command="uvx", args=["mcp-server-fetch"],
    ),
]
payload = serialize_mcp_servers(servers)
restored = deserialize_mcp_servers(payload)

agent = Agent(
    name="assistant", description="Uses MCP", instructions="Help the user",
    client=client, toolset=[local_tool], mcp=restored,
)
```

The agent discovers MCP tools when a run starts, prefixes their names with the
server ID, executes them in the host process, and closes the connections at the
end of the run. The Max AI client uses MCP 2.x and negotiates with both current
and earlier protocol versions.

Tools implement `CoreTool` or are regular Python callables wrapped by `FunctionAsTool`. Pass the wrapped tool in `toolset`:

```python
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.types.tools import ToolApprovalMode


def get_weather(city: str, unit: str = "celsius") -> dict[str, str | int]:
    """Return a weather report for a city."""
    return {
        "city": city,
        "condition": "sunny",
        "temperature": 24,
        "unit": unit,
    }

agent = Agent(
    name="assistant",
    description="Uses tools",
    instructions="Help the user",
    client=client,
    toolset=[
        FunctionAsTool(get_weather, approval_mode=ToolApprovalMode.ASK_APPROVED),
    ],
)
```

`FunctionAsTool` derives a tool name, description, and JSON schema from the Python function signature. Use approval modes for tools that can mutate state, access external systems, spend money, or expose sensitive data.

### Skills

Skills are local packages with a `SKILL.md` file and optional resources such as scripts or references.

```text
LocalSkills/
  create-report/
    SKILL.md
    scripts/
      create_report.py
  create-ppt/
    SKILL.md
    scripts/
      create_ppt.py
```

`LocalSkillRegistry` selects named skills from a source directory, caches them under `var/skills`, reads frontmatter metadata from `SKILL.md`, adds skill descriptions to the prompt through `SkillsLayer`, and materializes the selected skills into the runtime workspace before execution.

Skills require a sandbox executor. If skills are registered with the default local executor, the agent raises a safety error instead of running untrusted runtime code in-process.

### Execution

`LocalExecutor` is fast and simple. It runs tool code in the current Python process and is suitable for trusted helpers.

`DockerExecutor` isolates runtime execution. It stages the framework code for import inside the container, bind-mounts the per-user runtime directory, and executes tools as one-shot or persistent containers.

Import executors from `max_ai.capabilities.executor`:

```python
from max_ai.capabilities.executor import LocalExecutor, DockerExecutor

agent = Agent(
    name="assistant",
    description="Uses Docker",
    instructions="Help the user",
    client=client,
    executor=DockerExecutor(image="maxai-sandbox:latest"),
)
```

Per-user runtime layout on the host:

```text
<root>/<user_id>/
  workspace/          (shared workspace; bash starts here with $WORKSPACE)
  scratchpad/<session_id>/  (per-session temporary; $SCRATCHPAD)
  skills/             (materializes skills; $SKILLS)
```

### Middleware

`CoreMiddleware` is a serializable component with optional async hooks: run lifecycle (`on_run_start`, `on_final_response`, `on_run_end`, `on_run_error`), model calls (`on_model_request`, `on_model_chunk`, `on_model_response`, `on_model_error`), and approved tool calls (`on_tool_request`, `on_tool_response`). Raise `StopRun(message)` from `on_run_start` or a model hook to end the turn cleanly (`response.stop_message`); a tool hook blocks a single call by returning a `ToolResult` instead. Pass middlewares via `Agent(..., middlewares=[...])`. Built-ins: `LoggingMiddleware`, `BudgetMiddleware(max_tokens=..., max_cost_usd=..., max_seconds=..., quota=QuotaLimits(...), quota_store=LocalQuotaStore(...))`, `TracingMiddleware()` with `configure_langfuse()` for OpenTelemetry → Langfuse.

### Embeddings

`CoreEmbedding` (`max_ai.base.embedding`) turns text into vectors, in batches and with a cache. `FastEmbedEmbedding` (`pip install 'maxai[embeddings]'`) runs a local multilingual model; `OpenAIEmbedding` needs no download (better for serverless). Knowledge search uses one (local by default); memory uses one when given: `LocalMemoryRegistry(base_path=..., embedding=FastEmbedEmbedding())` makes `search_memory` match by meaning, in any language. `SQLiteMemoryRegistry` and `SQLiteKnowledgeRegistry` (`max_ai.capabilities.memory.sqlite`, `max_ai.capabilities.knowledge.sqlite`) store each vector next to its text, so stored memories and documents are never embedded again, even after a restart; knowledge is loaded with `upsert_block(block_id, KnowledgeBlock(...))`.

### Context Compaction

Compaction runs inside the loop before model calls when the window fills up. No default strategy (compaction=None means never compact). Strategies in max_ai.capabilities.compaction: `SummaryCompaction(threshold=0.8, keep_ratio=0.4)` (LLM summary of old turns; summary survives in prompt) and `SlidingWindowCompaction` (drops oldest turns without LLM).

### Events and Responses

The event stream exposes model calls, streamed chunks, reasoning iterations, tool calls, tool responses, approval pauses, compaction, and errors. The final item is always an `AgentResponse` unless the caller cancels.

`AgentResponse` contains the final `RunContext`, agent name, usage, and finish reason: `stop`, `max_iterations`, `output_limit`, `budget_exceeded`, `stopped`, `approval_needed`, `tool_denied`, `tool_direct_return`, `input_needed`, `no_result`, `error`, `incomplete`, `waiting`, or `cancelled`.

### Completion Gates

Gates accept or reject the model's proposed final response before the turn ends. The framework's `RuntimeCompletionGate` is always added; pass `RuntimeCompletionGate(RuntimeGateConfig(plan_must_close=False))` in `gates=[...]` to change its options. Custom gates subclass `CompletionBase` and implement `on_final_response(ctx) -> CompletionDecision`.

### Sessions

The host loads/saves runs: `ctx = await store.load(user_id, session_id)` (or new `RunContext(user_id=..., session_id=...)`), `response = await agent.run(text, run_context=ctx)`, `await store.save(response.context)`. `LocalSessionStore` in max_ai.capabilities.session_store. The Agent never touches the store.

### Concurrency

One Agent instance serves many users at once. Runs of different `(user_id, session_id)` execute in parallel; messages of the same session wait for each other. Memory is bound per run to `run_context.user_id` and `session_id`.

### Parallel Tools

Read-only tool calls of one model reply run at the same time (`FunctionAsTool(fn, read_only=True)`). Built-in read-only file tools and MCP tools with readOnlyHint already are. Other calls run alone, in order. Cap: env `MAX_PARALLEL_TOOLS` (default 8).

### Structured Output

`Agent(..., output_format=MyPydanticModel)` shapes only the final accepted answer. Read it from `response.final_message.structured_output`.

### Serialization

`row = agent.serialize().model_dump_json()`, `Agent.deserialize(row)`. Stored configs hold no secrets (clients store `api_key_env`, the env var name). Python-function tools can't be serialized—expose them through MCP. Third-party components need `from max_ai.base import allow_providers; allow_providers("my_pkg.")`.

## Installation

This project targets Python 3.11 or newer.

```bash
pip install maxai
```

Or with optional features:

```bash
pip install maxai[cli,embeddings,tracing,mongodb,runtime-modal]
```

For development:

```bash
uv sync --group dev
```

## Minimal Agent

```python
import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.core.model.llm import ModelConfig
from max_ai.types.tools import ToolApprovalMode


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main() -> None:
    client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        config=ModelConfig(
            supports_function_calling=True,
            max_context_window=128000,
        ),
    )

    agent = Agent(
        name="Sara",
        description="A helpful assistant.",
        instructions="Be concise and useful.",
        client=client,
        toolset=[FunctionAsTool(add, approval_mode=ToolApprovalMode.AUTO_APPROVED)],
    )

    response = await agent.run("What is 21 + 21?")
    print(response.finish_reason)
    print(response.context.messages[-1].content)


asyncio.run(main())
```

## Full-Featured Agent Example

`examples/01_agent_with_openai.py` demonstrates a complete agent with memory, knowledge, skills, and the CLI:

```python
from pathlib import Path

from max_ai.agents import Agent
from max_ai.base.knowledge import KnowledgeToolMode
from max_ai.base.memory import MemoryToolMode
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.compaction import SummaryCompaction
from max_ai.capabilities.knowledge.local import LocalKnowledgeRegistry
from max_ai.capabilities.memory.local import LocalMemoryRegistry
from max_ai.capabilities.skills.local import LocalSkillRegistry
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.middleware import TracingMiddleware
from max_ai.cli import run_cli
from max_ai.core.model.llm import ModelConfig
from max_ai.types.tools import ToolApprovalMode


def get_weather(city: str) -> dict:
    """Return the current weather for a city."""
    return {"city": city, "condition": "sunny", "temp_c": 24}


async def main() -> None:
    client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        config=ModelConfig(supports_function_calling=True, max_context_window=128000),
    )

    agent = Agent(
        name="Assistant",
        description="A helpful agent with memory and skills.",
        instructions="Help the user with their tasks.",
        client=client,
        toolset=[FunctionAsTool(get_weather, approval_mode=ToolApprovalMode.AUTO_APPROVED)],
        memory=LocalMemoryRegistry(base_path=Path("./local")),
        knowledge=[
            LocalKnowledgeRegistry(
                name="docs",
                description="Project documentation",
                base_path=Path("./local"),
                tool_mode=KnowledgeToolMode.FULL,
            )
        ],
        skills=LocalSkillRegistry(source=Path("./LocalSkills"), skills=["create-report"]),
        compaction=SummaryCompaction(threshold=0.8),
        middlewares=[TracingMiddleware()],
    )

    async with agent:
        await run_cli(agent, user_id="user_001")
```

## Extending the Framework

### Add a Provider Client

Subclass `CoreChatCompletionClient` and implement:

- `format_messages()`
- `build_api_messages()`
- `build_tool_schema()`
- `normalize_usage_stats()`
- `complete()`
- `stream()`

Keep provider-specific schema and message conversion inside the client. The agent, reasoning loop, and tool executor should continue to speak only in framework types.

### Add a Prompt Layer

Subclass `CoreLayer`, define a default template, and declare required or optional variables. Pass an instance through `framework_layers`. If the type matches a default layer, it replaces that default. Otherwise it is appended.

### Add a Reasoning Loop

Subclass `BaseReasoning`, declare a `LOOP_STATE_CLS`, and implement `execute_reasoning_loop()`. The agent will bind runtime dependencies with `bind(name, client, tool_executor, middleware_chain)` before execution.

### Add a Capability Registry

Subclass `CoreAgentCapabilities` or one of the specific registry contracts. A capability can provide tools, prompt metadata, persistent data, or runtime materialization behavior. `AgentCapabilities` discovers known registry types and merges their tools into the final catalog.

## Testing

```bash
uv run pytest
```

Focused examples:

```bash
uv run pytest tests/tools
uv run pytest tests/reasoning
uv run pytest tests/executor
```

## Repository Map

```text
max_ai/
  agents/            Agent orchestration and lifecycle
  base/              Core contracts: clients, tools, layers, reasoning, capabilities
  capabilities/      Local capability implementations
    clients/         Provider adapters (OpenAI, OpenRouter, Ollama)
    compaction/      Context compaction strategies
    completion_gate/ Completion decision gates
    executor/        Execution backends (Local, Docker, Modal)
    knowledge/       Knowledge registries and retrieval
    memory/          Memory registries and persistence
    middleware/      Logging, budget, tracing middleware
    mcp/             MCP server client integration
    reasoning/       ReAct loop implementation
    session_store/   Session persistence
    skills/          Skill registry and management
    tools/           Built-in tools (filesystem, bash, approval)
    workspace/       Workspace registry and runtime
  cli/               Textual terminal UI
  core/              Messages, events, tool state, primitives
  types/             Pydantic types used across runtime boundaries
examples/
  01_agent_with_openai.py    Full agent with OpenAI, memory, knowledge, skills
  02_agent_with_openrouter.py Full agent with OpenRouter
  LocalSkills/       Example skill packages
tests/               Unit, integration, executor, reasoning, and tool tests
```

## Design Principles

- Provider adapters own provider quirks.
- Reasoning loops own turn strategy, not prompt formatting.
- Prompt layers are validated before runtime.
- Tools are schema-validated before execution.
- Approval is represented as state, not as an exception.
- Sandboxed execution is required for skills.
- Events are first-class so UIs and debuggers can observe a run without scraping text output.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
