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
| Memory      |  | tools/      |  | rendered layers |  | Ollama adapter included  |
| Skills      |  | skills/     |  | layer usage     |  | provider contract        |
| Knowledge   |  | artifacts/  |  | token counts    |  +------------+-------------+
| Routines    |  +-------------+  +----------------+               |
| Logbook     |                                                   |
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

The `AgentCapabilities` manager is the component catalog. It accepts standalone tools plus capability registries for memory, skills, logbook context, routines, knowledge, and workspace. It validates tool-name uniqueness, validates `priority_tools`, loads skill metadata during `prepare()`, and exposes the final tool list through `all_tools`.

The `LayerContainer` stores the prompt stack. Default layers are:

```text
AgentPolicyLayer
TaskAnalysisLayer
RenderingLayer
PriorityToolsLayer
SkillsLayer
RoutineLayer
KnowledgeLayer
ContextLayer
MemoryLayer
```

Each layer is a Jinja2-backed `CoreLayer`. Layers validate their template variables at construction time, so broken prompt contracts fail before the agent runs. During `Agent.prepare()`, `PromptVariablesBuilder` collects the variables needed by each layer from the agent's capabilities and renders them into a `PromptCtx`.

The `CoreChatCompletionClient` is the provider boundary. The agent and reasoning loop do not know provider wire formats. A concrete client decides how to combine rendered layers and chat history into provider messages, convert `CoreMessage` objects to API payloads, convert `CoreTool` objects to provider tool schemas, normalize token usage, and perform streaming or non-streaming completions.

The included `OllamaChatCompletionClient` talks to Ollama through `ollama.AsyncClient`, supports OpenAI-style tool schemas, streaming, thinking configuration, and structured output schemas.

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

`max_ai.base.agent.Agent` is the main public abstraction. It exposes three APIs:

- `run_stream_events()` yields every `CoreEvent` and then one final `AgentResponse`.
- `run_stream()` yields assistant text chunks only.
- `run()` consumes the event stream and returns the final `AgentResponse`.

Use `run_stream_events()` for UIs, tracing, approvals, and debugging. Use `run_stream()` for chat-like text streaming. Use `run()` for scripts and tests.

### RunContext

`RunContext` carries per-run state: messages, user/session identifiers, run identifiers, and tool state. The reasoning loop mutates it as the turn progresses. Tool results are appended as `ToolMessage` instances so the model can reason over the outcome on the next iteration.

### Prompt Layers

Prompt layers are small, typed prompt fragments. Each layer declares required and optional variables, validates its Jinja2 template, and renders once during `prepare()`. Provider clients decide how rendered layers are assembled into the final model request.

Override default layers by passing `framework_layers` to `Agent`. An override of the same concrete type replaces the default. A new layer type is appended.

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

Tools implement `CoreTool` or are regular Python callables wrapped by `FunctionAsTool`. The `@tool` decorator is the easiest entry point.

```python
from max_ai.tools import tool


@tool(approval_mode="ask_for_approval")
def get_weather(city: str, unit: str = "celsius") -> dict[str, str | int]:
    """Return a weather report for a city."""
    return {
        "city": city,
        "condition": "sunny",
        "temperature": 24,
        "unit": unit,
    }
```

The decorator derives a tool name, description, and JSON schema from the Python function. Use approval modes for tools that can mutate state, access external systems, spend money, or expose sensitive data.

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

`DockerExecutor` isolates runtime execution. It stages the framework code for import inside the container, bind-mounts the per-user runtime directory, and executes normal tools as one-shot containers. Bash-like runtime tools can reuse a short-lived container session.

Host runtime layout:

```text
<workspace_root>/<user_id>/
  tools/
  skills/
  artifacts/
```

Container runtime layout:

```text
/mnt/tools
/mnt/skills
/mnt/artifacts
```

### Middleware

Middleware wraps tool execution. The tool executor performs name resolution, approval checks, and parameter validation before the middleware chain, then runs the actual side-effecting `tool.execute()` call through `MiddlewareChain`.

### Context Compaction

Before the reasoning loop starts, the agent can compact old context if the client declares a `max_context_window`. The default strategy is `SlidingWindowCompaction`, which preserves recent messages while keeping prompt and history under budget.

### Events and Responses

The event stream exposes model calls, streamed chunks, reasoning iterations, tool calls, tool responses, approval pauses, compaction, and errors. The final item is always an `AgentResponse` unless the caller cancels the run.

`AgentResponse` contains the final `RunContext`, the source agent name, usage metrics, and a normalized finish reason such as `stop`, `max_iterations`, `approval_needed`, `tool_direct_return`, `no_result`, `error`, or `cancelled`.

## Installation

This project targets Python 3.11 or newer.

```bash
uv sync
```

For UI dependencies:

```bash
uv sync --group ui
```

The project includes Ollama support. To run the example agent, make sure an Ollama server is reachable and the configured model is available.

## Minimal Agent

```python
import asyncio

from max_ai.base.agent import Agent
from max_ai.clients.ollama import OllamaChatCompletionClient
from max_ai.core.models import ModelConfig
from max_ai.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main() -> None:
    client = OllamaChatCompletionClient(
        model="llama3.1:8b",
        config=ModelConfig(
            supports_function_calling=True,
            max_context_window=8192,
        ),
    )

    agent = Agent(
        name="Sara",
        description="A helpful local assistant.",
        instructions="Be concise and useful.",
        client=client,
        toolset=[add],
    )

    response = await agent.run("What is 21 + 21?")
    print(response.finish_reason)
    print(response.context.messages[-1].content)


asyncio.run(main())
```

## Docker and Skills Example

`examples/file.py` builds a Docker-backed agent with local skills:

```python
from pathlib import Path

from max_ai.base.agent import Agent
from max_ai.capabilities.skills.local import LocalSkillRegistry
from max_ai.clients.ollama import OllamaChatCompletionClient
from max_ai.core.models import ModelConfig
from max_ai.executor import DockerExecutor
from max_ai.ui import server
from docker_tools import get_weather, check_link


EXAMPLE_DIR = Path(__file__).parent
TOOL_SOURCE = EXAMPLE_DIR / "docker_tools.py"
SKILLS_SOURCE = EXAMPLE_DIR / "LocalSkills"


agent = Agent(
    name="Sara",
    description="Example Max AI agent.",
    instructions="You are a helpful assistant. Be concise and useful.",
    client=OllamaChatCompletionClient(
        model="gemma4:e2b-it-q4_K_M",
        host="http://ollama:11434",
        config=ModelConfig(
            max_context_window=15000,
            supports_function_calling=True,
            supports_thinking=True,
            supports_vision=True,
        ),
        think=False,
        max_tokens=10000,
    ),
    skills=LocalSkillRegistry(
        source=SKILLS_SOURCE,
        skills=["create-report", "create-ppt"],
    ),
    toolset=[get_weather, check_link],
    executor=DockerExecutor(
        tool_files=TOOL_SOURCE,
        image="maxai-sandbox:skills-demo-v3",
    ),
)

server(agent, host="0.0.0.0", port=8000)
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
  base/              Core contracts: Agent, clients, tools, layers, reasoning
  capabilities/      Local capability implementations
  clients/           Provider adapters, including Ollama
  compaction/        Context compaction strategies
  core/              Messages, events, models, primitives, tool state
  executor/          Local and Docker execution backends
  manager/           Capability and prompt stack builders
  middleware/        Middleware chain implementation
  reasoning/         ReAct loop implementation
  stacks/            Built-in prompt layers and Jinja templates
  tools/             Built-in tools and @tool decorator
  types/             Pydantic types used across runtime boundaries
  ui/                FastAPI/static web UI
examples/
  file.py            Docker-backed agent + UI example
  docker_tools.py    Example decorated tools
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
