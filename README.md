# Max AI

Max AI is a Python framework (3.11+) for building tool-using LLM agents. One
`Agent` instance serves many users: the conversation lives in a `RunContext`
that your application keeps and saves, not in the agent. A built-in *harness*
checks the agent when it is built, adds the tools it needs, decides which tool
calls may run, and compacts the conversation when the context window fills.

Status: alpha (0.1.0, unreleased). The core was rewritten in November 2026 and
the API may still change. See [Status and known gaps](#status-and-known-gaps).

## Installation

```bash
pip install maxai                       # the package is "maxai", the import is "max_ai"
pip install "maxai[cli,embeddings,tracing,mongodb,e2b]"
```

| Extra | Adds | For |
|---|---|---|
| `cli` | textual | the terminal UI (`max_ai.cli.run_cli`) |
| `embeddings` | fastembed | local embeddings (`FastEmbedEmbedding`) |
| `tracing` | OpenTelemetry SDK + OTLP exporter | sending traces to Langfuse or any OTLP backend |
| `mongodb` | pymongo | `MongoDBSessionStore`, MongoDB quota store and knowledge |
| `e2b` | e2b | `E2BSandbox` |

The web UI (`max_ai.web.run_web`) uses Starlette and Uvicorn, which come with
the `mcp` dependency. For development: `uv sync --group dev`.

## Quick start

```python
import asyncio

from max_ai import Agent, OpenAIChatCompletionClient, tool
from max_ai.types.run_context import RunContext


@tool(read_only=True)
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main() -> None:
    agent = Agent(
        name="helper",
        description="A small assistant that can add numbers.",
        instructions="Answer briefly.",
        client=OpenAIChatCompletionClient(model="gpt-4o-mini"),  # reads OPENAI_API_KEY
        toolset=[add],
    )
    ctx = RunContext(user_id="ana", session_id="first")
    state = await agent.run("What is 21 + 21?", ctx)
    print(state.finish_reason, state.message.text() if state.message else "")
    await agent.close()


asyncio.run(main())
```

`run()` returns a `LoopState`: the last assistant message (`state.message`),
why the turn ended (`state.finish_reason`), the token `usage` and the number of
model calls (`iteration`). The conversation stays in `ctx`; pass the same `ctx`
to continue it.

### With a sandbox, skills and memory

```python
from max_ai import Agent, OpenRouterChatCompletionClient
from max_ai.capabilities.memory import Memory
from max_ai.capabilities.sandbox.docker import DockerSandbox
from max_ai.capabilities.skills import GitHubSkills
from max_ai.capabilities.tools import BashTool

agent = Agent(
    name="analyst",
    description="Builds reports from the user's files.",
    instructions="Work in the session folder. Say what you changed.",
    client=OpenRouterChatCompletionClient(model="openai/gpt-4o-mini"),  # OPENROUTER_API_KEY
    toolset=[BashTool()],                       # skills need bash
    sandbox=DockerSandbox(networks=["pypi.org", "files.pythonhosted.org"]),
    skills=GitHubSkills("org/agent-skills", ["create-report"]),  # GITHUB_TOKEN for private repos
    memory=Memory(),
)
```

## The Agent

```python
Agent(
    name, description, instructions, client,
    *, toolset=None, mcp=None, memory=None, knowledge=None, skills=None,
    reasoning=None, middlewares=None, output_format=None, prompt_layers=None,
    sandbox=None,
)
```

| Argument | Type | Notes |
|---|---|---|
| `client` | `CoreChatCompletionClient` | `OpenAIChatCompletionClient` or `OpenRouterChatCompletionClient` |
| `toolset` | list of `CoreTool` | your tools, `FunctionAsTool` / `@tool`, `BashTool()` |
| `mcp` | `StdioMCPServerConfig` / `HTTPServerConfig` | tools from MCP servers |
| `sandbox` | `SandboxCore` | `LocalSandbox`, `DockerSandbox`, `E2BSandbox` |
| `skills` | `SkillsCore` | `LocalSkills`, `GitHubSkills`; needs a sandbox and `BashTool()` |
| `memory` | `Memory` | Markdown files in the sandbox; needs a sandbox |
| `reasoning` | `CoreLoop` | default `ReactLoop(compaction=SummaryCompaction(...))` |
| `middlewares` | list of `CoreMiddleware` | `LoggingMiddleware`, `BudgetMiddleware`, `TracingMiddleware` |
| `output_format` | pydantic `BaseModel` class | structured final answer in `state.message.structured_output` |
| `prompt_layers` | list of `CoreLayer` | your own prompt text, e.g. `CustomLayer("...")` |
| `knowledge` | list of `CoreKnowledgeRegistry` | accepted but not wired yet, see known gaps |

Methods:

- `await agent.run(task, ctx)` runs one turn and returns a `LoopState`.
- `agent.run_stream(task, ctx)` yields only the text of the answer as it is written.
- `agent.run_stream_events(task, ctx, stream=False)` yields every `CoreEvent`
  (model calls, chunks, tool calls and results, compaction, errors), for UIs and debugging.
- `await agent.approve(ctx, call_id, always=False)` approves a call that waits
  for the user; then call `agent.run(None, ctx)` to continue. With `always=True`
  the rule is saved in the user's `setting.json` (never for a dangerous command).
- `await agent.close()` refuses new runs, waits for running ones, then closes MCP
  connections, the users' sandboxes and the client. `Agent` has no `async with`.

Pausing for the user:

```python
state = await agent.run("Clean the build folder", ctx)
if state.finish_reason == "waiting_for_user":
    for record in ctx.tool_state.pending_approvals:
        await agent.approve(ctx, record.id)       # or record.reject()
    state = await agent.run(None, ctx)
```

Questions from `ask_user` are in `ctx.tool_state.pending_user_input`; answer
with `record.store_result(AskUserTool.answer(record, answers))` and run with
`None`. If the user sends a new message instead, the waiting calls are closed
without running.

The constructor checks every piece and raises a clear error: wrong types,
skills or memory without a sandbox, skills without `BashTool()`, bash without a
sandbox, and the harness token limits below.

## How a turn runs

1. Turns of the same `(user_id, session_id)` wait for each other; other
   sessions run in parallel on the same agent.
2. The task goes into `ctx.inbox`; the middlewares wrap the whole turn.
3. Getting ready: the user's sandbox is bound (`sandbox.bind(user_id)`) and
   prepared once per user, and the skills are synced into it. The tool
   registry is built from your tools, the harness's tools and the MCP servers'
   tools (first name wins). The permission rules are read from `setting.json`,
   the memory index is read, and the system prompt is built from the prompt
   layers. The harness checks the prompt and tool sizes.
4. `ReactLoop` calls the model (with retries and middleware), runs the
   requested tools (read-only calls of one reply in parallel, up to 8), and
   repeats until the model answers without tools, a call waits for the user,
   `max_iterations` (50) is reached, or a rule stops it (repeated calls, empty
   answers). Before each model call it compacts the conversation if the window
   is filling.
5. `state.finish_reason` says why it ended: the model's own reason (`stop`,
   `length`), `waiting_for_user`, `max_iterations`, `cancelled`, `error`, or the
   reason a middleware or hook gave.

## The harness

`max_ai/core/harness/` holds the rules applied to every agent.

**Rules when the agent is built** (`rules.py`): description and instructions
at most about 100 tokens each, each tool description at most 100, your prompt
layers at most 1,500 together; at run time all loaded tools at most 20,000 and
the system prompt at most 10% of the window (a prompt under 2,000 tokens only
logs a warning). The harness also sets the model limits: `max_context_window`
and `max_output_tokens` you pass are replaced by its own defaults, with a warning.

**Tools it adds** (`Harness.tool_combination`):

| Tool | When |
|---|---|
| `ask_user`, `write_todos` | always |
| `read_file`, `write_file`, `edit_file` | with a sandbox |
| `save_memory` | with a memory that is not `read_only` |
| `read_skill` | with skills |
| `enter_worktree`, `exit_worktree` | when `bash` is in the toolset |
| `search_tools`, `call_tool` | when deferred tools (`deferred=True`) take 5,000 tokens or more; below that they are loaded directly |

`bash` itself is never added: pass `BashTool()` in `toolset`. A tool of yours
with the same name wins. `TracingMiddleware` is always added first.

**Permissions** (`permissions.py`): each user's rules live in
`.agent/setting.json` in their sandbox under `permissions`, with `allow`, `ask`
and `deny` lists. A rule is `tool` or `tool(pattern)`: a path pattern for file
tools; for bash the exact command, `bash(pytest -q)`, or a word prefix,
`bash(git status:*)`. Deny wins, then ask, then allow; a compound command needs
every part allowed. Some calls are always asked whatever the rules say:
`rm`, `sudo`, `git push --force`, `git reset --hard`, `find -delete`, command
substitution, anything touching `setting.json` or writing to the skills
folder. Without a matching rule, the tool's own `approval_mode` decides
(`BashTool` asks, the file tools run).

**Judge (optional, off by default)** (`judge.py`): set `HARNESS_JUDGE_MODEL`
(and `OPENROUTER_API_KEY`) and a bash command that would otherwise ask is sent
to that model on OpenRouter's decisions endpoint. If it is at least
`HARNESS_JUDGE_MIN_SAFE` (0.9) sure the command is safe, it runs without asking.
It only removes questions; it never overrides deny, ask or dangerous rules, and
any failure falls back to asking. **Data flow:** each judged call sends the
command, the model's description of it, and the user's last message (up to
4,000 characters each) to OpenRouter.

**Memory at compaction** (`hooks.py`): when messages leave the window, a model
extracts up to 5 facts the user stated about themselves or their work and saves
them to the memory (`HARNESS_MEMORY_AT_COMPACTION=false` turns it off).

## Capabilities

### Clients

`OpenAIChatCompletionClient(model, api_key=None, base_url=None, ...)` and
`OpenRouterChatCompletionClient(model, ..., fallback_models=None, provider=None)`,
both on the Chat Completions API. Without `api_key` they read `OPENAI_API_KEY`
/ `OPENROUTER_API_KEY`; the serialized config stores only the variable name.
`options={...}` is sent as is (temperature, top_p...). There is no Ollama
client; an OpenAI-compatible server can be used through `base_url`.

### Tools

```python
from max_ai import FunctionAsTool, ToolApprovalMode, tool

@tool(approval_mode=ToolApprovalMode.ASK_APPROVED)
def send_invoice(customer_id: str) -> str:
    """Send the open invoice to a customer."""
    ...

def get_weather(city: str) -> dict:
    """Return the current weather for a city."""
    ...

weather = FunctionAsTool(get_weather, read_only=True, deferred=True)
```

Name, description and JSON schema come from the signature and docstring.
`@tool` defaults to `AUTO_APPROVED`; `FunctionAsTool` and `CoreTool` default
to `ASK_APPROVED`. `read_only=True` lets calls run in parallel; `deferred=True`
keeps the tool out of the prompt until found with `search_tools`.

### MCP

```python
from max_ai.capabilities.mcp import HTTPServerConfig, StdioMCPServerConfig

mcp = [
    HTTPServerConfig(server_id="docs", url="https://example.com/mcp", token_env="DOCS_TOKEN"),
    StdioMCPServerConfig(server_id="fetch", command="uvx", args=["mcp-server-fetch"]),
]
```

Servers connect on the first run and are shared until `agent.close()`. Each has
`connect_timeout_seconds` (30), `retry_after_seconds` (60) and `required`
(fail the run instead of skipping a server that is down). MCP tools default to
`ASK_APPROVED` (`approval_mode`, `tool_approval_modes`) and run in the host
process, not in the sandbox.

### Sandbox

Commands, file tools, skills and memory live in the user's sandbox, in an
`.agent/` folder with `setting.json`, `skills/`, `workspace/<session>/`,
`tmp/<session>/` and `memory/`.

```python
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.sandbox.docker import DockerSandbox
from max_ai.capabilities.sandbox.e2b import E2BSandbox

LocalSandbox(root="./box")                                  # this machine, not isolated
DockerSandbox()                                             # built from our Dockerfile
DockerSandbox(dockerfile="./coding.Dockerfile", networks=["pypi.org"])
E2BSandbox(networks=["pypi.org"])                           # E2B_API_KEY, maxai[e2b]
```

Security:

- **`LocalSandbox` is not isolation.** Commands run as you, on your machine.
  The environment is filtered (API keys in your process are not passed), but a
  command can still read any file you can, including `.env` and `~/.ssh`. All
  users of one `LocalSandbox` share the same `root/.agent`. Use it for local
  development only. Use `DockerSandbox` or `E2BSandbox` for untrusted users or
  content (web pages, uploaded files, third-party skills).
- **`DockerSandbox`** runs one container per user (`docker-sandbox-<user_id>`)
  with a volume of the same name. The model's commands run as uid 1000 with
  `--cap-drop ALL`, `no-new-privileges` and a read-only root filesystem; they can
  write only the workspace, tmp and memory, not `setting.json` or the skills.
  With no `networks` there is no network; `networks=[...]` is an allowlist of
  domains reached through a proxy (HTTP(S) clients that honour `HTTPS_PROXY`).
  An open domain such as github.com is also a way to send files out. Limits:
  `memory="2g"`, `cpus=2`, `processes=512`. The host needs the `docker` CLI.
- **`E2BSandbox`** runs a cloud machine per user; commands run as a user
  without sudo, with the same network allowlist model.
- Only variables named in `extra_env` (and the skills' `skill_env`) reach the
  commands.

### Skills

A skill is a folder with a `SKILL.md` (frontmatter `name`, `description`) and
optional scripts. `LocalSkills(path, skills=[...])` reads `<path>/<name>/`;
`GitHubSkills(url, skills=[...])` downloads each folder through the GitHub API
(`url` may be `org/repo` or `https://github.com/org/repo/tree/<branch>/<folder>`;
token from `token=` or `GITHUB_TOKEN`, never serialized). Skills are
installed into the sandbox and updated when their hash or git id changes; the
prompt lists their descriptions and the model reads one with `read_skill`.
`skill_env={"NAME_IN_SKILL": "NAME_IN_YOUR_ENV"}` passes values to their
commands. Check each skill's license.

### Memory

`Memory(read_only=False)` keeps one Markdown file per fact in `.agent/memory/`.
An index of the files (one line each, up to 200) is shown in the system prompt;
the model saves facts with `save_memory` and reads files with `read_file`. With
`read_only=True` there is no `save_memory` and writes to the folder are refused.

### Compaction

`ReactLoop(max_iterations=50, compaction=...)`. The default loop uses
`SummaryCompaction`: when the messages fill `threshold` (0.8) of the room, long
old tool results are cut, then old messages are replaced by a summary. The
summary model is `SUMMARY_MODEL` on OpenRouter if that variable and
`OPENROUTER_API_KEY` are set, otherwise the agent's own model.
`SlidingWindowCompaction` drops old messages without a summary. Both are in
`max_ai.capabilities.compaction`.

### Prompt layers

The system prompt is built by `NativePrompt` from layers in a fixed order (stable
parts first, for provider caching): task analysis, rendering, agent policy,
your layers, tools, skills, knowledge, workspace, memory, session. Add your
own text with `prompt_layers=[CustomLayer("Our refund policy: ...")]`
(`from max_ai.capabilities.stacks import CustomLayer`).

### Middleware

Subclass `CoreMiddleware` and override any of `turn`, `model`, `stream` and
`tool`; each receives `call_next`. Keep per-turn data in `self.state(ctx)`,
never on `self`, and raise `self.stop(...)` to end the turn. Built-ins in
`max_ai.capabilities.middleware`: `LoggingMiddleware`, `BudgetMiddleware`
(tokens, cost, seconds, model and tool calls per turn; per-user quotas with
`quota=` and a `LocalQuotaStore` / `MongoDBQuotaStore`), and
`TracingMiddleware` with `configure_langfuse()`.

### Sessions

The host loads and saves conversations; the agent never touches the store.

```python
from max_ai.capabilities.session_store import LocalSessionStore

store = LocalSessionStore("./sessions")             # or MongoDBSessionStore (maxai[mongodb])
ctx = await store.load("ana", "first") or RunContext(user_id="ana", session_id="first")
await agent.run("Continue.", ctx)
await store.save(ctx)
```

### Serialization

Every component is plain data with no secrets: `agent.serialize()`,
`Agent.deserialize(data)`, `agent.to_yaml()` / `Agent.from_yaml(path_or_text)`,
`to_json()` / `from_json()`. Python-function tools cannot be stored; expose
them through MCP. Classes outside `max_ai` load only after
`allow_providers("my_pkg.")` or `MAXAI_ALLOWED_PROVIDERS`.

### CLI and web UI

```python
from max_ai.cli import run_cli      # maxai[cli]
from max_ai.web import run_web

await run_cli(agent, store=LocalSessionStore("./sessions"), user_id="ana")
await run_web(agent, store=LocalSessionStore("./sessions"), user_id="ana", port=8000)
```

Both close the agent when they end. See [max_ai/cli/README.md](max_ai/cli/README.md).

## Configuration

Settings are read from the environment or `.env` (`max_ai/config.py`). The
ones the code uses today:

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY`, `OPENROUTER_API_KEY` | | client keys |
| `TOOL_TIMEOUT_SECONDS` | 360 | most one tool call may run |
| `HARNESS_SANDBOX_SECONDS` | 600 | idle life of a sandbox; must be at least the tool timeout |
| `HARNESS_RETRY_ATTEMPTS` | 3 | tries per model call |
| `HARNESS_TOOLS_MAX_TOKENS` | 20000 | most all loaded tools may take |
| `HARNESS_DEFERRED_TOOLS_MIN_TOKENS` | 5000 | below this, deferred tools are loaded directly |
| `HARNESS_PROMPT_MAX_SHARE` | 0.10 | most of the window the system prompt may take |
| `HARNESS_CUSTOM_PROMPT_MAX_TOKENS` | 1500 | most your prompt layers may take |
| `SUMMARY_MODEL` | empty | OpenRouter model for compaction summaries |
| `HARNESS_MEMORY_AT_COMPACTION` | true | save user facts when messages leave |
| `HARNESS_JUDGE_MODEL` | empty | OpenRouter model that judges bash commands (off when empty) |
| `HARNESS_JUDGE_MIN_SAFE`, `HARNESS_JUDGE_SECONDS` | 0.9, 3 | judge threshold and timeout |
| `GITHUB_TOKEN`, `E2B_API_KEY` | | `GitHubSkills`, `E2BSandbox` |
| `MAXAI_ALLOWED_PROVIDERS` | | extra packages `deserialize` may load |

## Status and known gaps

Checked against the code on this branch:

- **`knowledge=` is not wired.** The agent validates and serializes knowledge
  registries (local, SQLite, MongoDB), but never adds their `search_knowledge`
  tool to the registry, and the knowledge prompt layer receives no tool names,
  so it renders empty. The model cannot query knowledge today.
- **Completion gates are not used.** `max_ai/capabilities/completion_gate` and
  `base/completion_gate.py` still exist, but no loop calls them and `Agent`
  has no `gates` argument.
- **Quota stores** (`LocalQuotaStore`, `MongoDBQuotaStore`) are used only by
  `BudgetMiddleware(quota=..., quota_store=...)`; nothing else reads them.
- **Examples:** most files in `examples/` use the old API, and
  `advance/04_custom_middleware.py` imports the removed `AgentResponse`. See
  [examples/README.md](examples/README.md).

## Repository map

```text
max_ai/
  agents/           Agent
  base/             contracts: clients, tools, loop, compaction, sandbox, skills, middleware, layers
  core/harness/     rules, permissions, judge, hooks
  core/             messages, events, tool registry and dispatcher, middleware chain, retry
  capabilities/
    clients/        OpenAI, OpenRouter
    loop/react/     ReactLoop
    compaction/     SummaryCompaction, SlidingWindowCompaction
    sandbox/        local, docker, e2b
    skills/         local, github
    memory/         Memory (Markdown files in the sandbox)
    tools/          bash, files, ask_user, todos, worktree, search/call_tool, read_skill, save_memory
    stacks/         prompt layers
    mcp/            MCP client
    middleware/     logging, budget, tracing
    session_store/  local, mongodb
    quota_store/    local, mongodb (used by BudgetMiddleware)
    knowledge/      local, sqlite, mongodb (not wired, see above)
    completion_gate/  not used by the current loop
  cli/              Textual terminal UI
  web/              browser UI
  types/            RunContext and other runtime types
examples/           see examples/README.md (mostly outdated)
docker-infra/       MongoDB Atlas Local, MCP server, Langfuse
tests/
```

## Testing

```bash
uv run pytest
uv run pytest -m "not integration"
```

## Roadmap

Planned, not started; order is not a commitment.

- Wire knowledge search into the agent, or remove it.
- Multi-agent delegation on the stateless `Agent`.
- Native clients: Anthropic first, then Gemini and Azure OpenAI.
- Optimistic concurrency in session stores.
- Guardrail middleware for PII and prompt injection.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
