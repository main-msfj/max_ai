# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Until 1.0, minor versions may break
the API.

## [Unreleased] — 0.1.0

First public version. The core was rewritten between 2026-11-15 and
2026-11-19 (PRs #28–#39); this entry describes the code as it is now.

### Added
- `Agent(name, description, instructions, client, *, toolset, mcp, memory,
  knowledge, skills, reasoning, middlewares, output_format, prompt_layers,
  sandbox)` with `run`, `run_stream`, `run_stream_events`, `approve` and
  `close`. `run` returns a `LoopState` (`message`, `finish_reason`, `usage`).
  One instance serves many users; turns of one session queue, other sessions
  run in parallel. The conversation lives in the `RunContext` the host keeps.
- Harness (`max_ai/core/harness`): token limits on description, instructions,
  tools, prompt layers and the system prompt; the tools it adds (`ask_user`,
  `write_todos`, file tools, `save_memory`, `read_skill`, worktree tools,
  `search_tools`/`call_tool` for deferred tools); `TracingMiddleware` always
  first; retries per model call.
- Permissions: per-user `allow` / `ask` / `deny` rules in the sandbox's
  `.agent/setting.json`; dangerous commands are always asked; "always allow"
  saves a rule through `agent.approve(..., always=True)`.
- Optional judge (`HARNESS_JUDGE_MODEL`): an OpenRouter decisions model may let
  a bash command run without asking. It receives the command, the model's
  description of it and the user's last message.
- Clients: `OpenAIChatCompletionClient` and `OpenRouterChatCompletionClient`.
  Configs store the env var name of the API key, never the key.
- `ReactLoop(max_iterations=50, compaction=...)` with guards for repeated
  calls, empty answers and the iteration limit; approvals and `ask_user`
  questions pause the turn (`waiting_for_user`) and `run(None, ctx)` resumes it.
- Compaction: `SummaryCompaction` (default; `SUMMARY_MODEL` on OpenRouter or
  the agent's model) and `SlidingWindowCompaction`. At compaction, facts the
  user stated are saved to the memory (`HARNESS_MEMORY_AT_COMPACTION`).
- Sandboxes: `LocalSandbox` (not isolated), `DockerSandbox` (one container per
  user, uid 1000, all capabilities dropped, read-only root, network only to an
  allowlist through a proxy) and `E2BSandbox` (`maxai[e2b]`).
- Skills: `LocalSkills` and `GitHubSkills` (GitHub API, token from
  `GITHUB_TOKEN`), installed into the sandbox and updated when they change.
- Memory: `Memory(read_only=False)`, one Markdown file per fact in the
  sandbox, an index in the prompt and the `save_memory` tool.
- Tools: `@tool` / `FunctionAsTool` (`read_only`, `deferred`), `BashTool`, file
  tools, MCP servers over stdio and HTTP shared by all runs.
- Prompt layers built by `NativePrompt` in a cache-friendly order;
  `CustomLayer` for your own text.
- Middleware with `turn`, `model`, `stream` and `tool` hooks;
  `LoggingMiddleware`, `BudgetMiddleware` (per-turn limits and per-user quotas
  in `LocalQuotaStore` / `MongoDBQuotaStore`) and `TracingMiddleware`
  (OpenTelemetry, Langfuse).
- Sessions: `LocalSessionStore` and `MongoDBSessionStore`; the host loads and
  saves.
- Serialization with no secrets: `serialize` / `deserialize`, `to_yaml` /
  `from_yaml`, `to_json` / `from_json`, and an allowlist for third-party
  classes (`allow_providers`, `MAXAI_ALLOWED_PROVIDERS`).
- Terminal UI (`run_cli`, `maxai[cli]`) and web UI (`run_web`).
- `docker-infra/`: MongoDB Atlas Local, an MCP server and Langfuse.

### Removed (compared with the pre-rewrite code)
- Ollama client, `Workspace` / `WorkspaceRegistry` (including Azure Blob and
  MinIO workspaces), `ToolExecutor` / `LocalExecutor` / `DockerExecutor` /
  `ModalExecutor`, `CapabilitiesManager`, routines, the old memory registries
  and the `gates=[...]` argument.

### Known gaps
- `AnthropicClient` (Messages API: streaming, thinking blocks, provider cache)
  is tested against a fake SDK only, not the real API.
- `knowledge=` is accepted and serialized but no tool queries it; it is kept
  to be connected.
- Completion gates are kept in the package to be connected; the loop does
  not use them yet.
- The examples were checked with a faked model only, not a real one.
