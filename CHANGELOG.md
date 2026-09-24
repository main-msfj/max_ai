# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Until 1.0, minor versions may break
the API.

## [Unreleased] — 0.1.0

First public version.

### Added
- `Agent`: stateless; one instance serves many users concurrently (runs of
  different sessions overlap, messages of one session queue). Memory is bound
  per run to the `RunContext` user and session.
- Clients: OpenAI, OpenRouter and Ollama. Configs store the env var name of
  the API key, never the key. Output capped at 32K tokens; replies cut at the
  limit are explained to the model instead of silently dropped.
- Tools: function tools, bash (allow/ask/deny permissions), workspace file
  tools, plan, ask_user, and MCP servers (stdio and HTTP) shared by all runs.
  Read-only calls of one model reply run in parallel (`read_only=True`).
- Reasoning: `ReactLoop` with loop guards; completion gates (`gates=[...]`)
  decide when a turn may end; approvals and questions pause and resume a run.
- Memory, knowledge and skills: local JSON, SQLite and MongoDB backends;
  `CoreEmbedding` (`FastEmbedEmbedding`, `OpenAIEmbedding`) for search by
  meaning; SQLite backends store vectors next to their text; MongoDB uses
  `$vectorSearch`. Registries use `FastEmbedEmbedding` when `embedding` isn't
  passed; `embedding=None` keeps memory search by words only.
- Skills from Git: `GithubSkillRegistry(source, skills, ref=..., token_env=...)`
  shallow-clones a repo (public, or private with a token read from an env var)
  and copies the named skills. Edits the agent makes to a materialized skill
  are kept; changes at the source are picked up.
- Workspaces: `LocalWorkspace`, plus `AzureBlobWorkspace` and `MinIOWorkspace`
  that download a user's files before a run and upload only what changed.
- Executors: `LocalExecutor`, `DockerExecutor` (non-root, read-only, no
  network by default, resource limits; image in
  `max_ai/capabilities/executor/docker/Dockerfile`) and `ModalExecutor`.
- Context compaction: `SummaryCompaction` and `SlidingWindowCompaction`.
- Sessions: `CoreSessionStore` / `LocalSessionStore`; the host loads, runs and
  saves.
- Serialization: `agent.serialize()` / `Agent.deserialize()` with no secrets
  and an allowlist for third-party components (`allow_providers`).
- Structured output: `output_format` shapes only the accepted final answer.
- Middleware: hooks for the run, every model call and every tool call;
  `LoggingMiddleware`, `BudgetMiddleware` (per-task limits and per-user quotas
  in local or MongoDB stores) and `TracingMiddleware` (OpenTelemetry, Langfuse).
- Terminal UI: `run_cli(agent, store=..., user_id=..., session_id=...)` with
  `/resume`, approvals, questions, plans and a context-window bar.
- `docker-infra/`: MongoDB Atlas Local (vector search), MinIO, Azurite,
  Ollama, an MCP server and Langfuse.
- Examples: `examples/basic/` (one feature per file) and `examples/advance/`
  (custom backends, middleware, layers, embeddings, serialization).
