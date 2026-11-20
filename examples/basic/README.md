# Basic Examples

Each file demonstrates one feature. From the repository root:

```sh
uv run python -m examples.basic.01_basic_agent
```

## Status

**None of the basic examples run against the current API.** They have not
been ported since the November 2026 rewrite. See [../README.md](../README.md)
for what changed and the main [README](../../README.md) for working code.

| File | Problem |
|---|---|
| `02_ollama_agent.py` | imports `max_ai.capabilities.clients.ollama` (removed) |
| `06_memory.py` | imports `max_ai.capabilities.memory.to_delete.local` (removed) |
| `10_skills.py` | imports `GithubSkillRegistry` (now `GitHubSkills`) |
| `13_workspace_files.py` | imports `max_ai.capabilities.workspace.local` (removed) |
| `15_full_agent.py` | imports `max_ai.base.memory`, `max_ai.capabilities.compaction.to_delete`, `max_ai.capabilities.memory.to_delete.local` (removed) and `GithubSkillRegistry` |
| `16_local_sandbox.py` | imports are valid, but uses `sandbox.root` (now `sandbox.layout.root`) and expects files without calling `await sandbox.prepare()` |
| `01`, `03`, `04`, `05`, `07`, `08`, `09`, `11`, `12`, `14` | imports are valid, but they use the old run API: `async with agent`, `run_context=`, `response.final_text`, `response.final_message`; `04` also uses `response.pending_approvals` and `agent.resume()` |

`07_knowledge_search.py` and the knowledge part of `15_full_agent.py` would
not work even once ported: `knowledge=` is not wired to any tool yet (see
"Status and known gaps" in the main README).
