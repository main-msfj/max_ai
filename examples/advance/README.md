# Advanced Examples

These examples show how to implement framework base interfaces. Run each file
from the repository root, for example:

```sh
uv run python -m examples.advance.07_custom_embedding
```

## Status

Match the current API (checked by reading the code, not by running):

- `07_custom_embedding.py`: a `CoreEmbedding` subclass.
- `09_serialize_agent.py`: saves the agent configuration to `local/agent.json`.
- `10_deserialize_agent.py`: rebuilds it. The API key is not stored: set
  `OPENAI_API_KEY` before rebuilding.

Outdated (written before the November 2026 rewrite):

| File | Problem |
|---|---|
| `02_postgres_memory.py` | imports `max_ai.base.memory` (removed; memory is now `Memory`, files in the sandbox) |
| `04_custom_middleware.py` | imports `MiddlewareContext` (removed; override `turn` / `model` / `stream` / `tool`) |
| `08_custom_workspace.py` | imports `max_ai.base.workspace` (removed; implement `SandboxCore` instead) |
| `01_native_tool.py`, `03_postgres_knowledge.py`, `05_in_memory_session_store.py`, `06_custom_prompt_layer.py` | imports are valid, but they use the old run API (`async with agent`, `run_context=`, `response.final_text`) |

`03_postgres_knowledge.py` also depends on `knowledge=`, which no tool queries
yet. The PostgreSQL examples need `pip install "psycopg[binary]"` and a
`DATABASE_URL` variable.

See [../README.md](../README.md) for the old-to-new API table.
