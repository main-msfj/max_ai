# MongoDB memory

Session memory stored in MongoDB, with indexed text search across the user's
other sessions.

```sh
uv sync --extra mongodb
export MONGODB_URI='mongodb://localhost:27017'
```

```python
from max_ai.capabilities.memory.mongodb import MongoDBMemoryRegistry

memory = MongoDBMemoryRegistry(
    user_id="alice", session_id="conversation_1", database="max_ai",
    context_days=30, search_limit=20,
)
agent = Agent(..., memory=memory)
```

Supports `async with`, lazy connection on first use, explicit `connect()` /
`disconnect()`, and component serialization. The config stores the env var
name (`uri_env`, default `MONGODB_URI`), never the URI or credentials. Also
configurable: `collection` (default `memory`) and `server_selection_timeout_ms`
(default 5000).

## Storage and search

- One document per `(user_id, session_id, category)`, enforced by a unique
  index. Updating replaces the category content atomically (UTC timestamp
  included), also under a concurrent first insert. Categories are
  case-sensitive. `context_days` filters what is loaded; it deletes nothing.
- Reads and deletes stay within the current user/session. Search covers only
  the same user's **other** sessions, ranked by text relevance then recency,
  capped at `search_limit` (also sessions older than `context_days`).
- Native `$text` index with language `none` (no stemming or stop words): word
  and phrase retrieval, **not** semantic/vector or substring search. No Atlas
  or embeddings needed.
- Indexes are created on connect; the account needs permission for that. Use
  a dedicated collection.

## Tests

Unit checks (no server): `tests/capabilities/test_mongodb_registries.py`.
Against a real server (creates and drops a `maxai_test_*` database):

```sh
MAXAI_TEST_MONGODB_URI='mongodb://localhost:27017' \
  uv run --extra mongodb pytest -q tests/integration/test_mongodb_backends_live.py
```
