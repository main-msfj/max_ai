# MongoDB knowledge

A named knowledge source stored in MongoDB, with indexed text search. The
agent can only search; loading documents is an application API.

```sh
uv sync --extra mongodb
export MONGODB_URI='mongodb://localhost:27017'
```

```python
from max_ai.capabilities.knowledge.mongodb import MongoDBKnowledgeRegistry
from max_ai.core import KnowledgeBlock

knowledge = MongoDBKnowledgeRegistry(
    name="manual", description="Search words or phrases in the project manual",
    database="max_ai",
)
agent = Agent(..., knowledge=[knowledge])

async with knowledge:
    await knowledge.upsert_block("guide/intro", KnowledgeBlock(
        content="The project stores session memories in MongoDB.",
        metadata={"source": "guide.md", "section": "intro"},
    ))
    blocks = await knowledge.search("MongoDB")
    await knowledge.delete_block("guide/intro")
```

Supports `async with`, lazy connection on first use, explicit `connect()` /
`disconnect()`, and component serialization. The config stores the env var
name (`uri_env`, default `MONGODB_URI`), never the URI or credentials. Also
configurable: `collection` (default `knowledge`) and
`server_selection_timeout_ms` (default 5000).

## Storage and search

- One document per `(source, block_id)`; `source` is the registry `name`.
  Knowledge is shared by everyone configured with that source, not per user:
  use distinct names or databases for separate domains.
- Use stable block ids when reloading documents. Replacing a block updates its
  metadata too; replace and delete update the text index immediately.
- `search` returns `KnowledgeBlock`s (content, tokens, metadata), 1–100
  results (default 5).
- Native `$text` index with language `none`: word and phrase retrieval, **not**
  semantic/vector or substring search. Quoted phrases and exclusions follow
  MongoDB's `$search` syntax.
- Indexes are created on connect; the account needs permission for that. Use
  a dedicated collection.

## Tests

Unit checks (no server): `tests/capabilities/test_mongodb_registries.py`.
Against a real server: see `tests/integration/test_mongodb_backends_live.py`
(`MAXAI_TEST_MONGODB_URI=...`).
