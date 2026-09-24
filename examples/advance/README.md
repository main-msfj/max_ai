# Advanced Examples

These examples show how to implement framework base interfaces. Run each file
from the repository root, for example:

```sh
uv run python -m examples.advance.01_native_tool
```

`09_serialize_agent` saves the agent configuration to `local/agent.json`;
`10_deserialize_agent` reads it and rebuilds the agent. The OpenAI key is not
stored in the JSON: set `OPENAI_API_KEY` before rebuilding the agent.

The PostgreSQL examples require `pip install "psycopg[binary]"` and a
`DATABASE_URL` variable, for example:

```sh
export DATABASE_URL="postgresql://user:password@localhost:5432/examples"
```

The example tables are created on startup. Knowledge search uses text matching
to keep the backend interface easy to follow.
