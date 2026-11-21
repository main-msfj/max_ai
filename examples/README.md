# Max AI examples

Small files, one idea each, written for the current API.

- [basic/](basic/README.md): one feature per file, in the order you would learn them.
- [advance/](advance/README.md): pieces of your own (a tool, a middleware, a session store, a prompt layer).
- [../example-deploy/](../example-deploy/README.md): whole agents, one per folder, run from the CLI.

Run any of them from the repository root:

```sh
uv run python examples/basic/01_basic_agent.py
```

Most need `OPENAI_API_KEY`; each file says in its first lines what else it
needs. What they write goes to `./local/`, which git ignores.

Every example follows the same shape:

```python
agent = Agent(name, description, instructions, client, ...)   # the pieces
ctx = RunContext(user_id="ana")                               # the conversation lives here
try:
    state = await agent.run("task", ctx)                      # one turn
    print(state.text)                                         # the answer
finally:
    await agent.close()                                       # MCP, sandboxes, client
```

Not here yet: knowledge search, which is not connected to the agent (see
"Status and known gaps" in the [main README](../README.md)).
