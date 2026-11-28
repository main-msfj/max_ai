# Runner

One agent to see background tasks working: it starts a slow test suite, keeps
helping while it runs, and reports when it ends.

```sh
uv run python example-deploy/runner/agent.py
```

It needs `OPENROUTER_API_KEY` (the model is `deepseek/deepseek-v4-flash`);
`EXA_API_KEY` for the web search is optional.

| Write this | What you see |
|---|---|
| `run the tests, and meanwhile search what is new in pytest` | bash asks once, with `in background` in its title; `1 task running` below; the web search goes on; about a minute later `◇ Background task 1: done, exit code 1.` and the agent names the test that failed |
| `/tasks` | the tasks of this conversation: number, state, when it started, command |
| `how are the tests going?` | `Task output` with the last lines printed so far |
| `run the tests slowly, 20 seconds each` and then `stop them` | `Stop task`, and the task says `stopped` |
| `run the tests slowly`, then `/exit`; start again and `/resume` | closing stopped the task; at your next message the agent is told it was stopped |

The pieces behind it: `BashTool` with `background=true`, the `task_output`
and `stop_task` tools that come with it, a local skill (`slow-tests`, a
made-up suite of twelve tests where one fails), an MCP server (Exa) and a
`LocalSandbox`.

What a run leaves behind (`box/`, `sessions/`) is ignored by git.
