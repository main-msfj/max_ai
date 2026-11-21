# Bookkeeper

One agent that uses most of the framework, on a task anyone can follow: it
closes the month of a small café and hands in the report as a web page and as
an Excel workbook.

```sh
uv run python example-deploy/bookkeeper/agent.py
```

Then write: `close October for me. I always want the amounts in quetzales.`

| What you see | The piece behind it |
|---|---|
| A plan you accept or reject | todos with plan approval |
| The books copied and added up with a script | `BashTool` in a `LocalSandbox`, with permissions |
| "What is the transfer of Q4,000?" | `ask_user` |
| The exchange rate, with its source | an MCP server (Exa) and its `HTTPSecret` |
| `report.html` and `report.xlsx` in the workspace panel | two local skills: `html-report`, `excel-report` |
| Next month it still answers in quetzales | `Memory` |
| It knows the café and where its books are | a `CustomLayer` in the system prompt |
| `/resume` | `LocalSessionStore` |
| `agent.yaml` | the agent saved with no secrets |

It needs `OPENROUTER_API_KEY`; `EXA_API_KEY` for the web search and the
Langfuse variables for tracing are optional. Nothing else is installed: both
skills use only Python's standard library.

Everything of this agent is in this folder: `agent.py`, its `skills/`, its
`data/`, and what a run leaves (`box/`, the sandbox; `sessions/`).

`data/` holds the books: `sales.csv` and `expenses.csv`, with one transfer
left without a category and three amounts in dollars, on purpose. What a run
leaves behind (`box/`, `sessions/`, `agent.yaml`) is ignored by git.
