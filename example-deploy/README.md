# Agents to run

Each folder is one whole agent: its code, its skills, its data, and what it
leaves behind when it runs (`box/`, its sandbox, and `sessions/`).

| Agent | What it does | Run it |
|---|---|---|
| [bookkeeper/](bookkeeper/README.md) | closes the month of a small business: a web page and an Excel workbook | `uv run python example-deploy/bookkeeper/agent.py` |
