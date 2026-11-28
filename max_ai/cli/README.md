# MaxAI Textual CLI

The MaxAI terminal UI (lilac accent, diamond glyphs) for a configured `Agent`. The transcript is a
stack of widgets, not a text log, so every piece of a turn stays live:

- **Thinking** streams into a box, then folds to one line
  (`◈ Thought for 3s · 120 words`). Click it to expand/collapse.
  It's optional: `run_cli(agent, show_thinking=False)` starts hidden, and
  `ctrl+t` / `/thinking` toggles it at runtime (past thoughts are kept).
- **Tool calls** render as `◆ name(main argument)` with `╰─ summary` under
  it; a diamond spinner shows while running, then turns green/red when done.
  Click to expand the output.
- **Status line** while working: `◈ Thinking… (12s · ↓ 340 tokens · esc to
  interrupt)`. `esc` stops the turn; what it already did stays in the
  conversation, and the model is told it was interrupted.
- **Send while it works**: `enter` during a turn adds your message to it; the
  model reads it on its next call (after the running tool, or right after the
  answer it was writing). `/commands` and `!shell` wait for the turn to end.
- **Bash** calls read `◆ Bash  <description>`, then `$ command` and
  `╰─` with what it printed (click to expand).
- **Todos** (`write_todos`) is a list under `● Todos 2/5 · current item`;
  consecutive updates refresh the same box.
- **Questions** (`ask_user`): every pending question shows in ONE form —
  tabs across the top (`←→`), options for the current one (`↑↓` + `enter`,
  or click), or type your own answer in the prompt. With several questions
  a **Submit** tab reviews them and sends them together. It then folds to
  `? Answered 3 questions ▸ expand`.
- **Approvals** appear in a card above the prompt: click an option or type
  its number. "Always allow" keeps the rule in the user's `setting.json`
  (only with a sandbox, never for a dangerous command). Typing anything
  else is a new message: the call is closed without running.
- **One summary line per turn**: `◆ Done in 16s · 3 model calls`, or
  `■ Stopped after …` with the reason (max iterations, length, error).

| Input | Action |
|---|---|
| `enter` / `shift+enter`, `ctrl+j` | send / new line |
| `@relative/path` | attach a workspace file |
| `!command` | run a shell command in the workspace |
| `/` then `↑↓`, `tab`, `enter` | command menu: move, complete, run |
| `/help` `/skills` `/tools` `/mcp` `/tasks` `/resume` `/new` `/session` `/clear` `/thinking` `/verbose` `/workspace` `/theme` `/exit` | commands |
| `/<skill-name> [args]` | run a skill (every loaded skill is listed in the menu) |
| `esc` · `ctrl+t` · `ctrl+o` · `ctrl+b` · `ctrl+l` · `pgup/pgdn` | interrupt · thinking · verbose events · workspace panel · clear screen · scroll |

The CLI is a host like any other: it receives an already built `Agent`
and only adds where conversations are saved and whose they are.

```python
from max_ai.capabilities.session_store import LocalSessionStore
from max_ai.cli import run_cli          # pip install "maxai[cli]"

await run_cli(                              # closes the agent at the end
    agent,
    store=LocalSessionStore("./sessions"),  # saves after each batch of tools and turn
    user_id="user_001",
    session_id=None,                        # an id resumes that conversation
)
```

`/resume` picks a saved conversation, `/new` (or `/clear`) starts a new
session and `/session` shows the current one. A resumed conversation that
was waiting for an approval or an answer asks it again.

The same agent can be served in the browser with
`max_ai.web.run_web(agent, store=..., user_id=...)`.
