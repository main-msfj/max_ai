# MaxAI Textual CLI

The MaxAI terminal UI (lilac accent, diamond glyphs) for a configured `Agent`. The transcript is a
stack of widgets, not a text log, so every piece of a turn stays live:

- **Thinking** streams into a box, then folds to one line
  (`◈ Thought for 3s · 120 words`). Click it to expand/collapse.
  It's optional: `run_repl(agent, show_thinking=False)` starts hidden, and
  `ctrl+t` / `/thinking` toggles it at runtime (past thoughts are kept).
- **Tool calls** render as `◆ name(main argument)` with `╰─ summary` under
  it; a diamond spinner shows while running, then turns green/red when done.
  Click to expand the output.
- **Status line** while working: `◈ Thinking… (12s · ↓ 340 tokens · esc to
  interrupt)`. `esc` cancels the turn and rolls the conversation back to
  before it (side effects that already ran are not undone).
- **Plan** (`update_plan`) is a fold-able box: it folds to
  `◇ Plan 2/5 · current step`; consecutive updates refresh the same box.
- **Questions** (`ask_user`): every pending question shows in ONE form —
  tabs across the top (`←→`), options for the current one (`↑↓` + `enter`,
  or click), or type your own answer in the prompt. With several questions
  a **Submit** tab reviews them and sends them together. It then folds to
  `? Answered 3 questions ▸ expand`.
- **Approvals** appear in a card above the prompt: click an option or type
  its number.
- **One summary line per turn**: `✓ Done in 16s · stop`, or `⚠ …` when the
  completion gate retried or closed with notes, or `■ Stopped after …`
  (max_iterations, waiting, error) with the gate's reasons.

| Input | Action |
|---|---|
| `enter` / `shift+enter`, `ctrl+j` | send / new line |
| `@relative/path` | attach a workspace file |
| `!command` | run a shell command in the workspace |
| `/` then `↑↓`, `tab`, `enter` | command menu: move, complete, run |
| `/help` `/skills` `/tools` `/clear` `/thinking` `/verbose` `/files` `/exit` | commands |
| `/<skill-name> [args]` | run a skill (every loaded skill is listed in the menu) |
| `esc` · `ctrl+t` · `ctrl+o` · `ctrl+b` · `ctrl+l` · `pgup/pgdn` | interrupt · thinking · verbose events · files · clear screen · scroll |

```python
from max_ai.cli import run_repl

await run_repl(agent, show_thinking=True, initial_context=RunContext(user_id=..., session_id=...))
```

`/clear` starts a fresh conversation but keeps `user_id`/`session_id`, so
session-bound memory keeps working.
