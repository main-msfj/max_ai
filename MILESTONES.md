# Milestones

What is still missing in `max_ai`, in the order it is meant to be done.
`[ ]` not started, `[>]` in progress, `[x]` done. The README's roadmap is the
short version of this file.

Last updated: 2026-11-28.

## 1. Gaps to close first

Pieces that exist in the code but do not work end to end yet.

- [ ] **Knowledge search.** `Agent(knowledge=...)` is accepted and saved, but
  the `search_knowledge` tool never reaches the registry and the prompt layer
  that names it renders empty. The model cannot query knowledge today.
- [ ] **Completion gates.** `capabilities/completion_gate` is kept, but no
  loop calls it and `Agent` has no `gates` argument. The gate should read
  `ctx.todos`; after that `AgentPlan` and `ctx.plan` can be deleted.
- [x] **Anthropic client against the real API.** Run on 2026-11-25 with
  `claude-haiku-5-5`: answer, stream, tools, structured output, thinking,
  cache, compaction and the bookkeeper. Found and fixed: thinking blocks
  sent back after a compaction were refused.
- [ ] **Save the session after each batch of tools.** Today a turn that dies
  in the middle loses what its tools already did.
- [ ] **After a compaction, put the working directory and the todos in the
  summary**, and empty the list of files already seen.
- [ ] **An MCP stdio server can still get a secret through `args`.** Keys go
  only through `secret=`; `args` is not checked.

- [>] **Check the prompt cache.** Measured on 2026-11-25 with one long
  conversation (22 turns, one tool, about 10 compactions forced by a small
  window). Share of the input read from the provider's cache:

  | Route | Read from cache |
  |---|---|
  | Claude Haiku, at Anthropic | 69% |
  | Claude Haiku, through OpenRouter | 0% |
  | GPT 6 Luna, through OpenRouter | 74% |
  | DeepSeek, through OpenRouter | 75% |

  Without a compaction a turn reads 77 to 83%. What is left to decide:

  - [x] **Claude through OpenRouter read nothing from the cache.** OpenRouter
    caches by itself for most providers, but Claude only when the request
    asks. No change in the client: `options={"cache_control": {"type":
    "ephemeral"}}` takes Haiku from 0% to 76% (2026-11-28). Documented in the
    README and set in the OpenRouter example and the bookkeeper.
  - [ ] **A compaction costs one rewrite of the whole conversation**, even
    when only one old tool result changed: the cache keeps marked points, not
    any prefix. Tried outside the repository with Haiku: a mark at the end of
    the system prompt gives 74%, and one more before the first tool result a
    compaction would cut gives 80% (36% less written to the cache). Open
    question: the client does not know which result will be cut; the
    compaction does.
  - [ ] Not measured yet: the order of the prompt layers and text that
    changes each turn (date, memory index, todos).

## 2. Examples

Checked on 2026-11-21 against a real model (`deepseek/deepseek-v4-flash`
through OpenRouter): 19 of 20 examples and the bookkeeper agent ran.

- [ ] **`examples/basic/12_skills.py` has never run.** GitHub's rate limit
  for requests without a token stops it. Needs a `GITHUB_TOKEN`, or a skill
  folder that is not behind that limit.
- [x] **`examples/basic/14_full_agent.py` stopped half way.** The model asks
  for `bash`, which needs approval, and the example does not answer: it ends
  in `waiting_for_user` and prints an unfinished sentence.
- [x] **`calculate_compound_interest` did not say its rate is a percentage.**
  The model sent 0.05 first in two examples and had to call again.
- [ ] **The bookkeeper in the real CLI**, by a person. So far it ran without
  the CLI, with every approval accepted by a script.
- [ ] **Open the bookkeeper's workbook in Excel itself.** The file is valid,
  but only a library has opened it.
- [ ] **The examples are tied to OpenAI.** Most use `gpt-5.4-nano`; decide if
  they should run on a cheaper or open model by default.

## 3. Documentation

- [ ] **The website still documents old names** (`maxai.*`) and the `env` and
  `headers` fields of the MCP configs, which no longer exist.
- [ ] **`REVIEW.md`**: how a change to this repository is reviewed.
- [x] **README:** the line that said the examples were checked only with a
  faked model is out of date.

## 4. Features

Planned, not started. Each one is designed before it is built.

- [ ] **Agent spawn and multi-agent.** An agent starts other agents and
  gathers their work, on the stateless `Agent`.
- [>] **Background tasks.** Built on 2026-11-28 and run in Local, Docker and
  E2B: `bash(background=true)`, `task_output`, `stop_task`, the time limit,
  three at once, the output kept, the model told when one ends, closing
  stops them, and the CLI shows them. Run by a person with the `runner`
  agent and DeepSeek. What is left:
  - [ ] **Send a command that already runs to the background.** Other
    harnesses do not restart it: every command is born with its own watcher
    and "in the foreground" only means the harness waits. Here it means
    running every bash as a task and removing its folder when it ends while
    the harness waits.
  - [ ] A task that ends while the agent sits idle is told only at the next
    model call.
  - [ ] A background command that changes a file the model had read does not
    say so when it ends; the next write still notices.
  - [ ] Subagents in the background: they need the model's key, which never
    enters the sandbox, so they cannot be a sandbox process.
  - [ ] MCP `cancel_task` is kept, but nothing calls it yet: an MCP task
    cannot be stopped by the model.
- [ ] **A2A.** Talk to agents of other frameworks over the Agent2Agent
  protocol, as a client and as a server.
- [ ] **Always-on agent.** An agent that stays up and wakes on a schedule or
  an event, not only on a user message.
- [ ] **Sandbox on first use.** A turn starts the user's sandbox before the
  model is called, even for a greeting. The open question is the memory
  index, which lives inside the sandbox. Measure the first message on Docker
  and E2B before building.
- [ ] **Memory "dream".** Tidy and merge the memory files between sessions.
- [ ] **Images from tools.** The image stays in the history; after a
  compaction it becomes "image omitted" and its path.
- [ ] **`ask_user`:** an unanswered question should not block the turn;
  questions with several answers.

## 5. CLI

- [ ] Information shown twice.
- [ ] A sidebar.
- [ ] Tool calls shown by kind.
- [ ] Cached tokens in the usage line.
- [ ] `bash` always asks: check the judge model setting.
- [ ] The rule for files of keys, seen working in the CLI.

## 6. Release

- [ ] **PyPI.** Waits for the website and then the pipelines.
- [ ] **Benchmarks**, after multi-agent.

## Later

- [ ] Native clients for Gemini and Azure OpenAI.
- [ ] Optimistic concurrency in session stores.
- [ ] Guardrail middleware for PII and prompt injection.
