# Milestones

What is still missing in `max_ai`, in the order it is meant to be done.
`[ ]` not started, `[>]` in progress, `[x]` done. The README's roadmap is the
short version of this file.

Last updated: 2026-11-21.

## 1. Gaps to close first

Pieces that exist in the code but do not work end to end yet.

- [ ] **Knowledge search.** `Agent(knowledge=...)` is accepted and saved, but
  the `search_knowledge` tool never reaches the registry and the prompt layer
  that names it renders empty. The model cannot query knowledge today.
- [ ] **Completion gates.** `capabilities/completion_gate` is kept, but no
  loop calls it and `Agent` has no `gates` argument. The gate should read
  `ctx.todos`; after that `AgentPlan` and `ctx.plan` can be deleted.
- [ ] **Anthropic client against the real API.** Only run against a fake SDK
  so far. Waits for a key.
- [ ] **Save the session after each batch of tools.** Today a turn that dies
  in the middle loses what its tools already did.
- [ ] **After a compaction, put the working directory and the todos in the
  summary**, and empty the list of files already seen.
- [ ] **An MCP stdio server can still get a secret through `args`.** Keys go
  only through `secret=`; `args` is not checked.

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
- [ ] **Background tasks.** Work that keeps running while the conversation
  goes on, and reports when it ends. MCP `cancel_task` is kept for this.
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
