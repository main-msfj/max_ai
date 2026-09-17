# Max AI — Full Framework Evaluation (Fable)

**Reviewer:** Claude (Fable 5)
**Date:** 2026-07-06
**Branch:** `p0-embeddings-cleanup`
**Method:** Full read of the reasoning, compaction, skills, persistence, executor, CLI, and prompt-stack subsystems, plus empirical verification (I ran round-trip serialization tests, compaction injection tests, and the full pytest suite against the working tree).

Every claim below that says **[verified]** was reproduced by actually running code, not just reading it.

---

## TL;DR

The single-agent core is in genuinely good shape: clean provider-agnostic contracts, approval-as-state, events-first streaming, sandbox-by-default skills, and real test investment. Your instinct to solidify the single agent before going multi-agent was correct, and the stateless live→die→rehydrate architecture **is** the right base for multi-agent later.

But I found several **real bugs**, two of which undermine the exact pillars you asked about:

1. **`RunContext` cannot be rehydrated once `message_history` is populated** — the persistence story is broken for its main use case. **[verified]**
2. **Compaction stacks contradictory summary blocks in the prompt** every time it runs more than once. **[verified]**
3. **The human-input (elicitation) tool dies after 300 seconds** because it runs under the tool executor's timeout. A user who takes 6 minutes to answer silently breaks the turn.
4. **`ReActLoopPlanning` never registers the human-input tool** (it skips `_register_runtime_tools`), so elicitation is silently unavailable in the planning loop.
5. **Elicitation does not survive process death** — unlike approvals, which are durable state. This asymmetry is the single most important architectural fix before multi-agent.

Full pytest suite state at review time: see §9.

---

## 1. ReAct + Human-in-the-Loop (elicitation)

### What you built

`UserInputTool` ([max_ai/tools/structure_human_in_loop.py](max_ai/tools/structure_human_in_loop.py)) blocks on an `asyncio.Future` inside tool execution, sets `loop_state.finish_reason = "input_needed"`, and the loop resolves the pause in `_handle_input_needed` ([max_ai/base/reasoning.py:391](max_ai/base/reasoning.py#L391)). The agent exposes `provide_user_input` / `pending_user_input`, the web UI answers concurrently, and the CLI polls `agent.pending_user_input` every 50 ms ([max_ai/cli/repl.py:141-185](max_ai/cli/repl.py#L141-L185)).

"Elicitation" is indeed the right term (it's what MCP calls this), and having it as a **native tool** rather than a prompt convention is the right call. The `TaskAnalysisLayer` clarification rules are also well written.

### What's wrong

**1.1 — The tool runs under the executor timeout (BUG).**
`UserInputTool` is a normal `CoreTool`, so it executes through `LocalExecutor.run`, which wraps it in `asyncio.wait_for(task, timeout=300)` ([max_ai/executor/local.py:64-75](max_ai/executor/local.py#L64-L75)). If the user takes more than 5 minutes to answer:
- the future is cancelled and a `ToolResult.timeout` is folded into the transcript,
- `loop_state.finish_reason` is still `"input_needed"` (the tool set it before blocking),
- `_handle_input_needed` sees a dead-but-unresolved future and pauses the turn anyway,
- any later `provide_user_input` resolves a future nobody is awaiting.

Elicitation must be exempt from execution timeouts (or have its own, much longer, explicitly-configured one).

**1.2 — The event is emitted *after* the block, forcing the CLI to poll.**
The loop blocks on `await future` inside the tool *before* `UserInputRequestEvent` is ever yielded. Your own module docstring in [max_ai/cli/repl.py](max_ai/cli/repl.py) documents the resulting deadlock and the 50 ms polling workaround. This is the design smell to fix, not the CLI: the question should reach the consumer **through the event stream before anything blocks**. The `ToolApprovalEvent` flow already does this correctly — approval events are emitted by the executor *without executing anything*, and the turn ends. Elicitation should work the same way.

Also fragile: the CLI dedupes questions by question text (`answered_question`), so the agent asking the same question twice in one turn is skipped.

**1.3 — Elicitation is ephemeral; approvals are durable (ARCHITECTURE GAP).**
`pending_user_input` is an `asyncio.Future` with `exclude=True` ([max_ai/base/reasoning.py:112](max_ai/base/reasoning.py#L112)). If the process dies while a question is pending:
- the future evaporates,
- the tool-call record is stuck `EXECUTING` → becomes a stale execution → gets force-failed on resume.

Compare with approvals: `ToolCallRecord` has a real state machine (`PENDING_APPROVAL → APPROVED/REJECTED → EXECUTING → CONSUMED`), it serializes, and `resume()` picks it up cleanly. Your approval design is *excellent* — and elicitation should be the same mechanism: a record state (e.g. `INPUT_NEEDED`), a `UserInputRequestEvent` emitted before execution, turn ends with `finish_reason="input_needed"`, answer applied via `ctx.tool_state` (analogous to `apply_approval`), then `agent.resume()`. That kills the polling hack, survives restarts, works across processes, and — critically — composes for multi-agent (a child agent's question can bubble up through a parent as data, not as a live future).

Keep the in-process future as an optional fast path for the web UI if you want same-turn continuation, but the durable path must exist.

**1.4 — Cross-layer coupling.** The tool mutates `loop_state.finish_reason` directly. That works, but it means any tool can hijack loop control flow via shared mutable state. With the record-state design above, this coupling disappears.

**1.5 — Parallel-batch edge case.** If the model emits `structure_human_in_loop` *plus* an approval-gated tool in the same batch, the approval branch returns first and the pending-input state dangles into the next segment. Low probability, but real.

---

## 2. Planning as a native, optional capability

### What you built

Two models, both syncing to the same `ctx.plan` / `PlanningEvent`:
- **Planning-as-controller** (`ReActLoopPlanning`, [max_ai/reasoning/react_planning.py](max_ai/reasoning/react_planning.py)): always plans (structured `AgentPlan` output), injects "you are working on step N" each iteration, marks steps done/failed from tool outcomes, replans after `max_step_retries` failures, optional self-eval with retry budget.
- **Planning-as-tool** (`ReActLoopSelfDirected`, [max_ai/reasoning/react_self_directed.py](max_ai/reasoning/react_self_directed.py)): the model calls `update_plan` when *it* decides to plan, with a structural nudge when the plan goes stale.

### Assessment

The **self-directed loop is the right default direction** — it exactly matches your goal ("native but not mandatory, the LLM decides"). It's the same pattern as Claude Code's TodoWrite. The controller loop is a legitimate alternative for small models (see §3). Keep both, but be honest about their roles: self-directed is the flagship, controller is the small-model harness.

### Problems

**2.1 — Three near-identical loop bodies.** `react_simple.py`, `react_planning.py`, and `react_self_directed.py` share ~80% of their `execute_reasoning_loop` code (pending-drain, LLM call, approval batching, scratchpad relay, input_needed handling). They are **already diverging by accident**:
- `ReActLoopPlanning.__init__` doesn't accept `enable_human_input` and its loop never calls `_register_runtime_tools` → **no elicitation in the planning loop**, even though it dutifully checks `finish_reason == "input_needed"` (dead code, the tool never exists there).
- The self-directed loop had to override the whole method just to add two small hooks (`_register_runtime_tools`, `_sync_plan`).

The fix is a template-method loop: one `execute_reasoning_loop` in a shared base with narrow hooks (`before_llm_call`, `after_tool_round`, `on_final_answer`). Each strategy becomes ~50 lines instead of ~300 copied ones.

**2.2 — Step-advancement heuristic is crude.** "All tools this round succeeded → active step is done" ([react_planning.py:376-383](max_ai/reasoning/react_planning.py#L376-L383)) conflates *one tool round* with *one plan step*. A step that needs three tool calls across three iterations gets marked done after the first. Options: let the model confirm step completion (cheap: include "current step" in the plan-progress message and ask it to call a `complete_step` tool / or reuse `update_plan`), or track step-scoped tool expectations via `tool_hint`.

**2.3 — Transcript pollution.** The controller injects a `SystemMessage("plan-progress")` **every iteration**, and the self-directed nudge appends another `SystemMessage` every non-updating round. These accumulate in `ctx.messages`, get persisted, get counted by compaction, and confuse small models with repeated near-identical system messages. Inject transient guidance at **prompt-assembly time** (a rendered-layer addition for the current call only) rather than appending durable messages, or at minimum dedupe/replace the previous nudge.

**2.4 — `update_plan` schema drops fields.** `AgentPlan.PlanStep` has `tool_hint` and `depends_on` ([max_ai/reasoning/plan.py](max_ai/reasoning/plan.py)), but the tool schema ([max_ai/tools/update_plan.py:49-82](max_ai/tools/update_plan.py#L49-L82)) only exposes `id/description/status`. Either expose them or delete them from the model — dead schema fields rot.

**2.5 — Plan validation is missing.** `update_plan` accepts duplicate step ids, multiple `active` steps, and dangling `depends_on`. Cheap Pydantic validators on `AgentPlan` would return a useful tool error the model can self-correct from — which is exactly the harness-guides-the-model behavior you want (§3).

---

## 3. Harness-managed loop for small models

You asked: how to move from "completely LLM-centric" to "the framework manages the LLM, and the LLM decides the next step" — especially for small models. You already have the seeds: the plan nudge, the intermediate-eval "reconsider your approach" injection, the `TaskAnalysisLayer`. What's missing is a **single, explicit place** where per-iteration steering lives. Right now every nudge is hand-rolled inside a loop body.

Recommendation: introduce a **loop-guard / steering stage** in the base loop (runs between "tool results collected" and "next LLM call") with pluggable checks. Concrete guards worth building, in order of payoff for small models:

1. **Malformed-call recovery.** When a tool call fails schema validation, don't just return the raw error — inject a compact retry prompt echoing the expected schema and the diff of what was wrong. Small models fix schemas well when shown the exact discrepancy.
2. **Repetition/loop detection.** Hash `(tool_name, parameters)` per turn; on the 2nd-3rd identical call, inject "you already called X with these arguments and got the same result — change strategy or finish." This is the #1 small-model failure mode.
3. **Budget awareness.** At `iteration >= 0.75 * max_loop_iterations`, inject "N iterations left; consolidate and answer." Prevents the `max_iterations_exceeded` cliff where the user gets nothing.
4. **No-progress detection.** If the assistant emits neither text nor tool calls (or empty content repeatedly), re-prompt with an explicit menu: "Either call one of [tools] or answer the user."
5. **Forced tool choice / constrained decoding** where the provider supports it (`tool_choice="required"`-style), used by the controller loop when a plan step has a `tool_hint`. This is the strongest lever for small models and it's provider-side, not prompt-side.
6. **Step-scoped tool filtering.** In the controller loop, only expose the tools relevant to the active step. Fewer tools → dramatically better tool selection on 4-8B models.

Each guard is a pure function `(ctx, loop_state) → optional steering message / config tweak`, so they're unit-testable and composable. This gives you the "harness manages the workflow, model decides within it" split you're describing, without a rigid state machine.

---

## 4. Compaction

### What's good

The architecture is more complete than most frameworks: atomic grouping so assistant+tool pairs never split ([base/compaction.py:142-187](max_ai/base/compaction.py#L142-L187)), previous-summary merging, a **structured** summary (`CompactionOutput` with objectives/pending/decisions), persistence of the summary to the logbook, and a memory-maintenance pass that upserts durable facts during compaction ([base/agent.py:566-633](max_ai/base/agent.py#L566-L633)). That last loop-closure is genuinely rare and good. Mid-loop compaction with a 0.8 trigger / 0.2 keep ratio is a sane hysteresis design.

### What's wrong

**4.1 — Summary blocks accumulate (BUG). [verified]**
`_inject_summary` appends: `prompts.rendered_layers[ContextLayer] = f"{existing}\n\n{block}"` ([compaction/sliding_window.py:135-151](max_ai/compaction/sliding_window.py#L135-L151)). I verified that two injections produce two `<COMPACTION_SUMMARY>` blocks. In any run where compaction fires more than once (pre-loop inject + one mid-loop compaction is enough), the model sees **multiple, contradictory summaries**, the oldest first, and the prompt grows without bound — the exact opposite of compaction. Fix: strip any existing `<COMPACTION_SUMMARY>` block before appending (or keep the base rendered layer and always compose `base + latest block`).

**4.2 — Ownership of `ctx.messages` is ambiguous.** `SlidingWindowCompaction.compact` mutates `ctx.messages` itself; `_run_mid_loop_compaction` *also* assigns `ctx.messages[:] = result.recent_messages`; `Agent._apply_compaction` relies on the strategy's internal mutation and applies nothing. Pick one contract — I'd make strategies **pure** (return `CompactionResult`, never touch `ctx`) and have the single caller apply it. Right now a custom `CoreCompaction` author cannot know what they're responsible for.

**4.3 — The budget math ignores the real prompt.** `live_message_capacity_tokens` subtracts a **fixed** `compaction_prompt_budget_tokens = 6000` ([config.py:44](max_ai/config.py#L44)) instead of the actual rendered prompt size — which the agent knows precisely (`self._prompt_tokens`, and it changes when memory/knowledge layers grow or the summary is injected). An agent with a 9k-token prompt on a 15k window will blow the window while compaction thinks it's fine. Pass the real `prompts.prompt_tokens` into the capacity calculation, and update `prompt_tokens` after summary injection.

**4.4 — `message_history` is invisible to compaction.** `_should_compact` counts only `ctx.messages`, but the model input is `message_history + messages` ([base/reasoning.py:326-332](max_ai/base/reasoning.py#L326-L332)). Anyone using `ChatHistory` for multi-turn sessions gets zero compaction coverage on it. Either include it in the count + compaction scope, or document loudly that history must be pre-budgeted by the caller.

**4.5 — The summarization call itself can overflow.** `_summarize_old_messages` sends the entire old-message transcript in one LLM call with no chunking and no per-message truncation ([sliding_window.py:159-192](max_ai/compaction/sliding_window.py#L159-L192)). A 100k-token tool output that just got evicted lands whole in the summary request — against the *same* client with the *same* context window. Truncate each message (tool outputs especially) to a per-message cap, and chunk the transcript if it exceeds the window.

**4.6 — Minor:** the mid-loop `CompactionEvent` hardcodes `live_message_threshold_tokens=0 / live_message_budget_tokens=0` ([base/reasoning.py:287-288](max_ai/base/reasoning.py#L287-L288)) while the pre-loop event computes them — inconsistent telemetry. And `_should_compact` constructs a fresh `TokenCounter` (tiktoken encoder lookup) every iteration; cache it on the loop.

### Are the summaries "generated correctly"?

The prompt design (previous summary + newer messages, newer wins, structured output) is correct. The two real correctness risks are 4.1 (stacked stale summaries poison continuity) and 4.5 (overflow → failed or truncated summary silently degrading to `CompactionOutput()` empty). Fix those two and I'd call the mechanism sound.

---

## 5. Skills

### Verdict: the approach is good — it's the Anthropic-style skills model

`SKILL.md` + YAML frontmatter, only name+description in the prompt (`SkillsLayer.j2`), full instructions loaded on demand via `read_skill` inside the sandbox, scripts executed via `bash` in Docker with `--network none` and only `/mnt` mounted, and a hard refusal to run skills without a sandbox (`_validate_runtime_safety` → `unsafe_local_executor_for_skills`). That's progressive disclosure + sandbox-by-default, which is the right threat model and the right context economy. Your question "should it behave more like a sandbox?" — it already **is** one; the direction is correct.

### Improvements, in order of importance

**5.1 — Harden the container.** You have `--network none` (good), but no `--memory`, `--cpus`, `--pids-limit`, `--cap-drop ALL`, `--security-opt no-new-privileges`, no read-only rootfs, and the container runs as root ([executor/docker/docker.py:399-413](max_ai/executor/docker/docker.py#L399-L413)). A fork bomb or memory balloon in a skill script takes the host with it today. All five flags are one-line additions.

**5.2 — Network policy will become a product question.** `--network none` is the safest default, but many real skills need controlled egress (pip install, API fetch). Plan for an opt-in per-skill network mode declared in frontmatter (`network: none|allowlist`) rather than flipping the global default later under pressure.

**5.3 — Session bleed.** The bash container is keyed per user, not per session — two concurrent sessions of the same user share shell state and `/mnt/artifacts`. Fine for now; make the key `(user_id, session_id)` when concurrency matters.

**5.4 — Frontmatter is unvalidated.** `_parse_frontmatter` returns a raw dict and only `name`/`description` are read ([base/skills.py:234-280](max_ai/base/skills.py#L234-L280)). Define a `SkillManifest` Pydantic model (name, description, version, and later: required packages, network policy, allowed tools). Validating at `prepare()` gives skill authors errors at load time, not mid-run.

**5.5 — Static selection only.** The registry takes a fixed `skills=[...]` list. There's no "library of 200 skills, agent discovers relevant ones" story. The metadata-only prompt footprint means you could scale the catalog cheaply — worth a `search_skills` tool later, same pattern as your routines registry.

**5.6 — No integrity story.** Fine for `LocalSkillRegistry`; the moment you add git/HTTP sources, you want a content hash pinned at selection time.

**5.7 — Nit:** `BashTool` is `AUTO_APPROVED` when skills are present ([manager/capabilities.py:147-151](max_ai/manager/capabilities.py#L147-L151)). Defensible *because* of the sandbox, but it should be a conscious, documented default, and it's another reason 5.1 matters.

---

## 6. Agent serialization / stateless rehydration

### Is live→die→rehydrate the right architecture? **Yes.**

Separating **agent definition** (component config, `_to_config`/`_from_config`) from **run state** (`RunContext` in a `RunContextStore`) is exactly how durable agent runtimes are built. Approval-as-state + `resume()` + stale-execution policy is the strongest part of the framework. And yes — it's the correct foundation for multi-agent: a child agent becomes "a `RunContext` + store key", a supervisor holds child run ids, and pauses (approvals, questions) bubble up as data. Do **not** switch to long-lived in-memory agent objects.

But the implementation has holes:

**6.1 — Rehydration is broken when `message_history` is non-empty (BUG). [verified]**
`ChatHistory`'s `mode="before"` validator rejects raw dicts ([types/chat_history.py:19-30](max_ai/types/chat_history.py#L19-L30)), and `model_validate_json` presents dicts to it. So:

```python
RunContext.model_validate_json(ctx.model_dump_json())
# → ChatHistoryError: message_history[0] is a raw dict. Use ChatHistory.load_from()
```

`FileSystemRunContextStore.load()` therefore **raises for any run that used history**. Your own store contract says load "never raises on miss" — this raises on *hit*. The validator should parse dicts through the discriminated union (`CoreMessage.parse_msg`) instead of rejecting them. Separately, `ChatHistoryBlock.message_history` is typed `list[CoreMessage]` (the base class) rather than the discriminated `Message` union — even without the validator, subclass identity (`tool_calls`, `tool_call_id`) would be lost on load. Use `list[Message]`.

**6.2 — `structured_output` silently degrades on rehydrate (BUG). [verified]**
`AssistantMessage.structured_output: BaseModel | None` deserializes into an **empty** `pydantic.BaseModel` — all fields gone, no error. Store it as `dict + type-ref` (you already have `_type_ref`/`_load_type_ref` machinery in [base/agent.py:318-334](max_ai/base/agent.py#L318-L334)), or exclude it from serialization explicitly.

**6.3 — Nothing actually checkpoints.** `persistence/` is a clean contract with a good filesystem impl — and the `Agent` never calls it. Crash mid-run = lose everything since the last manual save by the caller. Wire an optional `store: RunContextStore` into `Agent`: checkpoint after each tool round and on every pause (approval / input_needed / error). That's what makes "lives, dies, is rehydrated" true rather than aspirational.

**6.4 — `loop_state` is lost across pause/resume.** Iteration count, token usage, eval attempts live in `BaseLoopState`, which is created fresh every `run_stream_events` call. A run that pauses for approval 3 times gets `3 × max_loop_iterations` total budget, and usage reporting resets per segment. Persist a compact snapshot of loop metrics into `ctx.runtime_state` on pause; restore on resume.

**6.5 — Concurrency is the real blocker for multi-agent.** Two acknowledged-in-code issues become acute the day a supervisor fans out to N children:
- `reasoning.bind()` mutates the shared loop instance every run ([base/reasoning.py:219-245](max_ai/base/reasoning.py#L219-L245)) — two concurrent `agent.run()` calls on one agent corrupt each other's wiring.
- `agent._active_reasoning` is a single slot ([base/agent.py:949](max_ai/base/agent.py#L949)) — one pending question per agent instance, period.

Fix by making runs self-contained: clone the reasoning loop per run (or pass runtime deps as call arguments instead of `bind`), and key pending-input by `run_id`, not by agent.

**6.6 — `AgentAsTool` can't propagate pauses.** If a child agent ends with `approval_needed` or `input_needed`, `AgentAsTool._build_tool_result` just extracts last-message text and returns "success" ([tools/agent_as_tool.py:176-190](max_ai/tools/agent_as_tool.py#L176-L190)) — the child's pending approval is silently dropped and its context discarded. For your multi-agent future, this is the key composition test: **a child's pause must surface as the parent's pause**, carrying the child's `run_id` so the child can be resumed after the human answers. The durable-elicitation design from §1.3 is what makes this possible.

---

## 7. Cross-cutting observations

- **Prompt-template language errors are user-facing.** `AgentPolicyLayer.j2` contains broken directives — "user routine when user explicitly references a routine", "user skills when…" (presumably "use") — inside the system prompt of every agent. Small models take these literally. Also typos across the codebase: `Rolt Directory` ([config.py:30](max_ai/config.py#L30)), `.xlsl` (should be `.xlsx`, [config.py:40](max_ai/config.py#L40)), `Respresents`, `criteriont`, `Evalidate`, `strucure`. Cosmetic in code, harmful in templates.
- **Docstring drift:** [base/agent.py:370](max_ai/base/agent.py#L370) documents a `read_skill_resource` tool that doesn't exist (a test even asserts its absence).
- **`Agent` is declared `ABC` but has no abstract methods** and is instantiated directly everywhere. Drop the `ABC` or make the intent real.
- **Repo hygiene:** `user123/`, `user1234/`, `user_001/`, `var/`, `llm-models/` committed at the repo root; `.venv` present. Add to `.gitignore` before anyone external reads the repo.
- **Provider breadth:** Ollama + OpenAI clients exist. When you add an Anthropic client, the layered-prompt design maps cleanly onto system-prompt blocks, and prompt caching would directly reward your stable layer ordering.
- **What's genuinely strong** (keep and protect): the tool-call state machine, events-first streaming with exactly-one terminal response, prompt layers validated at construction, RoutingExecutor's trusted/untrusted split, compaction→memory-maintenance closure, and the test discipline.

---

## 8. Answers to your specific questions, in one line each

| Question | Answer |
|---|---|
| ReAct + HITL — how to improve? | Make elicitation durable state like approvals (record state + event before blocking); exempt it from tool timeouts; kill the CLI polling hack. |
| Planning native but optional — good? | Yes; self-directed (`update_plan`) is the right default; controller loop should become the small-model harness; deduplicate the three loop bodies first. |
| Harness-managed cycle for small models? | Add a pluggable loop-guard stage: schema-retry, repetition detection, budget warnings, forced tool choice, step-scoped tool filtering. |
| Compaction — does it make sense? | Design yes; two real bugs (stacked summary blocks, fixed prompt budget) and two gaps (history not counted, summarize call can overflow). |
| Skills — good or bad? Sandbox? | Good — it *is* a sandbox with the right disclosure model; harden the container (mem/cpu/caps/non-root) and validate frontmatter. |
| Serialization — is stateless right? Multi-agent ready? | Architecture right; implementation not yet: history rehydration is broken, nothing checkpoints automatically, `bind()`/`_active_reasoning` aren't concurrency-safe, and `AgentAsTool` drops child pauses. |

---

## 9. Test suite status

`pytest -q` at review time: **435 passed, 1 failed** (3m31s).

The failure is `tests/test_core_agent_smoke.py::test_core_agent_prepares_end_to_end`: the seeded fact "Software engineer in Buenos Aires" no longer appears in the rendered `MemoryLayer` — the layer renders its scaffolding but not the seeded memory. This looks like fallout from the embeddings/memory work on this branch (`p0-embeddings-cleanup`): the memory snapshot the prompt-variables builder collects is coming back empty. Worth fixing before merge since it means **memory is silently absent from the system prompt** in at least this path.

The other bugs documented above are not covered by the suite — each fix in the improvement plan lists the regression test to add alongside it.
