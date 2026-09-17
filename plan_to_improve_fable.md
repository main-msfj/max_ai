# Max AI — Improvement Plan (Fable)

**Companion to:** [feedback_fable.md](feedback_fable.md)
**Date:** 2026-07-06
**Ordering:** least painful → most painful. Each phase is independently shippable; later phases assume earlier ones (especially Phase 3 → Phase 5/9). "Pain" = code churn + design risk, not just hours.

---

## 📋 Execution status (updated 2026-07-06)

| Phase | Status | Notes |
|---|---|---|
| 0 — One-line fixes | ✅ DONE | Smoke test fixed (time-bomb memory date), template/typo sweep, `.gitignore`, `ABC` dropped, `integration` mark registered. |
| 1 — Serialization | ✅ DONE | `ChatHistory` rehydrates (dicts parsed via `CoreMessage.parse_msg`), `structured_output` round-trips (`SerializeAsAny` + type-ref in `max_ai/core/type_ref.py`), loop metrics survive pause/resume. |
| 2 — Compaction | ✅ DONE | Summary block replaces (never stacks), pure `compact()` contract (caller applies), real prompt tokens in the budget, chunk-folded + per-message-capped summarization, mid-loop event telemetry filled. |
| 3 — Loop consolidation | ✅ DONE (modified) | **Per user decision:** `react_simple.py` and `react_planning.py` were DELETED, not deduplicated — `ReActLoopSelfDirected` is the single loop and the Agent default. Plan validators + `tool_hint`/`depends_on` in the `update_plan` schema; plan nudge is transient (never enters the transcript). |
| 4 — Durable elicitation | ✅ DONE | `INPUT_NEEDED` record state + `apply_user_answer`; executor short-circuits the ask-the-user tool (no 300s timeout bomb); questions survive process death; CLI polling removed; web UI answers via resume; mixed question+approval batch covered. |
| 5 — Loop guards | ✅ DONE (partial by design) | `max_ai/reasoning/guards.py`: schema-retry, repetition, budget, no-progress — wired transiently into the loop, on by default (`guards=[]` disables). Guards 5–6 (forced tool choice, step-scoped filtering) NOT built: provider-capability work / controller-loop-only → moot. |
| 6 — Checkpointing | ✅ DONE | `Agent(..., store=RunContextStore)` checkpoints after task, per iteration, and on pause/terminal; `agent.load_run()`; failed saves never kill the run. |
| 7 — Skills hardening | ⏸ PENDING (next up) | Docker flags, `SkillManifest`, session-scoped bash containers. |
| 8 — Concurrency-safe runs | ⏸ DEFERRED (user has other plans) | Do not start without discussing. |
| 9 — Multi-agent | ⏸ DEFERRED (user has other plans) | Do not start without discussing. |

Test suite at last checkpoint: **429 passed** (`pytest tests --ignore=tests/integration`).

---

## Phase 0 — ✅ DONE — One-line and one-file fixes (hours, zero risk)

### 0.1 Fix the failing smoke test / empty MemoryLayer
- **File:** whatever the `p0-embeddings-cleanup` branch changed in `capabilities/memory/` + `manager/stacks.py` (PromptVariablesBuilder memory collection).
- **What:** `tests/test_core_agent_smoke.py::test_core_agent_prepares_end_to_end` fails — seeded memory facts don't reach the rendered `MemoryLayer`. Debug why the memory snapshot is empty at `prepare()`. This is a silent "memory missing from prompt" bug, not just a red test.

### 0.2 Prompt template language fixes
- **File:** [max_ai/stacks/prompts/AgentPolicyLayer.j2](max_ai/stacks/prompts/AgentPolicyLayer.j2)
- **What:** "user routine when user explicitly references…" → "Use a routine when…"; same for the skills line. These sentences are in every agent's system prompt; broken English measurably confuses small models.
- Also: `.xlsl` → `.xlsx` in [max_ai/config.py:40](max_ai/config.py#L40); typo sweep (`Rolt`, `Respresents`, `criteriont`, `Evalidate`, `strucure`, `Executore`) — cosmetic, batch it.

### 0.3 Docstring drift
- **File:** [max_ai/base/agent.py:370](max_ai/base/agent.py#L370)
- **What:** Remove the `read_skill_resource` mention (tool doesn't exist; `tests/test_core_agent_smoke.py:280` asserts its absence). Update README §Skills accordingly.

### 0.4 Repo hygiene
- **What:** Add `user123/`, `user1234/`, `user_001/`, `var/`, `llm-models/`, `.venv/` to `.gitignore` and remove from tracking. Register the `integration` pytest mark in `pyproject.toml` to kill the warning.

### 0.5 Drop `ABC` from `Agent`
- **File:** [max_ai/base/agent.py:79](max_ai/base/agent.py#L79)
- **What:** `Agent(ComponentBase[BaseModel], ABC)` has no abstract methods and is instantiated everywhere. Remove `ABC` (or introduce a real abstract contract — but you don't need one).

---

## Phase 1 — ✅ DONE — Serialization correctness (1–2 days, low risk, high value)

The persistence contract is good; the payloads don't round-trip. Fix the data model, not the store.

### 1.1 `ChatHistory` must rehydrate (BUG, verified)
- **File:** [max_ai/types/chat_history.py](max_ai/types/chat_history.py), [max_ai/core/blocks.py:33-46](max_ai/core/blocks.py#L33-L46)
- **What:**
  1. Change `ChatHistoryBlock.message_history` annotation from `list[CoreMessage]` to `list[Message]` (the discriminated union in [max_ai/core/messages.py:386](max_ai/core/messages.py#L386)) so subclass identity (`tool_calls`, `tool_call_id`, `success`) survives deserialization.
  2. Replace the dict-rejecting validator in `ChatHistory` with one that **parses** dicts via `CoreMessage.parse_msg` (keep rejecting non-dict/non-CoreMessage garbage). The "use load_from()" ergonomics can stay as a convenience, but `model_validate_json` must work — it's what `FileSystemRunContextStore.load()` calls.
- **Test:** round-trip `RunContext` with populated `message_history` through `model_dump_json` / `model_validate_json`; assert message subtypes and `tool_calls` survive. Add the same round-trip to `tests/persistence/test_filesystem_store.py`.

### 1.2 `structured_output` round-trip (BUG, verified)
- **File:** [max_ai/core/messages.py:281](max_ai/core/messages.py#L281)
- **What:** `structured_output: BaseModel | None` deserializes into an empty `BaseModel`. Options (pick one):
  - **(recommended)** store `structured_output_data: dict` + `structured_output_type: str` (reuse `_type_ref`/`_load_type_ref` from [max_ai/base/agent.py:318-334](max_ai/base/agent.py#L318-L334) — move them to a shared util) and expose a `structured_output` property that lazily revalidates;
  - or mark it `exclude=True` and document the loss explicitly.
- **Test:** round-trip an `AssistantMessage` with a custom model; assert field values survive (or that the field is documented-excluded).

### 1.3 Persist a loop-metrics snapshot across pause/resume
- **Files:** [max_ai/base/reasoning.py](max_ai/base/reasoning.py) (`BaseLoopState`), [max_ai/base/agent.py](max_ai/base/agent.py) (`run_stream_events`)
- **What:** on any pause (`approval_needed` / `input_needed`), write `{iteration, llm_calls, tokens_input, tokens_output, eval_attempt}` into `ctx.runtime_state.shared_state["loop_metrics"]`; on resume, seed the fresh `LOOP_STATE_CLS` from it. Prevents the "every resume resets the iteration budget" drift and makes usage reporting truthful across segments.
- **Test:** run → pause on approval → resume; assert total iterations respect `max_loop_iterations` across segments.

---

## Phase 2 — ✅ DONE — Compaction correctness (1–2 days, localized risk)

### 2.1 Replace, don't stack, the summary block (BUG, verified)
- **File:** [max_ai/compaction/sliding_window.py:135-157](max_ai/compaction/sliding_window.py#L135-L157)
- **What:** `_inject_summary` appends a new `<COMPACTION_SUMMARY>` block after the existing one. Strip any previous block first:
  ```python
  base = re.sub(r"\n*<COMPACTION_SUMMARY>.*?</COMPACTION_SUMMARY>", "", existing, flags=re.S).rstrip()
  prompts.rendered_layers[ContextLayer] = f"{base}\n\n{block}" if base else block
  ```
  (Or keep the pristine rendered `ContextLayer` aside on first injection and always compose from it.)
- **Test:** call `_inject_summary` twice; assert exactly one block, containing the latest summary. Extend `tests/reasoning/test_mid_loop_compaction.py` with a run that compacts twice.

### 2.2 Single owner for `ctx.messages` mutation
- **Files:** [max_ai/compaction/sliding_window.py:104-108](max_ai/compaction/sliding_window.py#L104-L108), [max_ai/base/reasoning.py:260-289](max_ai/base/reasoning.py#L260-L289), [max_ai/base/agent.py:488-520](max_ai/base/agent.py#L488-L520)
- **What:** make `CoreCompaction.compact` **pure** (never assigns `ctx.messages`); both callers (`Agent._apply_compaction`, `BaseReasoning._run_mid_loop_compaction`) apply `ctx.messages[:] = result.recent_messages` when `result.changed`. Document this in the `CoreCompaction` docstring so custom strategies know the contract.

### 2.3 Use real prompt tokens in the budget
- **Files:** [max_ai/base/compaction.py:104-133](max_ai/base/compaction.py#L104-L133), callers in agent + reasoning.
- **What:** thread `prompts.prompt_tokens` into `live_message_capacity_tokens(...)` as the prompt budget (keep `compaction_prompt_budget_tokens` as fallback when prompts are unavailable). Recompute `prompts.prompt_tokens` after summary injection. Also count `ctx.message_history` in `_should_compact` (or explicitly document it as out of scope and assert it's empty when compaction is enabled).
- **Test:** client with small window + fat prompt layer → compaction triggers earlier than with the fixed 6000 constant.

### 2.4 Bound the summarization request
- **File:** [max_ai/compaction/sliding_window.py:159-220](max_ai/compaction/sliding_window.py#L159-L220)
- **What:** truncate each message in `_messages_transcript` to a per-message cap (e.g. 1–2k tokens, tool outputs especially), and if the assembled task still exceeds ~50% of the window, summarize in chunks (fold chunk N's summary into chunk N+1's "previous summary"). Also cache one `TokenCounter` on the loop instead of rebuilding per iteration ([base/reasoning.py:252](max_ai/base/reasoning.py#L252)).
- **Test:** compact a transcript containing a single 200k-char tool message; assert the summary call payload stays under budget.

### 2.5 Telemetry consistency (small)
- **File:** [max_ai/base/reasoning.py:277-289](max_ai/base/reasoning.py#L277-L289)
- **What:** fill `live_message_threshold_tokens` / `live_message_budget_tokens` in the mid-loop `CompactionEvent` (values are already computable there) instead of hardcoding 0.

---

## Phase 3 — ✅ DONE (modified: loops deleted, self-directed is the only one) — Deduplicate the reasoning loops (2–4 days, medium churn, unlocks everything after)

### 3.1 Template-method base loop
- **Files:** [max_ai/reasoning/react_simple.py](max_ai/reasoning/react_simple.py), [react_planning.py](max_ai/reasoning/react_planning.py), [react_self_directed.py](max_ai/reasoning/react_self_directed.py), [max_ai/base/reasoning.py](max_ai/base/reasoning.py)
- **What:** the three `execute_reasoning_loop` bodies share ~80% (pending-drain, LLM call, approval batching, scratchpad relay, input_needed handling) and have already diverged by accident (see 3.2). Move the canonical iteration into the base (or a `CoreReActLoop`) with hooks:
  ```python
  async def on_turn_start(ctx, prompts, state) -> AsyncGen[CoreEvent]     # planning step
  async def before_llm_call(ctx, prompts, state) -> AsyncGen[CoreEvent]  # step injection
  async def after_tool_round(ctx, state, tool_msgs) -> AsyncGen[CoreEvent]  # plan sync/advance, nudges, intermediate eval
  async def on_final_answer(ctx, state) -> AsyncGen[CoreEvent]           # step close-out, self-eval wrapper
  ```
  `ReActLoop` = no-op hooks. `ReActLoopSelfDirected` = `_register_runtime_tools` + `after_tool_round` sync/nudge. `ReActLoopPlanning` = `on_turn_start` planning + `before_llm_call` step injection + `after_tool_round` advancement + `on_final_answer` eval.
- **Why now:** Phases 5 (loop guards) and 9 (multi-agent) need one loop body to hook into, not three.
- **Test:** the existing `tests/reasoning/*` suites are your safety net — they should pass unchanged.

### 3.2 Fix the planning loop's missing runtime tools (BUG)
- **File:** [max_ai/reasoning/react_planning.py:101-170](max_ai/reasoning/react_planning.py#L101-L170)
- **What:** `ReActLoopPlanning` neither accepts `enable_human_input` nor calls `_register_runtime_tools`, so elicitation is silently unavailable there (its `input_needed` branch is dead code). Falls out for free from 3.1; if you fix it before, add the `super().__init__(enable_human_input=...)` pass-through and the `_register_runtime_tools(loop_state)` call.
- **Test:** clone `tests/reasoning/test_human_in_loop.py`'s core case against `ReActLoopPlanning`.

### 3.3 Plan model hygiene
- **Files:** [max_ai/reasoning/plan.py](max_ai/reasoning/plan.py), [max_ai/tools/update_plan.py](max_ai/tools/update_plan.py)
- **What:**
  - Add `AgentPlan` validators: unique step ids, ≤1 `active` step, `depends_on` references exist. Return validation failures as tool errors so the model self-corrects.
  - Either expose `tool_hint`/`depends_on` in the `update_plan` schema or remove them from `PlanStep`.
  - Replace the per-iteration `SystemMessage("plan-progress")` and the repeated stale-plan nudge with **transient** injection: compose them into the prompt for the current LLM call only (a `before_llm_call` hook that appends to a scratch copy of messages), instead of appending durable messages that accumulate in the persisted transcript.

---

## Phase 4 — ✅ DONE — Durable elicitation (HITL v2) (3–5 days, medium-high design work, your #1 architectural fix)

Make asking-the-user work exactly like approvals: **state, not a live future.**

### 4.1 Record-level state
- **Files:** [max_ai/types/tool_call.py](max_ai/types/tool_call.py) (`ToolCallRecord`), [max_ai/core/tool_state.py](max_ai/core/tool_state.py)
- **What:** add a record state `INPUT_NEEDED` (carrying `question`, `options`) and a transition `apply_user_answer(tool_call_id, answer)` → moves the record to `APPROVED`-equivalent ("answer available"), storing the answer so the executor can complete the call by returning the answer as its `ToolResult`.

### 4.2 Executor short-circuit
- **File:** [max_ai/base/tool_executor.py](max_ai/base/tool_executor.py)
- **What:** treat `structure_human_in_loop` like approval-gated tools: when the executor meets a fresh user-input record, **do not execute** — mark `INPUT_NEEDED`, emit `UserInputRequestEvent`, and let the loop finish the turn with `finish_reason="input_needed"` (mirror the `ToolApprovalEvent` batching path in the loops). When it meets an answered record (post-`apply_user_answer`), synthesize the `ToolMessage` from the stored answer.
- **Consequences:**
  - the tool no longer blocks → no `asyncio.wait_for` 300s bomb ([max_ai/executor/local.py:64-75](max_ai/executor/local.py#L64-L75)) — the current timeout bug disappears structurally;
  - the CLI's 50 ms polling and the "event emitted after the block" contortion ([max_ai/cli/repl.py](max_ai/cli/repl.py)) go away — the CLI consumes `UserInputRequestEvent` from the stream, prompts, calls `ctx.tool_state.apply_user_answer(...)`, and `agent.resume(...)` — identical shape to the approval flow it already implements;
  - a pending question **survives process death** and serializes with `RunContext`.
- **Keep** the in-process fast path (future + `provide_user_input`) only if you want same-turn continuation in the web UI; put it behind the executor (answer arrives before the turn ends → complete inline). If that dual mode complicates things, drop it — resume-based flow is enough and is what multi-agent needs.
- **Tests:** rewrite `tests/reasoning/test_human_in_loop.py` for pause/resume semantics; add: process-death simulation (serialize ctx while `INPUT_NEEDED`, reload, answer, resume); same-batch question+approval ordering (the dangling-state edge case).

### 4.3 Elicitation policy knobs (small, after 4.2)
- optional `max_questions_per_turn` guard; optional default answer + timeout at the *agent* level (not the executor timeout); validate `options` non-empty strings.

---

## Phase 5 — ✅ DONE (guards 1–4; 5–6 skipped) — Small-model steering: loop guards (3–5 days, additive)

Builds on Phase 3's single loop body. New module `max_ai/reasoning/guards.py`.

### 5.1 Guard contract
```python
class LoopGuard(Protocol):
    def check(self, ctx: RunContext, state: BaseLoopState) -> GuardAction | None: ...
# GuardAction = inject transient steering message | force tool_choice | end turn
```
Run guards in `after_tool_round` / `before_llm_call` hooks. Config: `ReActLoop(guards=[...])` with a sensible default set.

### 5.2 The guards, in payoff order
1. **Schema-retry** — on `invalid_parameters` failures, inject the expected JSON schema + the specific mismatch. (Executor already produces the validation error; the guard reformats it.)
2. **Repetition detector** — hash `(tool_name, canonical_json(parameters))`; on N-th repeat inject "same call, same result — change approach or answer now."
3. **Budget warning** — at 75% of `max_loop_iterations` inject "K iterations left; consolidate and answer." Prevents the max-iterations cliff.
4. **No-progress detector** — empty assistant content and no tool calls → re-prompt with an explicit action menu.
5. **Forced tool choice** — when the active plan step has a `tool_hint` and the provider supports it, set the provider's forced/required tool-choice for that call (client capability flag in `ModelConfig`; wire through `client.run(**kwargs)`).
6. **Step-scoped tool filtering** — controller loop only: expose only the tools relevant to the active step (from `tool_hint` + priority tools). Biggest single win for ≤8B models.
- **Tests:** each guard is a pure function — unit-test with synthetic `ctx`/`state`; one integration test with a scripted fake client that loops until a guard fires.

---

## Phase 6 — ✅ DONE — Wire persistence into the run lifecycle (2–3 days)

### 6.1 Optional checkpointing on the Agent
- **Files:** [max_ai/base/agent.py](max_ai/base/agent.py), [max_ai/persistence/core.py](max_ai/persistence/core.py)
- **What:** `Agent(..., store: RunContextStore | None = None)`. When set, `run_stream_events` checkpoints `ctx` at safe points: after the task is appended, after each tool round, on every pause (approval / input_needed), and on terminal response. Failure to save logs a warning, never kills the run.
- **Why after Phase 1:** checkpointing a context that can't rehydrate is worse than useless.
- **Test:** kill a fake run after N tool rounds (raise inside a scripted client), reload from store, `resume()`, assert continuation; assert stale `EXECUTING` records get the fail-stale policy.

### 6.2 Session/run lifecycle helpers (optional)
- `Agent.load_run(run_id, store)` convenience; `store.list()` already exists for cleanup tooling.

---

## Phase 7 — ⏸ PENDING — Skills hardening (2–3 days, mostly executor flags)

### 7.1 Container hardening
- **File:** [max_ai/executor/docker/docker.py:399-413](max_ai/executor/docker/docker.py#L399-L413) (both the bash-session `docker run` and the one-shot path)
- **What:** add `--memory 1g --cpus 1 --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges`; run as a non-root user baked into the sandbox image; consider `--read-only` + tmpfs for `/tmp` (skills write to `/mnt`, which stays writable). Make limits `DockerExecutor.__init__` parameters with safe defaults.
- **Test:** extend `tests/executor/test_docker_executor.py` to assert the flags appear in the constructed command.

### 7.2 Skill manifest validation
- **File:** [max_ai/base/skills.py:234-280](max_ai/base/skills.py#L234-L280)
- **What:** `SkillManifest(BaseModel)`: `name`, `description` (required), `version`, and forward-compatible optional fields (`network: Literal["none","allowlist"]`, `packages: list[str]`). Validate in `_read_skill_block`; fail `prepare()` with a clear error. `SkillBlock` stays the lightweight prompt-facing projection.

### 7.3 Session-scoped bash containers
- **File:** [docker.py `_bash_session_key`](max_ai/executor/docker/docker.py)
- **What:** key sessions by `(user_id, session_id)` so concurrent sessions of one user don't share shell state. Keep TTL cleanup as is.

### 7.4 (Later) dynamic skill discovery
- A `search_skills` tool over a larger catalog (metadata-only, same pattern as routines), so agents aren't limited to a fixed constructor list. Only worth it once you have >10 skills.

---

## Phase 8 — ⏸ DEFERRED (per user) — Concurrency-safe runs (3–5 days, touches the agent/loop contract)

Prerequisite for multi-agent; acknowledged in your own code comments.

### 8.1 Kill the `bind()` mutation
- **Files:** [max_ai/base/reasoning.py:219-245](max_ai/base/reasoning.py#L219-L245), [max_ai/base/agent.py:701-728](max_ai/base/agent.py#L701-L728)
- **What:** two options —
  - **(cleaner)** pass runtime deps as an explicit `RunBinding` argument to `execute_reasoning_loop(binding=...)` and delete `bind()`/the runtime accessors;
  - **(less churn)** keep the user-facing config instance as a *prototype*: `_build_reasoning` clones it per run (`copy.copy` + bind on the clone). `_current_loop_state` then lives on the per-run clone, fixing shared-state races for free.
- **Test:** `asyncio.gather(agent.run(a), agent.run(b))` with a scripted client; assert both complete with uncorrupted, distinct transcripts.

### 8.2 Key pending input by run
- **File:** [max_ai/base/agent.py:1079-1131](max_ai/base/agent.py#L1079-L1131)
- **What:** `_active_reasoning` single slot → `dict[run_id, ...]`; `provide_user_input(answer, run_id=None)` (None = the only active run, error if ambiguous). Largely moot once Phase 4 lands (answers flow through `tool_state` + resume), but the web-UI fast path needs it if kept.

---

## Phase 9 — ⏸ DEFERRED (per user) — Multi-agent foundation (1–2+ weeks, the big one)

Your stateless architecture is the right base. Build multi-agent as **composition over RunContexts**, not as a new runtime.

### 9.1 Pause propagation through `AgentAsTool` (do this first — it's the litmus test)
- **File:** [max_ai/tools/agent_as_tool.py:176-190](max_ai/tools/agent_as_tool.py#L176-L190)
- **What:** today a child ending in `approval_needed`/`input_needed` is flattened into a "success" text result and the child context is discarded. Instead:
  - persist the child `RunContext` (Phase 6 store) under its `run_id`;
  - return a **paused** `ToolResult` carrying `{child_run_id, pause_kind, question/approvals}` in metadata;
  - the parent loop surfaces this as its own pause (same `INPUT_NEEDED`/approval machinery from Phase 4 — this is why Phase 4's data-not-future design matters);
  - on resume, the answer routes to the child (`apply_user_answer` on the child's ctx), child resumes, its final text becomes the tool result.
- **Test:** parent → `AgentAsTool` child that asks a question → parent pauses → answer → child completes → parent continues. This single test proves the whole architecture composes.

### 9.2 Orchestration patterns (pick ONE to start: supervisor)
- New module `max_ai/orchestration/`:
  - `Supervisor`: an agent whose toolset is `AgentAsTool` children + Phase 9.1 propagation. You almost have this today — it's mostly the pause plumbing plus fan-out (`asyncio.gather` over child runs, needs Phase 8).
  - Defer graphs/handoffs/blackboards until supervisor works end-to-end; they're additional routing policies over the same primitives (`RunContext` + store + events).
- Shared identifiers: add `parent_run_id: str | None` to `RunContext` (you already stash `parent_tool_call_id` in `shared_state` — promote it to a typed field).

### 9.3 Cross-agent observability
- Events already carry `source`; add `run_id` to `CoreEvent` (many events have it implicitly via ctx but not on the event) so a UI can interleave parent+child streams. Consider an OpenTelemetry exporter middleware later — the event stream maps 1:1 onto spans.

---

## Suggested sequence at a glance

| Phase | Theme | Effort | Risk | Unblocks |
|---|---|---|---|---|
| 0 | Typos, docs, hygiene, failing test | hours | none | credibility |
| 1 | Serialization round-trip | 1–2 d | low | 6, 9 |
| 2 | Compaction bugs | 1–2 d | low | long runs |
| 3 | Loop deduplication | 2–4 d | med | 4, 5, 9 |
| 4 | Durable elicitation | 3–5 d | med | 9.1, kills 3 bugs |
| 5 | Small-model loop guards | 3–5 d | low | quality on small models |
| 6 | Checkpointing wired in | 2–3 d | low | crash recovery, 9 |
| 7 | Skills hardening | 2–3 d | low | safe skills at scale |
| 8 | Concurrency-safe runs | 3–5 d | med | fan-out |
| 9 | Multi-agent (supervisor first) | 1–2+ w | high | the roadmap goal |

Rationale for the order: Phases 0–2 are pure debt payoff with immediate correctness wins. Phase 3 is refactoring you must do before adding more loop behavior, or the three copies diverge further. Phase 4 is the one *architectural* change you asked about that everything else (CLI simplicity, crash recovery, multi-agent pause bubbling) depends on. Phases 5–7 are additive value. Phases 8–9 are the multi-agent runway — and by then, every primitive they need (durable state, pause-as-data, checkpoints, one loop body) already exists.
