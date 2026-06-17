# Max AI — Model/Framework Evaluation — 2026-06-17

Evaluation of the agent framework after this session's work, written from the
actual repo state (branch `p0-embeddings-cleanup`), not from memory. 417 tests
pass. Sections: what's solid, what's missing, what's *fragile or wrong* (the
"feedback I don't think is right" the user asked to flag).

---

## 1. What was built this session (verified)

| Area | State | Tests |
|---|---|---|
| Planning-as-controller (`ReActLoopPlanning`) | steps 1–5: status model, plan on `RunContext`, live step injection, mark-done (rules A+C), replan on failure | yes |
| Planning-as-tool (`ReActLoopSelfDirected`) | model drives its own plan via `update_plan`; loop syncs `plan_draft → ctx.plan` | yes |
| `EvalConfig` refactor | killed 4 boolean flags; `__init__` 9→4 params; planning intrinsic, eval opt-in | yes |
| Self-eval | cost cap, per-run criteria, intermediate eval | yes |
| Live plan in UI | `PlanningEvent` → server payload → `#planPanel` + trace panel | manual only |
| Human-in-the-loop in UI | agent `provide_user_input` + `/api/chat/input` + interactive card | manual only |
| Example | `examples/agent_self_directed_loop.py` (Ollama/OpenAI + MCP + self-directed) | n/a |

This is **core framework**, not integrations. The hard part. It is in good shape.

---

## 2. Real gaps found (with evidence)

### 2.1 BLOCKER — the human-input tool is never registered
- `UserInputTool` (`TOOL_NAME = "structure_human_in_loop"`, in
  `max_ai/tools/structure_human_in_loop.py`) needs a per-run `loop_state`, exactly
  like `UpdatePlanTool`.
- **No loop auto-registers it.** `ReActLoopSelfDirected` auto-registers
  `update_plan`, but nothing registers `structure_human_in_loop`.
- **Consequence:** the human-in-the-loop UI we wired this session cannot actually
  fire from the example — the model has no tool to ask a question with. The UI is
  correct; the tool just never reaches the model.
- **Fix:** mirror the `update_plan` auto-registration pattern (register the tool
  bound to the run's `loop_state` at loop start), or register it in the agent.
  Decide *which loops* should expose it (probably all, since all three emit
  `UserInputRequestEvent`).

### 2.2 Event union is incomplete
- `AgentEvents` union (`max_ai/core/event_type.py`) lists `PlanningEvent`,
  `ScratchpadUpdateEvent`, `EvalEvent` — but **`UserInputRequestEvent` is NOT in
  the union**.
- Risk: anywhere the union is used for discriminated (de)serialization or
  validation, `UserInputRequestEvent` falls through. Works today because the UI
  path uses `isinstance`, but it's a latent inconsistency.
- **Fix:** add `UserInputRequestEvent` (and audit the union vs. the full event
  list — confirm nothing else is missing).

### 2.3 UI features have zero automated coverage
- The live plan and human-input flows were verified **only by payload checks and
  reasoning**, never in a browser or against a running agent (no node/headless in
  this env).
- The `turn_lock` + blocking `await future` interaction (input endpoint resolving
  a future held by an in-flight SSE stream) is **unproven end-to-end**. This is the
  single most likely place for a real-world bug (potential deadlock if the resume
  path ever contends for `turn_lock`).
- **Fix:** one integration test that drives a fake agent through an input request
  and resumes it; and a manual browser smoke test.

---

## 3. Fragile / questionable design choices (flag for review)

These are the "feedback I'm not sure is right" — judgment calls worth a second look.

### 3.1 Rule C marks a step `done` even after a failed tool
- In `ReActLoopPlanning`, if a tool fails and the model then gives a final answer,
  Rule C marks the active step `done` — even though it failed. Defensible ("the
  model decided it's finished"), but it means a *failed* step can end up `done`,
  not `failed`. **Is that the semantics you want?** Possibly fine; flagging because
  it surprised us when we found it.

### 3.2 Replan trigger is failure-count only
- Piece 5 replans after `max_step_retries` *tool failures*. It does **not** replan
  on "stuck but not failing" (model loops without progress, or the plan is just
  wrong). For a small local model, "wrong plan, no failures" is a real failure mode
  with no recovery today.

### 3.3 Two plan representations still coexist
- `ScratchpadTool` (`update_todo`) writes `loop_state.scratchpad`; `UpdatePlanTool`
  writes `ctx.plan` (`AgentPlan`). We chose `AgentPlan` as the canonical one for
  the UI, but `scratchpad` still exists and emits `ScratchpadUpdateEvent`. Two
  "todo/plan" concepts in the framework is a smell — decide if scratchpad stays
  (for free-form notes) or gets folded in.

### 3.4 `self_directed` planning is purely optional with no nudge
- `ReActLoopSelfDirected` only produces a plan if the model *chooses* to call
  `update_plan`. With a weak local model (qwen 4b) it often won't, so the plan
  panel stays empty. The example's instructions ask for it, but there's no
  structural guarantee. Acceptable by design — just know the UX depends on model
  quality.

### 3.5 Commit history is messy
- Four commits literally named "planning-as-controller done" + one "adding new
  features". Fine on a WIP branch, but **squash/rewrite before merging to main** so
  the history tells the real story.

---

## 4. What's missing for a "good framework" — but is NOT urgent

Separating *core that matters* from *optional plugins* (the Hermes-envy trap):

**Core-ish (worth doing eventually):**
- Durable run store / proper resume across process restarts (P2, deferred)
- Anthropic client (P2, deferred — "API caro por ahora")
- Observability / OpenTelemetry (P4, deferred)
- Eval & human-input UI coverage (see 2.3)

**Plugins (add one at a time, only when a real case needs it — NOT framework work):**
- Browser automation (Hermes uses Browser Use) — you have Tavily for search only
- More tools (TTS, image gen, etc.) — each is "an afternoon", not architecture
- Multi-agent (P3, deferred — correctly, "no hay multiagente sin un agente bueno")

**The honest takeaway:** the *core* is ~90% there. The long list that feels
overwhelming is mostly plugins, which are quantity, not difficulty. Don't chase
Hermes's feature count; it doesn't make a framework good — a clean loop does, and
that part is done.

---

## 5. Priority order to act on this

1. **Fix 2.1** (register the human-input tool) — otherwise this session's UI work
   is dead code. Highest leverage.
2. **Fix 2.2** (add `UserInputRequestEvent` to the union) — 1 line, prevents a
   latent bug.
3. **Add 2.3** (one human-input integration test) — proves the riskiest path.
4. **Decide 3.1 / 3.3** (Rule-C-on-failure semantics, scratchpad vs plan) — design
   calls, not bugs.
5. **Squash before merging to main** (3.5).
