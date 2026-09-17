# MaxAI Harness — Project Considerations

> Editable project brief retained from the initial draft. The project owner may change any part of this document.

## Goal

Improve MaxAI into a reliable agent harness by making runtime capabilities, execution, state, permissions, budgets, stopping conditions, and evidence explicit. The model should interpret requests and propose actions; the harness should decide which capabilities are available, enforce policy and resource limits, track outcomes, and report what it can verify.

Preserve useful MaxAI components and evolve them incrementally. Do not propose a full rewrite without concrete evidence that the existing core cannot support the needed behavior.

## Current migration scope

Build a **document-focused harness first**. The first useful workflow is:

> Open a document in an explicitly authorized workspace, change a requested section, save the change, and report what was saved and checked.

Start with text and Markdown. Add DOCX or other formats only through format-aware adapters and tests that verify structure is preserved. Keep generated artifacts separate from the user's source documents.

Do not include a coding agent, arbitrary shell/code execution, Git integration, or deployment in this first stage. Revisit coding after the document workflow is reliable. Git is not a prerequisite for editing documents.

## Desired harness responsibilities

- **Agent loop:** control turns, tool execution, continuation, cancellation, and terminal states.
- **Context and compaction:** assemble relevant instructions, history, workspace facts, and tool results within explicit limits.
- **Tools and dispatch:** expose only capabilities configured for the run; validate inputs and route calls to the correct executor.
- **Workspace:** provide the same authorized document root to the UI, tools, and runtime. Do not expose the whole host filesystem by default.
- **Policy and approvals:** allow, ask, or deny actions based on the operation, arguments, identity, and workspace scope.
- **State and persistence:** distinguish conversation history, execution records, approvals, and durable user memory; support safe pause/resume.
- **Memory:** let the harness decide when to extract, validate, store, and retrieve memories. A model may suggest candidates; its claims alone are not proof.
- **Budgets:** enforce run-level token, time, and money limits across model calls, retries, compaction, memory work, tools, and future subagents.
- **Stop conditions and completion:** report completed, partial, failed, blocked, unverified, cancelled, or budget-exhausted outcomes. A model's final text alone does not prove a side effect happened.
- **Observability:** make model proposals, policy decisions, tool execution, evidence, errors, and resource usage inspectable.
- **Multiagent:** consider only after a single-agent document workflow is dependable. Subagents must receive scoped context and permissions and share the parent's budget.

## Document-workspace boundaries

- Preserve the current artifacts workspace for generated deliverables.
- Introduce a separately configured document/project workspace for source files.
- Bind the selected root consistently across UI, agent context, and document tools.
- Validate paths and supported formats at the runtime boundary, not only in prompts.
- Keep version/revision information so conflicts are visible and changes can be recovered.
- Do not enable Bash merely to provide document editing.

## Findings already checked in the repository

These are current observations from the prior code review; re-check the implementation before changing it.

- `max_ai/tools/workspace.py` exposes read/list operations for user artifacts; it is not a general project-document editor.
- `AgentCapabilities` currently adds native Bash when skills are configured. The relationship between effective runtime tools and the sandbox requirement needs review.
- Planning through `update_plan` is optional in the self-directed loop; the model can decide whether to use it.
- The loop already has iteration limits and guards. Guards often steer the model; steering is not itself a hard policy guarantee.
- Context token accounting/compaction and per-tool timeouts exist. A global ledger covering all model, auxiliary, tool, and future subagent work needs review.
- Memory has persistent backends and maintenance after compaction; persistence of run state, semantic memory, and tool outcomes should remain distinct.
- Grok Build, DeepSeek Harness, OpenCode, and Pydantic AI/Harness have been researched as architectural references. Use the reports in `a-harness-max-ai/`; do not treat another project's design as proof that MaxAI should adopt its dependency or copy its features.

## Working method

1. Inspect the actual entrypoint/configuration and relevant code before diagnosing a reported behavior.
2. Separate verified code behavior, documented behavior, inference, and unknowns.
3. Prefer small, reviewable migration steps with explicit acceptance criteria and rollback/migration notes.
4. Keep existing interfaces compatible where practical; version persisted state when its schema changes.
5. Test harness behavior with deterministic/mock model calls first. Do not require paid APIs for routine tests.
6. Do not silently widen filesystem, network, or execution permissions.
7. Do not implement coding, Git, or multiagent scope until the owner brings it into the active stage.
8. Explain the concrete behavior change and the evidence used to validate it.

## Planning documents

- Migration plan: `a-harness-max-ai/06-plan-migracion.md`
- Local architecture diagnosis: `a-harness-max-ai/03-diagnostico-capabilities.md`
- Memory, completion, and budgets: `a-harness-max-ai/04-memoria-gates-presupuestos.md`
- Open-source harness comparison: `a-harness-max-ai/02-componentes-open-source.md`
- Pydantic AI and Harness research: `a-harness-max-ai/05-pydantic-ai-luna.md`

The first planned implementation step is to establish the current behavior, then make document capabilities and workspace binding explicit. No migration phase should be considered implemented just because it appears in the plan.

## Owner decisions to edit

- Supported document formats for the first release:
- How the user selects/configures the document workspace:
- Whether edits are saved directly or proposed as a reviewable revision first:
- Which operations need human approval:
- Default token, time, and money limits:
- Memory retention and deletion rules:
- Other requirements or corrections:
