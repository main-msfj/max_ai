# Max AI Examples

- [Basic](basic/README.md): one feature per file.
- [Advanced](advance/README.md): implementing base interfaces.

> **Most examples are outdated.** They were written for the framework before
> the November 2026 rewrite and have not been ported yet. Each folder's README
> lists which files still match the current API and what changed. For the
> current API, start from the quick start in the [main README](../README.md).

What changed, in short:

| Old | Now |
|---|---|
| `async with agent:` | no context manager; call `await agent.close()` at the end |
| `agent.run(task, run_context=ctx)` | `agent.run(task, ctx)` |
| `response.final_text`, `response.context` (`AgentResponse`) | `state.message.text()`, `state.finish_reason` (`LoopState`); the conversation stays in `ctx` |
| `response.pending_approvals`, `agent.resume(ctx)` | `ctx.tool_state.pending_approvals`, `agent.approve(ctx, id)`, then `agent.run(None, ctx)` |
| `OllamaChatCompletionClient` | removed; OpenAI and OpenRouter clients only |
| `LocalMemoryRegistry`, `max_ai.base.memory` | `max_ai.capabilities.memory.Memory` (needs a sandbox) |
| `LocalWorkspace`, `max_ai.base.workspace` | sandboxes: `max_ai.capabilities.sandbox.{local,docker,e2b}` |
| `GithubSkillRegistry` | `max_ai.capabilities.skills.GitHubSkills` / `LocalSkills` (need a sandbox and `BashTool()`) |
| `compaction=` on `Agent` | `reasoning=ReactLoop(compaction=...)` |
| `MiddlewareContext`, `on_*` hooks | `CoreMiddleware.turn / model / stream / tool` |

The `example-for-deployment/` folder mentioned in older docs is not in this
repository.
