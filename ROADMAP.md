# Roadmap

Planned, not started. Order is not a commitment.

- **Multi-agent** — agents that delegate to other agents, designed on the
  stateless `Agent` (replaces the removed `AgentAsTool`).
- **Native clients** — Anthropic first (prompt caching, extended thinking),
  then Gemini and Azure OpenAI. Today: OpenAI, OpenRouter, Ollama.
- **Session store versioning** — optimistic concurrency so two processes
  saving the same conversation never overwrite each other silently.
- **Search past conversations** — a tool backed by the session store and
  `CoreEmbedding`.
- **Guardrails** — middleware for PII and prompt injection with a pluggable
  classifier.
