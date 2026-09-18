# New runtime

The new `max_ai.base.agent.Agent` now uses the registry, dispatcher and session
manager. Execution providers and usage examples are documented in
[../runtime/README.md](../runtime/README.md).

- `max_ai.base.tool_registry.ToolRegistry`: native tool catalog and remote references.
- `max_ai.base.tool_dispatcher.ToolDispatcher`: validation, approval and invocation.
- `max_ai.base.runtime_executor.Executor`: tool execution and environment contract.
- `max_ai.environment.session_manager.EnvironmentManager`: session lifecycle and sync.
- `max_ai.runtime`: Local, Docker and Modal implementations.

The previous agent is preserved in `max_ai/base/agent_copy.py`. Existing
`environment/manager.py`, `environment/docker.py` and `executor/` remain as
reference during migration. Use the new runtime imports for the new Agent.
