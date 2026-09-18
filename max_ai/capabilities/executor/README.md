# Native tool execution runtime

This is the new implementation used by `max_ai.base.agent.Agent`. The previous
agent is preserved, unchanged at the time of copying, in
`max_ai.base.agent_copy.Agent`. Old executors remain in `max_ai.executor` for
reference; new providers are imported from `max_ai.runtime`.

```text
Agent → ToolRegistry → ToolDispatcher → Executor.run_tool
                            │               ├── Local: CoreTool.execute
                            │               ├── Docker: Python tool worker
                            │               └── Modal: Python tool worker
                            └── explicit host tools: live MCP / SDK adapters

EnvironmentManager → connect → sync → reuse → idle expiry → clean
Workspace          → .agents/<user>/skills + <conversation>/files
```

## Configure an agent

```python
from max_ai.base.agent import Agent
from max_ai.runtime import LocalExecutor, DockerExecutor, ModalExecutor
from max_ai.types.run_context import RunContext

# client is an existing CoreChatCompletionClient instance.
agent = Agent(
    name="documents",
    description="Create and edit documents in the user's workspace",
    instructions="Use the registered file tools. Ask for approval before writing.",
    client=client,
    executor=DockerExecutor(image="maxai-runtime:latest", network="none"),
)

async with agent:
    response = await agent.run(
        "Create notes.txt containing Hello",
        run_context=RunContext(user_id="user_123", session_id="conversation_1"),
    )
    # Your application displays response.pending_approvals to the user.
    # Only after the human approves:
    # response.context.tool_state.apply_approval(call_id, approved=True)
    # response = await agent.run(run_context=response.context)
```

The conversation-relative native file tools are registered by default.
Built-in filesystem, user-input and plan tools run on the host. User-supplied
tools use the configured executor. Bash is optional and
uses the same path; it is not required to read, find, create or edit files.
Local executes native Python adapters in the agent process. Tools must use
ToolContext's filesystem/environment instead of assuming a global cwd. Local
is not a filesystem or network sandbox; timeout cannot forcibly stop arbitrary
Python threads that ignore cancellation.

## Additional CoreTools and MCP

`CoreTool` is this repository's base tool class. Existing `@tool` functions and
CoreTool instances work locally. Remote tools must be importable and installed
in the image. Live objects/closures are never pickled or silently run on the host.

```python
agent = Agent(
    name="assistant", description="Assistant", instructions="Help the user",
    client=client, toolset=[my_tool],
)
```

Agent owns its registry internally; there is no public `registry` constructor
argument. Custom remote tools provide reconstruction through `docker_ref()`.
The Agent API does not currently expose a per-tool host override for live MCP
connections; do not send those connections to a remote executor.

For a top-level function, use `kind="function"`; for a factory exposing `.tools`,
use `kind="factory", tool_name="the_name"`. Built-in filesystem tools use that
factory mechanism. Existing `docker_ref()` implementations are adapted, and
BashTool has a built-in reference. Worker schema and call identity
must match the advertised tool. Custom dependencies must be in the image;
custom host `deps` and secrets are not automatically transmitted.

## Docker

The new Dockerfile reuses utilities and non-root image configuration from
`environment/Dockerfile`, and adds the framework installation for native tools.
The older environment image lacks that installation and is not interchangeable.

```bash
docker build -f max_ai/runtime/Dockerfile -t maxai-runtime:latest .
```

Run the build from the repository root. UID/GID can be selected at build time
with `--build-arg AGENT_UID=... --build-arg AGENT_GID=...`. The image user must
have write access to the host user's workspace; startup fails explicitly
otherwise. The executor never switches to root or changes host ownership.
Network accepts `none` or `unrestricted`. Only the user's directory is mounted
at `/workspaces/<user_id>`. Shared mount synchronization requires no transfer.
Timeout/cancellation removes that container; files remain on the host.

## Modal

```bash
uv sync --extra runtime-modal
uv run modal setup
```

Pass a prepared Modal Image or a published image reference containing the
framework and custom tools:

```python
executor = ModalExecutor(
    image="your-registry/maxai-runtime:your-tag",
    network="none",
)
```

A local Docker tag is not automatically uploaded to Modal. The image must run
as non-root and permit writes under `/workspaces`. `connect` creates a Sandbox;
there are no cloud operations until tools actually need a session.

The initial sync implementation transfers regular-file content, including
creations, edits and deletions, with conflict detection against the last common
snapshot. It rejects symlinks and special files and limits snapshots to 8 MiB
and 10,000 files. Modes and empty directories are not synchronized. Updates
are atomic per file, not transactional across the workspace. Concurrent external
writers require coordination; conflicts retain the session for recovery.

Cancellation asks a command supervisor to stop its process group while retaining
the sandbox filesystem for download. The Sandbox has a hard lifetime (default
one hour); process/server failure or expiry before sync can still lose unsaved
remote changes. This is workspace synchronization, not an artifact backup.

Official references used for the implementation, reviewed 2026-09-15:
[Sandbox lifecycle](https://modal.com/docs/guide/sandboxes),
[command execution](https://modal.com/docs/guide/sandbox-spawn), and
[filesystem access](https://modal.com/docs/guide/sandbox-files).

## Current migration boundary

The new Agent implements model/tool turns, native tools, skills materialization,
approval pause/resume, cancellation, iteration limits, model streaming and tool
call/result events. Manager sessions are internal and expire after five idle
minutes by default. Close the agent explicitly or use `async with`.

It is a new, smaller API: legacy memory, compaction, reasoning plugins,
middleware, checkpoint stores, agent serialization and UI configuration have
not been migrated. Unsupported constructor arguments raise an error. Existing
examples/UI using the former API must be migrated or explicitly import
`agent_copy.Agent`. A model final answer is not an independent verifier.

Remote worker-internal progress events are not yet streamed back; dispatcher
call/approval/result events are emitted on the host. Modal SDK installation,
Docker image building and real provider execution were not performed in this
change. The legacy test suite was not run.
