# Execution environments

`Environment` is the shared parent for `start`, `execute`, `stop`, and runtime
path variables. `DockerEnvironment` implements it using the Docker CLI, without
an SDK dependency. One instance belongs to one user and conversation.

Build the runtime image explicitly (Bash, Python/uv, Node/npm, Rust/Cargo):

```sh
docker build -t maxai-environment:latest -f max_ai/environment/Dockerfile .
```

Use from async application code:

```python
from max_ai.base.workspace import Workspace
from max_ai.environment import DockerEnvironment
from max_ai.termination import CancellationToken

async def example():
    workspace = Workspace()
    token = CancellationToken()
    async with DockerEnvironment(
        workspace, "user_123", "conversation_1", network="unrestricted"
    ) as env:
        result = await env.execute('ls "$WORKSPACE"', timeout=10, cancellation_token=token)
        print(result.stdout, result.stderr, result.exit_code)
```

The daemon must run on the machine containing the workspace. If the application
itself runs in a container, its workspace path must also exist at that same path
on the Docker daemon's host. No automatic remote directory transfer is provided.

The user directory is mounted read-write at `/workspace`, including skills.
Only `$WORKSPACE` is exposed; commands start there. Other conversations of the same
user remain accessible. No other users' directories are mounted. The image root
is read-only; `/tmp` is ephemeral. Resource limits apply.

`network="none"` (default) disables networking. `network="unrestricted"` uses
Docker bridge networking with no application destination filter; actual access
still depends on the host network. No allowlist mode is provided.

The image declares `USER agent` (UID/GID 1000), `WORKDIR`, `$WORKSPACE`, writable
package-cache locations, and its startup command. Python does not override the
image's user or command. Runtime libraries are Debian's packaged versions; change
the image when another language/toolchain version is needed. Install project
packages in workspace environments (uv venv, npm project, Cargo project).

On Linux bind mounts retain host ownership. Build with nonzero `AGENT_UID` and
`AGENT_GID` matching the workspace's owning service user when it differs from 1000:

```sh
docker build --build-arg AGENT_UID=1001 --build-arg AGENT_GID=1001 \
  -t maxai-environment:latest -f max_ai/environment/Dockerfile .
```

Startup checks both the effective non-root UID and write access to `/workspace`.
Root-owned workspaces must be provisioned with access for the image user by the
host administrator; the adapter does not recursively chown or chmod user files.
Custom images must keep `/workspace` as WORKDIR, set WORKSPACE=/workspace, provide
Bash/coreutils, declare a non-root USER, and start a long-lived process.
Network, mounts, read-only root and resource restrictions remain runtime Docker
flags because Dockerfiles cannot enforce them.

Each command starts a fresh Bash process. Files persist across commands, but shell
variables and `cd` do not. Output is bounded per stream. Timeouts remove the whole
container, return `timed_out=True`, and discard partial output. Cancellation also
removes it and propagates `CancelledError`. Pass the existing `CancellationToken`
to `execute`; triggering `token.cancel()` cancels that command. An already cancelled
token prevents execution. `stop()` is the separate explicit cleanup operation. Call `start()` to use the instance
again. Cleanup failures are raised instead of claiming processes were stopped.

BashTool routes commands to this environment through the existing tool executor.
Command/argument filtering remains a separate integration step.
`execute` accepts shell text from its caller; it does not implement an allowlist.
It does not sync Artifacts, install skills, or provide Local/E2B backends.

Design reference: the lifecycle separation in
https://github.com/coderonfleek/harness-course-build/blob/main/harness/sandbox/docker_sandbox.py.

## Idle lifecycle

The image is built once and reused; creating a container does not rebuild it.
`idle_timeout=300` removes an idle container after five minutes. The timer pauses
while a command runs and restarts after it finishes. Call `start()` before a new
execution to recreate the container if it expired (or renew it if still alive).
Keep the same environment instance for the same user/conversation to reuse it.

Call `await env.release()` when a turn/conversation finishes to begin the grace
period. `await env.stop()` and exiting `async with` remove it immediately.
Cancellation/timeouts also remove it immediately to terminate descendants.
Workspace files survive removal; processes and temporary files do not.
The application still needs to wire conversation lifecycle events to these methods.
The idle timer needs the application event loop to remain running; application
shutdown must call `stop()`. This is not a daemon-side expiry after a host crash.

The image also preinstalls wget, jq, zip/unzip, less, tree, fd-find (Debian command:
`fdfind`), sqlite3 and openssl. Installing these at build time avoids repeating
system setup per conversation and preserves Docker layer caching when later
image instructions change. These utilities do not enforce command permissions;
the non-root user, mounts and network mode remain the execution constraints.

## BashTool and execution events

Configure a factory on the agent and register an unbound BashTool:

```python
from functools import partial
from max_ai.base.agent import Agent
from max_ai.environment import DockerEnvironment, EnvironmentManager
from max_ai.tools import BashTool
from max_ai.types.run_context import RunContext

# Keep this factory object stable; its identity separates configurations.
environment_factory = partial(DockerEnvironment, network="none", idle_timeout=300)

async def serve(client):
    async with EnvironmentManager() as manager:
        agent = Agent(
            name="documents", description="Workspace assistant",
            instructions="Work with files in $WORKSPACE using bash.",
            client=client, workspace=Workspace(),
            environment=environment_factory, environment_manager=manager,
            toolset=[BashTool()],
        )
        result = await agent.run(
            "List my files",
            run_context=RunContext(user_id="user_123", session_id="conversation_1"),
        )
        # Default Bash approval may pause the run; resume through normal approvals.
```

Agent obtains a conversation lease and injects its Environment into ToolContext.
BashTool reads it per call; it never mutates shared tool configuration. A directly
bound BashTool(environment=env) remains supported for manually managed callers.
Skills do not silently register Bash or approve it automatically.

The manager reuses environments by resolved workspace root, factory identity,
user and conversation. Configuration is immutable for a given factory identity;
use a new factory object when changing image/network settings. The manager must
live in the same application/event loop across turns to provide reuse. Each Agent
serializes its own turns because its prompt/reasoning state is mutable. A shared
manager additionally serializes leases for the same key across Agent instances.

Containers start on the first Bash call. Every managed turn releases its lease in
finally, including approval pauses, failures and cancellation. Docker's release
renews the idle grace period; command cancellation still stops the container.
Text-only turns never invoke Docker, including at shutdown. Lightweight manager
entries remain cached until close even if an idle container has expired.

Without an explicit manager, Agent owns one: call await agent.close() at application
shutdown. With a shared manager, the application owns await manager.close() (or
its async context manager). The Web UI closes its agents' managers in its lifespan
shutdown. Stop/cancel active runs before closing; manager.close waits for leases.
For manually consumed streams, use contextlib.aclosing if you stop iteration early.
Factories/managers are runtime configuration and are not persisted in checkpoints;
reapply them when reconstructing an agent from serialized configuration.

The model supplies `command`, `action`, and `description`. Actions are display
intent only: read_file, search, list_directory, write_file, edit_file, run_skill,
or execute. They never grant permissions or establish that a file was changed.
The normal tool approval policy still applies (ASK_APPROVED by default).
No host-shell fallback or legacy read_skill pseudo-command is provided.

The tool emits bash_started, bash_finished, bash_failed and bash_cancelled through
ToolContext.emit_event. The executor forwards these into the existing event stream,
including parallel calls, using tool_call_id for correlation. Model tool messages
remain in call order. The UI displays declared intent and observed command status.
Exit code != 0 is a completed command, not an infrastructure failure. Timeout is
reported on bash_finished; outer executor cancellation can produce bash_cancelled.
No file-read/write or skill-completion events are inferred from shell text.
The old implementation is retained in `max_ai/tools/bash copy.py` for reference.
