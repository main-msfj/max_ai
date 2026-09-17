# Workspace files and artifact synchronization

The workspace keeps editable files in a user-scoped tree:

```text
<workspace-root>/<user>/<session>/<path>
```

File tools read that tree directly. A managed workspace also synchronizes these
files with an `ArtifactStore`, so a tool write becomes a user artifact and can
be recovered when an agent or workspace object is recreated. The default
backend is SQLite under `<workspace-root>/.artifacts`; SQLite retains committed
revisions. A workspace may instead receive an explicit store:

```python
from max_ai.capabilities.workspace.local import WorkspaceLocal
from max_ai.workspace.artifacts import LocalArtifactStore

workspace = WorkspaceLocal(
    root="/srv/maxai/workspaces",
    artifact_store=LocalArtifactStore("/srv/maxai/artifacts"),
)
```

`WorkspaceLocal` serializes built-in local store configuration as its root.
Custom store objects are rejected during component serialization instead of
being silently dropped. Workspace refresh returns per-path state, revision,
SHA-256, and any safe error message. States such as `synced`, `pending`,
`conflict`, and `error` indicate whether a file reached the store. Refresh runs
at the start and end of an agent turn and before each file-tool entry. Newly
created and edited text files attempt synchronization immediately and include
the result in their `sync` field. Binary imports are limited to 8 MiB and are
create-only.

## Azure Blob Storage

Azure support is optional. Install it with:

```bash
uv sync --extra artifacts-azure
```

Create the store from `max_ai.workspace.azure_artifacts`. It uses the existing
container and resolves credentials lazily through an explicitly supplied
credential or Azure's `DefaultAzureCredential` chain:

```python
from max_ai.capabilities.workspace.local import WorkspaceLocal
from max_ai.workspace.azure_artifacts import AzureBlobArtifactStore

store = AzureBlobArtifactStore(
    account_url="https://example.blob.core.windows.net",
    container="agent-artifacts",
    prefix="maxai",
)
workspace = WorkspaceLocal(root="/srv/maxai/workspaces", artifact_store=store)
```

Serialized Azure configuration includes only `account_url`, `container`, and
`prefix`; it does not store credentials. A recreated workspace uses the
Azure SDK default credential chain. Explicit credential objects are not
serializable. No live Azure account is required for the local test suite.
Azure synchronization uses the current blob ETag for conditional updates;
access to historical revisions depends on Azure container versioning being
enabled separately.

## Legacy artifact helpers

`save_or_upload(user, file, session_id=None)` imports bytes to
`legacy/<filename>` when no session is supplied, and returns the user-relative
path. Passing a session stores the file beneath that session. Files must be at
most 8 MiB. `get_or_download` maps a bare filename to `legacy/<filename>`; a
user-relative path can refer to another session. `list_files` lists the user's
workspace files, and `get_artifacts_dir(user, session_id=None)` returns the
session directory, defaulting to `legacy`.

These helpers return local `Path` values for developer code. Their paths are
trusted local API values; use the descriptor-backed file tools when handling
untrusted model-supplied paths.
