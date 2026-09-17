# v2 migration contract tests

Run with:

```bash
uv run pytest -q tests/v2 --tb=short
```

The tests cover the migrated workspace user, skills, and conversation layout;
execution copy snapshots, binary changes, publication conflicts, unrelated edit
preservation, discard, and symlink or special-file rejection; environment
working-copy retention, reconnect, cancellation, explicit publication, close,
and sync-failure recovery; registry metadata and host/provider dispatch;
approval state and Bash parsing/events; skill materialization; a mocked Agent
model/tool loop; LocalExecutor process bounds; and Docker/Modal configuration
plus a mocked Docker startup argv contract.

Provider tests use mocks or constructor validation only. They do not require
network access, credentials, Docker, or Modal. They intentionally do not test
provider service startup, Modal SDK calls, remote sync protocols in full,
persistence recovery after process restart, or concurrency across manager
processes. These are initial migration contracts, not exhaustive coverage of
every tool, guardrail, or failure mode.

Validation on 2026-09-15: 40 tests passed. Luna wrote the tests using low
reasoning effort; the parent agent reviewed and ran them. Execution used the
existing environment (`uv run --no-sync pytest -q tests/v2 --tb=short`). The
restricted sandbox stalled on a local subprocess; the complete run outside
that sandbox passed. No legacy tests or production files were changed.
