"""Live, one-off check that ModalExecutor actually works end-to-end.

Not part of the test suite (needs a real Modal account). Run after
`modal setup`:

    .venv/bin/python -m examples.verify_modal_executor

Uses the executor's default image (built and cached by Modal the first
time), starts a sandbox, runs a shell command in it as a non-root user, and
confirms the network is blocked.
"""

from __future__ import annotations

import asyncio

from max_ai.capabilities.workspace import LocalWorkspace


async def main() -> None:
    from max_ai.capabilities.executor.modal import ModalExecutor

    executor = ModalExecutor(network="none", lifetime=120)
    workspace = LocalWorkspace()
    session = await executor.connect(workspace, user_id="verify", conversation_id="c1")
    try:
        result = await executor.execute(session, "echo hello from Modal && id -u")
        print("stdout:", result.stdout.strip())
        assert "hello from Modal" in result.stdout

        blocked = await executor.execute(session, "curl -s -m 3 https://example.com; echo exit=$?")
        print("network check:", blocked.stdout.strip())
    finally:
        await executor.clean(session)
    print("ModalExecutor OK")


if __name__ == "__main__":
    asyncio.run(main())
