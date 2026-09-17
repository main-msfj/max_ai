"""Sandbox command supervisor: bounded output and cooperative cancellation.

The cancellation marker stops the owned process group without destroying the
sandbox filesystem before the manager downloads workspace changes.
"""

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ..base.environment import ExecutionResult
from ..termination import CancellationToken
from .process import run_process


async def main():
    payload = json.load(sys.stdin)
    marker = Path("/tmp") / ("maxai-cancel-" + payload["id"])
    token = CancellationToken()

    async def watch():
        while not marker.exists():
            await asyncio.sleep(0.1)
        token.cancel()

    watcher = asyncio.create_task(watch())
    try:
        result = await run_process(
            payload["argv"], stdin=payload.get("stdin"), timeout=payload["timeout"],
            max_output_bytes=payload["max_output_bytes"], cancellation_token=token,
        )
    except asyncio.CancelledError:
        result = ExecutionResult("", "Command cancelled", None)
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        marker.unlink(missing_ok=True)
    print(json.dumps(asdict(result)), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
