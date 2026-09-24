"""Bounded subprocess execution for runtime providers."""

from __future__ import annotations

import asyncio
import math
import os
import signal
import weakref
from collections.abc import Sequence

from ...base.executor import ExecutionResult
from ..termination import CancellationToken


async def run_process(
    argv: Sequence[str], *, cwd: str | None = None,
    env: dict[str, str] | None = None, stdin: str | None = None,
    timeout: float = 60, max_output_bytes: int = 1 << 20,
    cancellation_token: CancellationToken | None = None,
) -> ExecutionResult:
    if not argv:
        raise ValueError("argv cannot be empty")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be positive")
    if cancellation_token is not None and cancellation_token.is_cancelled():
        raise asyncio.CancelledError

    process = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, env=env, stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=(os.name == "posix"),
    )

    async def capture(stream):
        data = bytearray()
        truncated = False
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                break
            remaining = max_output_bytes - len(data)
            if remaining > 0:
                data.extend(chunk[:remaining])
            truncated |= len(chunk) > max(remaining, 0)
        return data.decode("utf-8", errors="replace"), truncated

    async def feed():
        if stdin is None:
            return
        try:
            try:
                process.stdin.write(stdin.encode())
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            try:
                process.stdin.close()
                await process.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass

    readers = [asyncio.create_task(capture(process.stdout)), asyncio.create_task(capture(process.stderr))]
    feeder = asyncio.create_task(feed())
    cancelled = False

    process_ref = weakref.ref(process)
    loop = asyncio.get_running_loop()
    active = True

    def cancel():
        nonlocal cancelled
        cancelled = True
        def terminate_active():
            if not active:
                return
            owned = process_ref()
            if owned is not None:
                _terminate(owned)
        if not loop.is_closed():
            loop.call_soon_threadsafe(terminate_active)

    if cancellation_token is not None:
        cancellation_token.add_callback(cancel)
    try:
        try:
            async with asyncio.timeout(timeout):
                await process.wait()
                await feeder
                stdout, stderr = await asyncio.gather(*readers)
        except TimeoutError:
            _terminate(process)
            await process.wait()
            await asyncio.gather(feeder, return_exceptions=True)
            await asyncio.gather(*readers, return_exceptions=True)
            return ExecutionResult("", "Command timed out", None, timed_out=True)
        except asyncio.CancelledError:
            _terminate(process)
            await process.wait()
            await asyncio.gather(feeder, *readers, return_exceptions=True)
            raise
        if cancelled:
            raise asyncio.CancelledError
        return ExecutionResult(stdout[0], stderr[0], process.returncode,
                               truncated=stdout[1] or stderr[1])
    except BaseException:
        _terminate(process)
        if process.returncode is None:
            await process.wait()
        raise
    finally:
        active = False
        for task in (feeder, *readers):
            if not task.done():
                task.cancel()
        await asyncio.gather(feeder, *readers, return_exceptions=True)


def _terminate(process: asyncio.subprocess.Process) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
