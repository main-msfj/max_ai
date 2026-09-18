"""Small Docker CLI environment; requires a local Docker daemon and image."""

import asyncio
import math
import os
from uuid import uuid4
import weakref

from .environment import Environment, ExecutionResult
from ...base.workspace import Workspace
from ...termination import CancellationToken


class DockerEnvironment(Environment):
    """A persistent container with a user-only bind mount at /workspace.

    Files persist on the host; shell variables/cwd do not persist between execs.
    Skills are editable. Network is disabled. Timeout or cancellation removes
    the container (including descendants); call start again before further use.
    The image must contain bash and sleep. Image building/pulling is explicit.
    """

    def __init__(
        self,
        workspace: Workspace,
        user_id: str,
        conversation_id: str,
        *,
        image: str = "maxai-environment:latest",
        max_output_bytes: int = 65536,
    ):
        super().__init__(workspace, user_id, conversation_id)
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.image = image
        self.max_output_bytes = max_output_bytes
        self._name = f"maxai-env-{uuid4().hex}"
        self._started = False
        self._lock = asyncio.Lock()

    @property
    def variables(self) -> dict[str, str]:
        return {
            "WORKSPACE": "/workspace",
        }

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            directory = self.workspace.materialize(self.user_id, self.conversation_id)
            # Docker --mount uses commas as separators; reject ambiguous paths.
            if "," in str(directory.root):
                raise ValueError("Docker workspace paths cannot contain commas")
            args = [
                "run",
                "--detach",
                "--pull=never",
                "--name",
                self._name,
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=128",
                "--memory=512m",
                "--cpus=1",
                "--init",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=64m",
                "--mount",
                f"type=bind,src={directory.root},dst=/workspace",
                "--workdir",
                self.variables["WORKSPACE"],
            ]
            for key, value in self.variables.items():
                args.extend(["--env", f"{key}={value}"])
            args.extend(["--entrypoint", "sleep", self.image, "infinity"])
            try:
                result = await self._docker(*args, timeout=30)
                if result.exit_code != 0:
                    raise RuntimeError(result.stderr or "Docker startup failed")
                self._started = True
            except BaseException:
                await self.stop()
                raise

    async def execute(
        self,
        command: str,
        *,
        timeout: float = 60,
        cancellation_token: CancellationToken | None = None,
    ) -> ExecutionResult:
        if cancellation_token is not None and cancellation_token.is_cancelled():
            raise asyncio.CancelledError
        task = asyncio.create_task(self._execute(command, timeout=timeout))
        if cancellation_token is not None:
            loop = asyncio.get_running_loop()
            task_ref = weakref.ref(task)

            def cancel_command():
                pending = task_ref()
                if pending is not None and not pending.done() and not loop.is_closed():
                    loop.call_soon_threadsafe(pending.cancel)

            cancellation_token.add_callback(cancel_command)
        return await task

    async def _execute(self, command: str, *, timeout: float) -> ExecutionResult:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command cannot be empty")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        async with self._lock:
            if not self._started:
                raise RuntimeError("Call start() before execute()")
            try:
                return await self._docker(
                    "exec",
                    "--workdir",
                    self.variables["WORKSPACE"],
                    self._name,
                    "bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    command,
                    timeout=timeout,
                )
            except TimeoutError:
                await self.stop()
                return ExecutionResult("", "Command timed out", None, timed_out=True)
            except asyncio.CancelledError:
                await self.stop()
                raise

    async def stop(self) -> None:
        cleanup = asyncio.create_task(self._remove_container())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    async def _remove_container(self) -> None:
        # Remove only this instance, never containers selected by a prefix.
        result = await self._docker("rm", "--force", self._name, timeout=15)
        if result.exit_code != 0 and "No such container" not in result.stderr:
            raise RuntimeError(result.stderr or "Docker cleanup failed")
        self._started = False

    async def _docker(self, *args: str, timeout: float) -> ExecutionResult:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def capture(stream):
            data = bytearray()
            truncated = False
            while chunk := await stream.read(8192):
                remaining = self.max_output_bytes - len(data)
                data.extend(chunk[:remaining])
                truncated |= len(chunk) > remaining
            return data.decode("utf-8", errors="replace"), truncated

        readers = [
            asyncio.create_task(capture(process.stdout)),
            asyncio.create_task(capture(process.stderr)),
        ]
        try:
            async with asyncio.timeout(timeout):
                await process.wait()
                stdout, stderr = await asyncio.gather(*readers)
            return ExecutionResult(
                stdout[0],
                stderr[0],
                process.returncode,
                truncated=stdout[1] or stderr[1],
            )
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
