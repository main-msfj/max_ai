"""Docker sandbox command runner.

This executor is intentionally focused on the sandbox image: it runs a
single shell command inside ``maxai-sandbox:py311`` with
``server_workspace`` mounted at ``/server_workspace``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ..base.executor import CoreExecutor
from ..base.tools import CoreTool, ToolContext
from ..termination import CancellationToken
from ..types.tool_call import ToolCallRecord, ToolResult


class DockerExecutor(CoreExecutor):
    """Run bash-style tools inside the Docker sandbox image.

    The first supported shape is a tool call with a ``command`` argument.
    That matches ``skill_bash`` and keeps Docker mode centered on the
    sandbox instead of a Python worker protocol.
    """

    def __init__(
        self,
        image: str = "maxai-sandbox:py311",
        default_timeout: int = 600,
        server_workspace: str | Path | None = None,
        container_workspace: str = "/server_workspace",
        sessions_subdir: str = "tmp/session",
        skills_cache_subdir: str = "var/skills-cache",
        docker_bin: str = "docker",
        extra_env: dict[str, str] | None = None,
        extra_volumes: list[tuple[str | Path, str, str]] | None = None,
    ) -> None:
        super().__init__(default_timeout=default_timeout)
        self.image = image
        self.server_workspace = Path(
            server_workspace or Path(os.getcwd()) / "server_workspace"
        ).expanduser().resolve()
        self.container_workspace = self._clean_container_path(container_workspace)
        self.sessions_subdir = self._clean_relative_path(sessions_subdir)
        self.skills_cache_subdir = self._clean_relative_path(skills_cache_subdir)
        self.docker_bin = docker_bin
        self.extra_env = dict(extra_env or {})
        self.extra_volumes = list(extra_volumes or [])

    async def run(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_context: ToolContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        timeout = tool.timeout_seconds or self.default_timeout

        command = record.parameters.get("command")
        if not isinstance(command, str) or not command:
            return ToolResult.execution_error(
                record.id,
                "DockerExecutor currently supports tools with a non-empty "
                "'command' argument.",
            )

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._docker_command(command, tool_context),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            task = asyncio.create_task(proc.communicate())
            if cancellation_token is not None:
                cancellation_token.link_future(task)

            stdout_b, stderr_b = await asyncio.wait_for(task, timeout=timeout)
            stdout = stdout_b.decode(errors="replace")
            stderr = stderr_b.decode(errors="replace")

            if proc.returncode != 0:
                return ToolResult.execution_error(
                    record.id,
                    f"Docker command exited with code {proc.returncode}.",
                )

            return ToolResult.success_result(
                record.id,
                {
                    "exit_code": proc.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                },
                {"executor": "docker", "image": self.image},
            )

        except asyncio.TimeoutError:
            if proc is not None:
                await self._kill_process(proc)
            return ToolResult.timeout(record.id, timeout_seconds=timeout)

        except asyncio.CancelledError:
            if proc is not None:
                await self._kill_process(proc)
            return ToolResult.cancelled_during_execution(record.id)

        except Exception as e:
            return ToolResult.execution_error(record.id, str(e))

    def _docker_command(self, shell_command: str, tool_context: ToolContext) -> list[str]:
        docker_command = [
            self.docker_bin,
            "run",
            "--rm",
            "-v",
            f"{self.server_workspace}:{self.container_workspace}",
            "-w",
            self.container_workspace,
        ]

        env = self._container_env(tool_context)
        for key, value in env.items():
            docker_command.extend(["-e", f"{key}={value}"])

        for host, container, mode in self._container_volumes():
            docker_command.extend(["-v", f"{host}:{container}:{mode}"])

        docker_command.extend([self.image, "/bin/bash", "-lc", shell_command])
        return docker_command

    def _container_env(self, tool_context: ToolContext) -> dict[str, str]:
        sessions_dir = self._join_container_path(
            self.container_workspace,
            self.sessions_subdir,
        )
        skills_cache_dir = self._join_container_path(
            self.container_workspace,
            self.skills_cache_subdir,
        )

        env: dict[str, str] = {
            "SERVER_WORKSPACE": self.container_workspace,
            "SESSIONS_DIR": sessions_dir,
            "SKILLS_CACHE_DIR": skills_cache_dir,
            "RUN_ID": tool_context.run_id,
            "SESSION_ID": tool_context.session_id,
        }
        if tool_context.session_id:
            env["SKILLS_DIR"] = self._join_container_path(
                sessions_dir,
                tool_context.session_id,
                "skills",
            )
        env.update(self.extra_env)
        return env

    def _container_volumes(self) -> list[tuple[str, str, str]]:
        volumes: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add(host: str | Path, container: str, mode: str) -> None:
            host_path = str(Path(host).expanduser().resolve())
            key = (host_path, container)
            if key in seen:
                return
            seen.add(key)
            volumes.append((host_path, container, mode))

        for host, container, mode in self.extra_volumes:
            add(host, container, mode)

        return volumes

    @staticmethod
    async def _kill_process(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            return
        await proc.communicate()

    @staticmethod
    def _clean_container_path(path: str) -> str:
        cleaned = "/" + path.strip("/")
        if cleaned == "/":
            raise ValueError("container_workspace cannot be root.")
        return cleaned

    @staticmethod
    def _clean_relative_path(path: str) -> str:
        cleaned = path.strip("/")
        if not cleaned or cleaned.startswith("..") or "/../" in f"/{cleaned}/":
            raise ValueError(f"Invalid relative container path: {path!r}")
        return cleaned

    @staticmethod
    def _join_container_path(*parts: str) -> str:
        first, *rest = parts
        joined = first.rstrip("/")
        for part in rest:
            joined += "/" + part.strip("/")
        return joined
