"""Docker sandbox command runner.

This executor runs tool calls inside ephemeral Docker containers created
from the configured sandbox image.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path

from ...config import setting
from ...base.executor import CoreExecutor
from ...base.tools import CoreTool, ToolContext
from ...termination import CancellationToken
from ...types.tool_call import ToolCallRecord, ToolResult
from ...types.tools import DockerToolRef


class DockerExecutor(CoreExecutor):
    """Run tools inside the Docker sandbox image."""

    def __init__(
        self,
        user_id: str,
        tool_source: str | Path | None = None,
        image: str = "maxai-sandbox:py311",
        default_timeout: int = 600,
        server_workspace: str | Path | None = None,
    ) -> None:
        super().__init__(
            default_timeout=default_timeout,
            server_workspace=server_workspace,
        )

        self.image = image
        self.user_id = self._clean_user_id(user_id)
        self.container_workspace = setting.sandbox_name
        self.container_tools_dir = self._join_container_path(
            self.container_workspace,
            "tools",
        )
        self.tool_source = self._resolve_tool_source(tool_source)

        self.docker_bin = "docker"
        self.get_or_create_tmp_dir()

    def get_or_create_tmp_dir(self) -> Path:
        """Create the user-scoped runtime workspace."""
        self.user_tmp_dir = self.server_workspace / "tmp" / self.user_id
        self.user_skills_dir = self.user_tmp_dir / "skills"
        self.user_workspace_dir = self.user_tmp_dir / "workspace"
        self.user_tools_dir = self.user_tmp_dir / "tools"

        self.user_skills_dir.mkdir(parents=True, exist_ok=True)
        self.user_workspace_dir.mkdir(parents=True, exist_ok=True)
        self.user_tools_dir.mkdir(parents=True, exist_ok=True)

        return self.user_tmp_dir

    async def run(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_context: ToolContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        timeout = tool.timeout_seconds or self.default_timeout
        proc: asyncio.subprocess.Process | None = None

        try:
            await self._ensure_connected()

            tool_ref = tool.docker_ref()
            self._sync_tool_source()
            payload = json.dumps(
                self._invocation_payload(tool_ref, record, tool_context),
                ensure_ascii=True,
            )

            proc = await asyncio.create_subprocess_exec(
                *self._docker_command(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            task = asyncio.create_task(proc.communicate(payload.encode("utf-8")))
            if cancellation_token is not None:
                cancellation_token.link_future(task)

            stdout_b, stderr_b = await asyncio.wait_for(task, timeout=timeout)
            stdout = stdout_b.decode(errors="replace")
            stderr = stderr_b.decode(errors="replace")

            result = self._tool_result_from_stdout(stdout)
            if result is not None:
                return result

            if proc.returncode != 0:
                detail = stderr.strip() or stdout.strip()
                msg = f"Docker worker exited with code {proc.returncode}."
                if detail:
                    msg = f"{msg} {detail}"
                return ToolResult.execution_error(record.id, msg)

            return ToolResult.execution_error(
                record.id,
                "Docker worker completed without returning a ToolResult on stdout.",
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

    async def connect(self) -> None:
        """Verify Docker is ready and the sandbox image exists."""
        await self._ensure_docker_available()
        await self._ensure_image_exists()

    async def disconnect(self) -> None:
        """No persistent Docker resources are held by this executor."""
        return None

    async def _ensure_docker_available(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin,
            "version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()

        if proc.returncode != 0:
            stderr = stderr_b.decode(errors="replace").strip()
            stdout = stdout_b.decode(errors="replace").strip()
            detail = stderr or stdout or "Docker is not available."
            raise RuntimeError(detail)

    async def _ensure_image_exists(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin,
            "image",
            "inspect",
            self.image,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()

        if proc.returncode == 0:
            return

        await self._build_image()

    async def _build_image(self) -> None:
        dockerfile = setting.get_or_create_docker_worker_dir() / setting.dockerfile_name
        build_context = Path(__file__).resolve().parents[3]

        proc = await asyncio.create_subprocess_exec(
            self.docker_bin,
            "build",
            "-f",
            str(dockerfile),
            "-t",
            self.image,
            str(build_context),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()

        if proc.returncode != 0:
            stderr = stderr_b.decode(errors="replace").strip()
            stdout = stdout_b.decode(errors="replace").strip()
            detail = stderr or stdout or f"Failed to build Docker image {self.image}."
            raise RuntimeError(detail)

    @staticmethod
    def _clean_user_id(user_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", user_id):
            raise ValueError(
                "user_id must contain only letters, numbers, underscore, dot, or dash."
            )
        return user_id

    @staticmethod
    def _resolve_tool_source(tool_source: str | Path | None) -> Path | None:
        if tool_source is None:
            return None

        path = Path(tool_source).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"tool_source does not exist: {path}")
        if not (path.is_file() or path.is_dir()):
            raise ValueError("tool_source must be a Python file or directory.")
        if path.is_file() and path.suffix != ".py":
            raise ValueError("tool_source file must be a .py file.")
        return path

    def _invocation_payload(
        self,
        tool_ref: DockerToolRef,
        record: ToolCallRecord,
        tool_context: ToolContext,
    ) -> dict[str, object]:
        return {
            "tool_ref": tool_ref.model_dump(mode="json"),
            "record": record.model_dump(mode="json"),
            "context": {
                "run_id": tool_context.run_id,
                "session_id": tool_context.session_id,
                "retry_count": tool_context.retry_count,
                "deps": tool_context.deps,
            },
            "tool_sources": self._tool_sources_payload(),
        }

    def _docker_command(self) -> list[str]:
        command = [
            self.docker_bin,
            "run",
            "--rm",
            "-i",
            "-v",
            f"{self.server_workspace}:{self.container_workspace}",
            "-w",
            self.container_workspace,
            "-e",
            f"MAX_AI_TOOL_SOURCE_DIR={self._container_tool_source_dir()}",
        ]

        command.extend(
            [
                self.image,
                "python",
                "-m",
                "max_ai.executor.docker.worker",
                "-",
                "-",
            ]
        )
        return command

    def _container_tool_source_dir(self) -> str:
        return self._join_container_path(
            self.container_workspace,
            "tmp",
            self.user_id,
            "tools",
        )

    def _sync_tool_source(self) -> None:
        if self.tool_source is None:
            return

        self.user_tools_dir.mkdir(parents=True, exist_ok=True)
        if self.tool_source.is_file():
            shutil.copy2(self.tool_source, self.user_tools_dir / self.tool_source.name)
            return

        for child in self.tool_source.iterdir():
            target = self.user_tools_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)

    def _tool_sources_payload(self) -> dict[str, str]:
        if self.tool_source is None:
            return {}

        if self.tool_source.is_file():
            return {
                self.tool_source.name: self.tool_source.read_text(encoding="utf-8")
            }

        sources: dict[str, str] = {}
        for path in self.tool_source.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative_path = path.relative_to(self.tool_source).as_posix()
            sources[relative_path] = path.read_text(encoding="utf-8")
        return sources

    @staticmethod
    def _tool_result_from_stdout(stdout: str) -> ToolResult | None:
        for line in reversed(stdout.splitlines()):
            candidate = line.strip()
            if not candidate:
                continue
            try:
                return ToolResult.model_validate_json(candidate)
            except Exception:
                continue
        return None

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
    def _join_container_path(*parts: str) -> str:
        first, *rest = parts
        joined = "/" + first.strip("/")
        for part in rest:
            joined += "/" + part.strip("/")
        if joined == "/":
            raise ValueError("container path cannot be root.")
        return joined
