"""Docker sandbox command runner.

This executor runs tool calls inside ephemeral Docker containers created
from the configured sandbox image.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from ...config import setting
from ...base.executor import CoreExecutor
from ...base.tools import CoreTool, ToolContext
from ...capabilities.workspace import LocalWorkSpaceRegistry
from ...termination import CancellationToken
from ...types.tool_call import ToolCallRecord, ToolResult
from ...types.tools import DockerToolRef


@dataclass
class _ComposeSession:
    env: dict[str, str]
    last_used_at: float


class DockerExecutor(CoreExecutor):
    """Run tools inside the Docker sandbox image."""

    def __init__(
        self,
        tool_source: str | Path | None = None,
        image: str = "maxai-sandbox:py311",
        default_timeout: int = 600,
        server_workspace: str | Path | None = None,
        container_workspace: str = "/sandbox",
        workspace_registry: object | None = None,
        compose_file: str | Path | None = None,
        bash_ttl_seconds: float = 120,
        cleanup_orphaned_projects: bool = True,
    ) -> None:
        super().__init__(default_timeout=default_timeout)

        self.image = image
        self.container_workspace = container_workspace
        self.workspace_registry = workspace_registry or LocalWorkSpaceRegistry(
            root=server_workspace
        )
        self.server_workspace = self.workspace_registry.base_root
        self.tool_source = self._resolve_tool_source(tool_source)
        self.compose_file = (
            Path(compose_file).expanduser().resolve()
            if compose_file is not None
            else Path(__file__).with_name("docker-compose.yml")
        )
        self.bash_ttl_seconds = bash_ttl_seconds
        self.cleanup_orphaned_projects = cleanup_orphaned_projects
        self._compose_sessions: dict[str, _ComposeSession] = {}

        self.docker_bin = "docker"

    def get_or_create_tmp_dir(self, user_id: str) -> Path:
        """Create the user-scoped runtime workspace."""
        clean_user_id = self._clean_user_id(user_id)
        return self.workspace_registry.materialize(clean_user_id).root

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
            await self._prune_expired_sessions()
            self.get_or_create_tmp_dir(tool_context.user_id)
            self._normalize_host_runtime_layout(tool_context.user_id)

            compose_env = self._compose_env(tool_context)
            await self._prune_orphaned_projects(tool_context, compose_env)
            await self._compose_up(tool_context, compose_env)
            self._remember_persistent_session_if_needed(tool, tool_context, compose_env)

            if tool.name == "bash":
                return await self._run_bash_direct(
                    tool,
                    record,
                    tool_context,
                    compose_env,
                    cancellation_token,
                )

            tool_ref = tool.docker_ref()
            self._sync_tool_source(tool_context.user_id)
            payload = json.dumps(
                self._invocation_payload(tool_ref, record, tool_context),
                ensure_ascii=True,
            )
            proc = await asyncio.create_subprocess_exec(
                *self._docker_command(tool_context),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=compose_env,
            )

            task = asyncio.create_task(proc.communicate(payload.encode("utf-8")))
            if cancellation_token is not None:
                cancellation_token.link_future(task)

            stdout_b, stderr_b = await asyncio.wait_for(task, timeout=timeout)
            stdout = stdout_b.decode(errors="replace")
            stderr = stderr_b.decode(errors="replace")

            result = self._tool_result_from_stdout(stdout)
            if result is not None:
                self._sync_workspace_from_result(tool_context.user_id, result)
                self._remember_persistent_session_if_needed(tool, tool_context, compose_env)
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

        finally:
            if not self._keeps_compose_alive(tool):
                try:
                    await self._compose_down(tool_context)
                except Exception:
                    pass

    async def connect(self) -> None:
        """Verify Docker is ready and the sandbox image exists."""
        await self._ensure_docker_available()
        await self._ensure_image_exists()

    async def disconnect(self) -> None:
        """Stop any compose runtimes kept alive for interactive tools."""
        await self._close_all_sessions()

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
        dockerfile = self._dockerfile_path()
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

    def _dockerfile_path(self) -> Path:
        dockerfile_name = getattr(setting, "dockerfile_name", "Dockerfile.sandbox")
        dockerfile = Path(dockerfile_name)
        if dockerfile.is_absolute():
            return dockerfile

        local_dockerfile = Path(__file__).with_name(dockerfile_name)
        if local_dockerfile.is_file():
            return local_dockerfile

        docker_dir_factory = getattr(setting, "get_or_create_docker_worker_dir", None)
        if callable(docker_dir_factory):
            return docker_dir_factory() / dockerfile_name

        return (
            setting.root_dir
            / "serverWorkspace"
            / "var"
            / "docker-cache"
            / dockerfile_name
        )

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
                "user_id": tool_context.user_id,
                "session_id": tool_context.session_id,
                "retry_count": tool_context.retry_count,
                "deps": self._container_context_deps(tool_context.deps),
            },
            "tool_sources": self._tool_sources_payload(),
            "runtime_files": self._runtime_files_payload(tool_context.user_id),
        }

    def _docker_command(self, tool_context: ToolContext) -> list[str]:
        command = self._compose_base_command(tool_context)
        command.extend(
            [
                "exec",
                "-T",
                "runtime",
                "python",
                "-m",
                "max_ai.executor.docker.worker",
                "-",
                "-",
            ]
        )
        return command

    def _bash_command(self, tool_context: ToolContext, shell_command: str) -> list[str]:
        command = self._compose_base_command(tool_context)
        command.extend(["exec", "-T", "runtime", "bash", "-lc", shell_command])
        return command

    async def _run_bash_direct(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_context: ToolContext,
        env: dict[str, str],
        cancellation_token: CancellationToken | None,
    ) -> ToolResult:
        validation = tool.validate_parameters(record)
        if not validation.is_tool_valid:
            return ToolResult.invalid_parameters(
                record.id,
                validation.msg_error or "Invalid bash parameters.",
            )

        shell_command = record.parameters.get("command")
        if not isinstance(shell_command, str) or not shell_command.strip():
            return ToolResult.invalid_parameters(record.id, "command cannot be empty.")

        timeout = record.parameters.get("timeout_seconds") or tool.timeout_seconds
        proc = await asyncio.create_subprocess_exec(
            *self._bash_command(tool_context, shell_command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        task = asyncio.create_task(proc.communicate())
        if cancellation_token is not None:
            cancellation_token.link_future(task)

        try:
            stdout_b, stderr_b = await asyncio.wait_for(task, timeout=float(timeout))
        except asyncio.TimeoutError:
            await self._kill_process(proc)
            return ToolResult.timeout(record.id, timeout_seconds=float(timeout))

        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        max_output_chars = int(getattr(tool, "max_output_chars", 20000))
        stdout, stdout_truncated = self._truncate_text(stdout, max_output_chars)
        stderr, stderr_truncated = self._truncate_text(stderr, max_output_chars)

        return ToolResult.success_result(
            record.id,
            {
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "cwd": self.container_workspace,
                "command": shell_command,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            },
            metadata={"name": tool.name, "executor": "docker"},
        )

    def _compose_base_command(self, tool_context: ToolContext) -> list[str]:
        return [
            self.docker_bin,
            "compose",
            "-f",
            str(self.compose_file),
            "-p",
            self._compose_project_name(tool_context),
        ]

    async def _compose_up(
        self,
        tool_context: ToolContext,
        env: dict[str, str],
    ) -> None:
        proc = await asyncio.create_subprocess_exec(
            *self._compose_up_command(tool_context),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout_b, stderr_b = await proc.communicate()
        if proc.returncode != 0:
            detail = stderr_b.decode(errors="replace").strip()
            if not detail:
                detail = stdout_b.decode(errors="replace").strip()
            raise RuntimeError(detail or "Docker compose failed to start runtime.")

    def _compose_up_command(self, tool_context: ToolContext) -> list[str]:
        command = self._compose_base_command(tool_context)
        command.extend(["up", "-d", "--force-recreate", "runtime"])
        return command

    async def _compose_down(self, tool_context: ToolContext) -> None:
        project_name = self._compose_project_name(tool_context)
        session = self._compose_sessions.get(project_name)
        env = session.env if session is not None else self._compose_env(tool_context)
        await self._compose_down_project(project_name, env)
        self._compose_sessions.pop(project_name, None)

    async def _compose_down_project(self, project_name: str, env: dict[str, str]) -> None:
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin,
            "compose",
            "-f",
            str(self.compose_file),
            "-p",
            project_name,
            "down",
            "--remove-orphans",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        await proc.communicate()

    def _compose_env(self, tool_context: ToolContext) -> dict[str, str]:
        clean_user_id = self._clean_user_id(tool_context.user_id)
        host_runtime = self.workspace_registry.materialize(clean_user_id).root
        env = os.environ.copy()
        env.update(
            {
                "MAXAI_RUNTIME_IMAGE": self.image,
                "MAXAI_HOST_RUNTIME_DIR": str(host_runtime),
                "MAXAI_CONTAINER_WORKSPACE": self.container_workspace,
            }
        )
        return env

    def _compose_project_name(self, tool_context: ToolContext) -> str:
        raw = tool_context.session_id or tool_context.run_id or tool_context.user_id
        value = re.sub(r"[^a-z0-9_-]+", "_", raw.lower()).strip("_-")
        return f"maxai_{value or 'runtime'}"

    def _keeps_compose_alive(self, tool: CoreTool) -> bool:
        return tool.name == "bash"

    def _remember_persistent_session_if_needed(
        self,
        tool: CoreTool,
        tool_context: ToolContext,
        env: dict[str, str],
    ) -> None:
        if not self._keeps_compose_alive(tool):
            return
        self._compose_sessions[self._compose_project_name(tool_context)] = _ComposeSession(
            env=env,
            last_used_at=time.monotonic(),
        )

    async def _prune_expired_sessions(self) -> None:
        if not self._compose_sessions:
            return
        now = time.monotonic()
        expired = [
            project_name
            for project_name, session in self._compose_sessions.items()
            if now - session.last_used_at >= self.bash_ttl_seconds
        ]
        for project_name in expired:
            session = self._compose_sessions.pop(project_name, None)
            if session is None:
                continue
            try:
                await self._compose_down_project(project_name, session.env)
            except Exception:
                pass

    async def _prune_orphaned_projects(
        self,
        tool_context: ToolContext,
        env: dict[str, str],
    ) -> None:
        """Stop MaxAI compose runtimes this process no longer owns.

        Bash sessions are intentionally kept alive for a short TTL, but if the
        app process restarts the in-memory session registry is empty. Docker
        then keeps the old compose projects around forever unless we discover
        and remove them on the next run.
        """
        if not self.cleanup_orphaned_projects:
            return

        current_project = self._compose_project_name(tool_context)
        tracked_projects = set(self._compose_sessions)
        for project_name in await self._list_compose_projects(env):
            if not project_name.startswith("maxai_"):
                continue
            if project_name == current_project or project_name in tracked_projects:
                continue
            try:
                await self._compose_down_project(project_name, env)
            except Exception:
                pass

    async def _list_compose_projects(self, env: dict[str, str]) -> list[str]:
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin,
            "compose",
            "ls",
            "--all",
            "--format",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        stdout_b, _ = await proc.communicate()
        if proc.returncode != 0:
            return []

        try:
            payload = json.loads(stdout_b.decode(errors="replace") or "[]")
        except json.JSONDecodeError:
            return []

        if not isinstance(payload, list):
            return []

        projects: list[str] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = item.get("Name") or item.get("name")
            if isinstance(name, str):
                projects.append(name)
        return projects

    async def _close_all_sessions(self) -> None:
        sessions = list(self._compose_sessions.items())
        self._compose_sessions.clear()
        for project_name, session in sessions:
            try:
                await self._compose_down_project(project_name, session.env)
            except Exception:
                pass

    def _container_tool_source_dir(self) -> str:
        return self._join_container_path(self.container_workspace, "tools")

    def _container_skills_dir(self) -> str:
        return self._join_container_path(self.container_workspace, "skills")

    def _container_artifacts_dir(self) -> str:
        return self._join_container_path(self.container_workspace, "artifacts")

    def _container_context_deps(self, deps: dict[str, object]) -> dict[str, object]:
        container_deps = dict(deps)
        container_deps.update(
            {
                "runtime_root": self.container_workspace,
                "tools_dir": self._container_tool_source_dir(),
                "skills_dir": self._container_skills_dir(),
                "artifacts_dir": self._container_artifacts_dir(),
            }
        )
        return container_deps

    def _sync_tool_source(self, user_id: str) -> None:
        if self.tool_source is None:
            return

        clean_user_id = self._clean_user_id(user_id)
        user_tools_dir = self.workspace_registry.materialize(clean_user_id).tool_dir
        user_tools_dir.mkdir(parents=True, exist_ok=True)
        if self.tool_source.is_file():
            shutil.copy2(self.tool_source, user_tools_dir / self.tool_source.name)
            return

        for child in self.tool_source.iterdir():
            target = user_tools_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)

    def _normalize_host_runtime_layout(self, user_id: str) -> None:
        clean_user_id = self._clean_user_id(user_id)
        directory = self.workspace_registry.materialize(clean_user_id)
        runtime_root = directory.root.resolve()
        nested_root = (runtime_root / "tmp" / clean_user_id).resolve()
        try:
            nested_root.relative_to(runtime_root)
        except ValueError:
            return
        if not nested_root.exists():
            return

        targets = {
            "tools": directory.tool_dir,
            "skills": directory.skill_dir,
            "artifacts": directory.artifacts_dir,
        }
        for name, target in targets.items():
            source = nested_root / name
            if not source.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            for child in source.iterdir():
                destination = target / child.name
                if child.is_dir():
                    shutil.copytree(child, destination, dirs_exist_ok=True)
                elif child.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(child, destination)

        try:
            shutil.rmtree(runtime_root / "tmp")
        except OSError:
            pass

    def _tool_sources_payload(self) -> dict[str, str]:
        if self.tool_source is None:
            return {}

        if self.tool_source.is_file():
            return {self.tool_source.name: self.tool_source.read_text(encoding="utf-8")}

        sources: dict[str, str] = {}
        for path in self.tool_source.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative_path = path.relative_to(self.tool_source).as_posix()
            sources[relative_path] = path.read_text(encoding="utf-8")
        return sources

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> tuple[str, bool]:
        if len(value) <= max_chars:
            return value, False
        keep = max(max_chars, 0)
        return value[:keep] + "\n[output truncated]", True

    def _runtime_files_payload(self, user_id: str) -> dict[str, object]:
        """Send user runtime files to Docker for daemon-isolated filesystems.

        The bind mount is still the fast path when Docker shares the host
        filesystem. This payload makes skills/artifacts available even when
        the Docker daemon cannot see ``self.server_workspace``.
        """
        clean_user_id = self._clean_user_id(user_id)
        user_tmp_dir = self.workspace_registry.materialize(clean_user_id).root
        if not user_tmp_dir.exists():
            return {}

        files: dict[str, object] = {}
        for root_name in ("skills", "artifacts"):
            root = user_tmp_dir / root_name
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                relative_path = path.relative_to(user_tmp_dir).as_posix()
                files[relative_path] = self._file_payload(path)
        return files

    def _sync_workspace_from_result(self, user_id: str, result: ToolResult) -> None:
        files = result.metadata.get("workspace_files")
        if not isinstance(files, dict):
            return

        clean_user_id = self._clean_user_id(user_id)
        artifacts_dir = self.workspace_registry.materialize(clean_user_id).artifacts_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        for relative_path, payload in files.items():
            if not isinstance(relative_path, str):
                continue
            target = (artifacts_dir / relative_path).resolve()
            try:
                target.relative_to(artifacts_dir.resolve())
            except ValueError:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            data = self._payload_bytes(payload)
            if data is None:
                continue
            target.write_bytes(data)

    @staticmethod
    def _file_payload(path: Path) -> dict[str, str]:
        data = path.read_bytes()
        try:
            return {"encoding": "text", "content": data.decode("utf-8")}
        except UnicodeDecodeError:
            return {
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
            }

    @staticmethod
    def _payload_bytes(payload: object) -> bytes | None:
        if isinstance(payload, str):
            return payload.encode("utf-8")
        if not isinstance(payload, dict):
            return None
        encoding = payload.get("encoding")
        content = payload.get("content")
        if not isinstance(content, str):
            return None
        if encoding == "text":
            return content.encode("utf-8")
        if encoding == "base64":
            try:
                return base64.b64decode(content.encode("ascii"), validate=True)
            except Exception:
                return None
        return None

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
