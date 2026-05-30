"""Docker executor for sandboxed tool execution.

This executor intentionally avoids Docker Compose. It uses the Docker CLI
directly so the runtime boundary stays easy to reason about:

Host workspace:
    <workspace_root>/<user_id>/tools
    <workspace_root>/<user_id>/skills
    <workspace_root>/<user_id>/artifacts

Container workspace:
    /mnt/tools
    /mnt/skills
    /mnt/artifacts

Docker never needs to know the user id. The executor resolves the host
runtime directory for the user, then bind-mounts that directory as /mnt.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import time
import typing as t
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ...config import setting
from ...base.executor import CoreExecutor
from ...base.tools import CoreTool, ToolContext
from ...termination import CancellationToken
from ...types.tool_call import ToolCallRecord, ToolResult


_READ_SKILL_RE = re.compile(r"^\s*read_skill\s+([A-Za-z0-9_-]+)\s*$")

# Marker the worker prints around its ToolResult JSON so we don't have to
# guess which stdout line is the result (see _tool_result_from_stdout).
_RESULT_BEGIN = "<<<MAXAI_TOOL_RESULT>>>"
_RESULT_END = "<<<END_MAXAI_TOOL_RESULT>>>"

# Reserved user ids that would resolve outside the workspace root.
_RESERVED_USER_IDS = {".", ".."}


@dataclass
class _BashSession:
    """A short-lived interactive-ish bash container."""

    name: str
    last_used_at: float


class DockerExecutor(CoreExecutor):
    """Run tools inside Docker with a small, explicit runtime contract.

    Non-bash tools run in one-shot containers and are removed as soon as the
    tool finishes. Bash runs in a short-lived container so skills can issue
    several commands against the same runtime state; idle bash containers are
    removed after ``bash_ttl_seconds``.
    """

    CONTAINER_WORKSPACE = setting.mtn_folder

    def __init__(
        self,
        image: str = "maxai-sandbox:py311",
        default_timeout: int = 600,
        bash_ttl_seconds: float = 180,
        force_build_image: bool = False,
        docker_bin: str = "docker",
        repo_root: str | Path | None = None,
        dockerfile: str | Path | None = None,
        tool_files: str | Path | t.Sequence[str | Path] | None = None,
        # Resource / hardening knobs (None disables the corresponding flag).
        container_user: str | None = "1000:1000",
        memory_limit: str | None = "512m",
        cpu_limit: str | None = "1",
        pids_limit: int | None = 256,
    ) -> None:
        super().__init__(default_timeout=default_timeout)
        self.image = image
        self.bash_ttl_seconds = bash_ttl_seconds
        self.force_build_image = force_build_image
        self.docker_bin = docker_bin
        self.repo_root = (
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[3]
        )
        self.dockerfile = (
            Path(dockerfile).expanduser().resolve()
            if dockerfile is not None
            else Path(__file__).with_name("Dockerfile.sandbox")
        )
        self.tool_files = self._normalize_tool_files(tool_files)
        self.container_user = container_user
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit

        self.workspace_root: Path | None = None
        self.app_stage_dir: Path | None = None
        self._bash_sessions: dict[str, _BashSession] = {}

        # Concurrency guards.
        self._app_stage_lock = asyncio.Lock()
        self._app_staged = False
        self._bash_session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def bind_to_workspace(self, workspace_registry_root: str | Path) -> None:
        """Bind the host workspace root used to resolve user runtimes."""
        self.workspace_root = Path(workspace_registry_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    async def connect(self) -> None:
        """Verify Docker is available and ensure the sandbox image exists."""
        await self._ensure_docker_available()
        if self.force_build_image or not await self._image_exists():
            await self._build_image()
        # Stage /app once up front so per-call runs never mutate the shared
        # mount directory underneath live containers (see _ensure_app_mount).
        await self._ensure_app_mount()

    async def disconnect(self) -> None:
        """Remove bash containers and the staged /app mount."""
        await self._close_all_bash_sessions()
        self._cleanup_app_stage()

    async def run(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_context: ToolContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        """Execute one tool call inside Docker and always return ToolResult."""
        timeout = tool.timeout_seconds or self.default_timeout
        proc: asyncio.subprocess.Process | None = None
        container_name: str | None = None

        try:
            await self._ensure_connected()
            await self._ensure_app_mount()
            await self._prune_expired_bash_sessions()
            host_runtime = self._host_runtime(tool_context.user_id)
            self._ensure_runtime_layout(host_runtime)

            if tool.name in {"bash", "skill_bash"}:
                return await self._run_bash_direct(
                    tool=tool,
                    record=record,
                    tool_context=tool_context,
                    host_runtime=host_runtime,
                    cancellation_token=cancellation_token,
                )

            payload = json.dumps(
                self._invocation_payload(tool, record, tool_context),
                ensure_ascii=True,
            )
            # Named one-shot container so we can force-remove it if the call
            # times out or is cancelled (--rm only fires when it exits clean).
            container_name = f"maxai-run-{uuid.uuid4().hex[:16]}"
            proc = await asyncio.create_subprocess_exec(
                *self._docker_run_once_command(host_runtime, container_name),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )

            task = asyncio.create_task(proc.communicate(payload.encode("utf-8")))
            if cancellation_token is not None:
                cancellation_token.link_future(task)

            stdout_b, stderr_b = await asyncio.wait_for(task, timeout=float(timeout))
            stdout = stdout_b.decode(errors="replace")
            stderr = stderr_b.decode(errors="replace")

            result = self._tool_result_from_stdout(stdout)
            if result is not None:
                return result

            detail = stderr.strip() or stdout.strip()
            msg = "Docker worker completed without returning a ToolResult."
            if detail:
                msg = f"{msg} {detail}"
            return ToolResult.execution_error(record.id, msg)

        except asyncio.TimeoutError:
            if proc is not None:
                await self._kill_process(proc)
            if container_name is not None:
                await self._remove_container(container_name)
            return ToolResult.timeout(record.id, timeout_seconds=float(timeout))

        except asyncio.CancelledError:
            if proc is not None:
                await self._kill_process(proc)
            if container_name is not None:
                await self._remove_container(container_name)
            return ToolResult.cancelled_during_execution(record.id)

        except Exception as exc:
            if container_name is not None:
                await self._remove_container(container_name)
            return ToolResult.execution_error(record.id, str(exc))

    async def _ensure_docker_available(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin,
            "version",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_b = await proc.communicate()
        if proc.returncode != 0:
            detail = stderr_b.decode(errors="replace").strip()
            raise RuntimeError(detail or "Docker is not available.")

    async def _image_exists(self) -> bool:
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin,
            "image",
            "inspect",
            self.image,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def _build_image(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin,
            "build",
            "-f",
            str(self.dockerfile),
            "-t",
            self.image,
            str(self.repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        if proc.returncode == 0:
            return

        detail = stderr_b.decode(errors="replace").strip()
        if not detail:
            detail = stdout_b.decode(errors="replace").strip()
        raise RuntimeError(detail or f"Failed to build Docker image {self.image}.")

    def _host_runtime(self, user_id: str) -> Path:
        root = (self.workspace_root or setting.root_dir).expanduser().resolve()
        clean_user_id = self._clean_user_id(user_id)
        candidate = (root / clean_user_id).expanduser().resolve()
        # Defense in depth: even with the charset restriction, make sure the
        # resolved path can never escape the workspace root.
        if not candidate.is_relative_to(root):
            raise ValueError("resolved runtime path escapes workspace root.")
        return candidate

    def _ensure_runtime_layout(self, host_runtime: Path) -> None:
        for name in (setting.tool_dir, setting.skill_dir, setting.artifacts_dir):
            (host_runtime / name).mkdir(parents=True, exist_ok=True)

    def _docker_env_args(self) -> list[str]:
        runtime_root = self.CONTAINER_WORKSPACE
        return [
            "-e",
            "PYTHONPATH=/app",
            "-e",
            "MAX_AI_TOOL_SOURCE_DIR=/app/tools",
            "-e",
            f"RUNTIME_DIR={runtime_root}",
            "-e",
            f"TOOLS_DIR={self._container_path(setting.tool_dir)}",
            "-e",
            f"SKILLS_DIR={self._container_path(setting.skill_dir)}",
            "-e",
            f"ARTIFACTS_DIR={self._container_path(setting.artifacts_dir)}",
            "-e",
            f"WORKSPACE_DIR={self._container_path(setting.artifacts_dir)}",
        ]

    def _docker_security_args(self) -> list[str]:
        """Hardening flags shared by one-shot and bash containers."""
        args: list[str] = [
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
        ]
        if self.container_user:
            args += ["--user", self.container_user]
        if self.memory_limit:
            args += ["--memory", self.memory_limit]
        if self.cpu_limit:
            args += ["--cpus", self.cpu_limit]
        if self.pids_limit:
            args += ["--pids-limit", str(self.pids_limit)]
        return args

    def _docker_mount_args(self, host_runtime: Path) -> list[str]:
        app_mount = self.app_stage_dir
        if app_mount is None:
            raise RuntimeError("app mount not staged; call connect() first.")
        return [
            "-v",
            f"{host_runtime}:{self.CONTAINER_WORKSPACE}",
            "-v",
            f"{app_mount}:/app:ro",
        ]

    async def _ensure_app_mount(self) -> None:
        """Stage /app exactly once (idempotent, lock-protected).

        Staging mutates a shared directory (rmtree + copytree). Doing it per
        call races with concurrent runs and with live bash containers that
        have /app bind-mounted, so it must happen once and be serialized.
        """
        if self._app_staged and self.app_stage_dir is not None:
            return
        async with self._app_stage_lock:
            if self._app_staged and self.app_stage_dir is not None:
                return
            # copytree/rmtree are blocking; keep them off the event loop.
            stage_dir = await asyncio.to_thread(self._prepare_app_mount)
            self.app_stage_dir = stage_dir
            self._app_staged = True

    def _prepare_app_mount(self) -> Path:
        """Stage only the files needed to import/install max_ai in Docker."""
        stage_root = self._app_stage_root()
        stage_root.mkdir(parents=True, exist_ok=True)

        pyproject = self.repo_root / "pyproject.toml"
        if not pyproject.is_file():
            raise RuntimeError(f"pyproject.toml not found at {pyproject}.")
        shutil.copy2(pyproject, stage_root / "pyproject.toml")

        readme = self.repo_root / "README.md"
        if readme.is_file():
            shutil.copy2(readme, stage_root / "README.md")
        else:
            (stage_root / "README.md").write_text(
                "MaxAI runtime package.\n",
                encoding="utf-8",
            )

        lockfile = self.repo_root / "uv.lock"
        if lockfile.is_file():
            shutil.copy2(lockfile, stage_root / "uv.lock")

        package_source = self.repo_root / "max_ai"
        if not package_source.is_dir():
            raise RuntimeError(f"max_ai package not found at {package_source}.")

        package_target = stage_root / "max_ai"
        if package_target.exists():
            shutil.rmtree(package_target)
        shutil.copytree(
            package_source,
            package_target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        self._stage_tool_files(stage_root / "tools")
        return stage_root

    def _cleanup_app_stage(self) -> None:
        if self.app_stage_dir is not None and self.app_stage_dir.exists():
            shutil.rmtree(self.app_stage_dir, ignore_errors=True)
        self.app_stage_dir = None
        self._app_staged = False

    def _stage_tool_files(self, target_root: Path) -> None:
        """Copy user-provided tool files into /app/tools for worker imports."""
        if target_root.exists():
            shutil.rmtree(target_root)
        target_root.mkdir(parents=True, exist_ok=True)

        for source in self.tool_files:
            if not source.exists():
                raise RuntimeError(f"tool file source does not exist: {source}")
            if source.is_file():
                shutil.copy2(source, target_root / source.name)
                continue
            if not source.is_dir():
                raise RuntimeError(f"tool file source must be a file or directory: {source}")
            for child in source.iterdir():
                destination = target_root / child.name
                if child.is_dir():
                    shutil.copytree(
                        child,
                        destination,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
                    )
                else:
                    shutil.copy2(child, destination)

    @staticmethod
    def _normalize_tool_files(
        tool_files: str | Path | t.Sequence[str | Path] | None,
    ) -> list[Path]:
        if tool_files is None:
            return []
        if isinstance(tool_files, str | Path):
            tool_files = [tool_files]
        return [Path(path).expanduser().resolve() for path in tool_files]

    def _app_stage_root(self) -> Path:
        root = self.workspace_root or setting.root_dir
        return (root / ".docker-runtime" / "app").expanduser().resolve()

    def _docker_run_once_command(
        self,
        host_runtime: Path,
        container_name: str,
    ) -> list[str]:
        return [
            self.docker_bin,
            "run",
            "--rm",
            "-i",
            "--name",
            container_name,
            "--workdir",
            self.CONTAINER_WORKSPACE,
            *self._docker_security_args(),
            *self._docker_mount_args(host_runtime),
            *self._docker_env_args(),
            self.image,
            "python",
            "-m",
            "max_ai.executor.docker.worker",
            "-",
            "-",
        ]

    def _invocation_payload(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_context: ToolContext,
    ) -> dict[str, object]:
        return {
            "tool_ref": tool.docker_ref().model_dump(mode="json"),
            "record": record.model_dump(mode="json"),
            "context": {
                "run_id": tool_context.run_id,
                "user_id": tool_context.user_id,
                "session_id": tool_context.session_id,
                "retry_count": tool_context.retry_count,
                "deps": self._container_context_deps(tool_context.deps),
            },
        }

    def _container_context_deps(self, deps: dict[str, t.Any]) -> dict[str, t.Any]:
        container_deps = dict(deps)
        container_deps.update(
            {
                "runtime_root": self.CONTAINER_WORKSPACE,
                "tools_dir": self._container_path(setting.tool_dir),
                "skills_dir": self._container_path(setting.skill_dir),
                "artifacts_dir": self._container_path(setting.artifacts_dir),
            }
        )
        return container_deps

    async def _run_bash_direct(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_context: ToolContext,
        host_runtime: Path,
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
        shell_command = self._expand_internal_bash_command(shell_command)

        # Always fall back to default_timeout; tool.timeout_seconds may be None.
        timeout = (
            record.parameters.get("timeout_seconds")
            or tool.timeout_seconds
            or self.default_timeout
        )
        timeout = float(timeout)

        container_name = await self._get_or_create_bash_container(
            tool_context=tool_context,
            host_runtime=host_runtime,
        )

        # Enforce the timeout *inside* the container with coreutils `timeout`
        # so the actual command process dies, not just the docker exec client.
        # The asyncio timeout below is a slightly looser backstop for the case
        # where `docker exec` itself hangs.
        inner = f"timeout -s KILL {timeout:g}s bash -lc {shlex.quote(shell_command)}"
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin,
            "exec",
            "--workdir",
            self.CONTAINER_WORKSPACE,
            container_name,
            "bash",
            "-lc",
            inner,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        task = asyncio.create_task(proc.communicate())
        if cancellation_token is not None:
            cancellation_token.link_future(task)

        try:
            stdout_b, stderr_b = await asyncio.wait_for(task, timeout=timeout + 10.0)
        except asyncio.TimeoutError:
            await self._kill_process(proc)
            return ToolResult.timeout(record.id, timeout_seconds=timeout)
        except asyncio.CancelledError:
            await self._kill_process(proc)
            return ToolResult.cancelled_during_execution(record.id)

        # `timeout` returns 124 (or 137 with -s KILL) when it fires.
        if proc.returncode in (124, 137):
            return ToolResult.timeout(record.id, timeout_seconds=timeout)

        session_key = self._bash_session_key(tool_context)
        if session_key in self._bash_sessions:
            self._bash_sessions[session_key].last_used_at = time.monotonic()

        max_output_chars = int(getattr(tool, "max_output_chars", 20000))
        stdout, stdout_truncated = self._truncate_text(
            stdout_b.decode(errors="replace"),
            max_output_chars,
        )
        stderr, stderr_truncated = self._truncate_text(
            stderr_b.decode(errors="replace"),
            max_output_chars,
        )
        return ToolResult.success_result(
            record.id,
            {
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "cwd": self.CONTAINER_WORKSPACE,
                "command": shell_command,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            },
            metadata={"name": tool.name, "executor": "docker"},
        )

    async def _get_or_create_bash_container(
        self,
        tool_context: ToolContext,
        host_runtime: Path,
    ) -> str:
        key = self._bash_session_key(tool_context)
        # Serialize creation per session so two concurrent first-calls don't
        # both try to `run -d --name X` (the loser fails: name already in use).
        async with self._bash_session_locks[key]:
            session = self._bash_sessions.get(key)
            if session is not None and await self._container_exists(session.name):
                session.last_used_at = time.monotonic()
                return session.name

            if session is not None:
                self._bash_sessions.pop(key, None)

            name = self._bash_container_name(tool_context)
            await self._remove_container(name)
            proc = await asyncio.create_subprocess_exec(
                self.docker_bin,
                "run",
                "-d",
                "--name",
                name,
                "--workdir",
                self.CONTAINER_WORKSPACE,
                *self._docker_security_args(),
                *self._docker_mount_args(host_runtime),
                *self._docker_env_args(),
                self.image,
                "sleep",
                "infinity",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )
            _, stderr_b = await proc.communicate()
            if proc.returncode != 0:
                detail = stderr_b.decode(errors="replace").strip()
                raise RuntimeError(detail or "Failed to start bash container.")

            self._bash_sessions[key] = _BashSession(
                name=name,
                last_used_at=time.monotonic(),
            )
            return name

    async def _prune_expired_bash_sessions(self) -> None:
        if not self._bash_sessions:
            return

        now = time.monotonic()
        expired = [
            key
            for key, session in self._bash_sessions.items()
            if now - session.last_used_at >= self.bash_ttl_seconds
        ]
        for key in expired:
            session = self._bash_sessions.pop(key, None)
            if session is not None:
                await self._remove_container(session.name)

    async def _close_all_bash_sessions(self) -> None:
        sessions = list(self._bash_sessions.values())
        self._bash_sessions.clear()
        for session in sessions:
            await self._remove_container(session.name)

    async def _container_exists(self, name: str) -> bool:
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin,
            "container",
            "inspect",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def _remove_container(self, name: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin,
            "rm",
            "-f",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()

    async def _kill_process(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        proc.kill()
        await proc.communicate()

    @staticmethod
    def _tool_result_from_stdout(stdout: str) -> ToolResult | None:
        # Prefer the explicit marker block the worker emits. Fall back to the
        # last JSON-parseable line for backward compatibility.
        begin = stdout.rfind(_RESULT_BEGIN)
        if begin != -1:
            end = stdout.find(_RESULT_END, begin)
            raw = (
                stdout[begin + len(_RESULT_BEGIN) : end]
                if end != -1
                else stdout[begin + len(_RESULT_BEGIN) :]
            ).strip()
            try:
                return ToolResult.model_validate_json(raw)
            except Exception:
                pass

        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                return ToolResult.model_validate_json(line)
            except Exception:
                continue
        return None

    @staticmethod
    def _clean_user_id(user_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", user_id):
            raise ValueError(
                "user_id must contain only letters, numbers, underscore, dot, or dash."
            )
        if user_id in _RESERVED_USER_IDS:
            raise ValueError("user_id cannot be '.' or '..'.")
        return user_id

    def _bash_session_key(self, tool_context: ToolContext) -> str:
        return tool_context.session_id or tool_context.run_id or tool_context.user_id

    def _bash_container_name(self, tool_context: ToolContext) -> str:
        raw = self._bash_session_key(tool_context)
        clean = re.sub(r"[^a-z0-9_.-]+", "-", raw.lower()).strip(".-")
        return f"maxai-bash-{clean or 'runtime'}"

    def _container_path(self, name: str) -> str:
        return f"{self.CONTAINER_WORKSPACE.rstrip('/')}/{name.strip('/')}"

    def _expand_internal_bash_command(self, command: str) -> str:
        match = _READ_SKILL_RE.match(command)
        if match is None:
            return command
        skill_name = match.group(1)
        skill_file = f"{self._container_path(setting.skill_dir)}/{skill_name}/SKILL.md"
        return f"cat {shlex.quote(skill_file)}"

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> tuple[str, bool]:
        if len(value) <= max_chars:
            return value, False
        keep = max(max_chars, 0)
        return value[:keep] + "\n[output truncated]", True