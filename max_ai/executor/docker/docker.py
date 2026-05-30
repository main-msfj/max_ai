"""Docker executor for sandboxed skill execution.

This executor runs ONLY runtime tools — i.e. ``bash`` — inside a Docker
container. Ordinary tools (FunctionAsTool / ``@tool`` wrappers,
WorkspaceTool, knowledge tools, …) are developer-authored and run
in-process; the ``RoutingExecutor`` keeps them local and sends only
``CoreRuntimeTool`` instances here. The variable, model-driven surface
that actually needs isolation is the skill scripts the LLM chooses to
run through ``bash``.

It uses the Docker CLI directly (no Compose) so the runtime boundary
stays easy to reason about:

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
Because the bind mount is shared with the host, files the container
writes under /mnt/artifacts are immediately visible to the local
WorkspaceTool — no copy-back round trip is needed.

Bash runs in a short-lived persistent container so a skill can issue
several commands against the same runtime state; idle bash containers
are removed after ``bash_ttl_seconds``.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import typing as t
from dataclasses import dataclass
from pathlib import Path

from ...config import setting
from ...base.executor import CoreExecutor
from ...base.tools import CoreTool, ToolContext
from ...termination import CancellationToken
from ...types.tool_call import ToolCallRecord, ToolResult
from ...core.primitives import FailureReason


# Strict form: exactly `read_skill <name>` with no extra arguments.
_READ_SKILL_RE = re.compile(r"^\s*read_skill\s+([A-Za-z0-9_-]+)\s*$")
# Loose detector: the command intends to be a read_skill invocation even
# if it's malformed (extra args, quotes, chaining). Used to return a
# helpful error instead of a raw "command not found".
_READ_SKILL_PREFIX_RE = re.compile(r"^\s*read_skill\b")


@dataclass
class _BashSession:
    """A short-lived interactive-ish bash container."""

    name: str
    last_used_at: float


class DockerExecutor(CoreExecutor):
    """Run runtime tools (bash) inside Docker with an explicit contract.

    Bash runs in a short-lived container so skills can issue several
    commands against the same runtime state; idle bash containers are
    removed after ``bash_ttl_seconds``. No other tool type is accepted —
    ordinary tools run locally via the ``RoutingExecutor``.
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
        self.workspace_root: Path | None = None
        self._bash_sessions: dict[str, _BashSession] = {}
        self._bash_cleanup_tasks: dict[str, asyncio.Task[None]] = {}

    async def bind_to_workspace(self, workspace_registry_root: str | Path) -> None:
        """Bind the host workspace root used to resolve user runtimes."""
        self.workspace_root = Path(workspace_registry_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    async def connect(self) -> None:
        """Verify Docker is available and ensure the sandbox image exists."""
        await self._ensure_docker_available()
        if self.force_build_image or not await self._image_exists():
            await self._build_image()

    async def disconnect(self) -> None:
        """Remove any bash containers still kept alive by this executor."""
        await self._close_all_bash_sessions()

    async def run(
        self,
        tool: CoreTool,
        record: ToolCallRecord,
        tool_context: ToolContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        """Execute one runtime tool call inside Docker.

        Only ``bash`` (``CoreRuntimeTool``) is supported. Anything else
        reaching this executor is a routing error and is returned as a
        failed result rather than executed.
        """
        try:
            await self._ensure_connected()
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

            # Ordinary tools run locally via the RoutingExecutor; nothing
            # else should be dispatched here. Surface as a clear failure.
            return ToolResult.execution_error(
                record.id,
                (
                    f"DockerExecutor only runs runtime tools (bash); received "
                    f"{tool.name!r}. Ordinary tools run locally."
                ),
            )

        except asyncio.CancelledError:
            return ToolResult.cancelled_during_execution(record.id)

        except Exception as exc:
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
        root = self.workspace_root or setting.root_dir
        clean_user_id = self._clean_user_id(user_id)
        return (root / clean_user_id).expanduser().resolve()

    def _ensure_runtime_layout(self, host_runtime: Path) -> None:
        for name in (setting.tool_dir, setting.skill_dir, setting.artifacts_dir):
            (host_runtime / name).mkdir(parents=True, exist_ok=True)

    def _docker_env_args(self) -> list[str]:
        runtime_root = self.CONTAINER_WORKSPACE
        return [
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

    def _docker_mount_args(self, host_runtime: Path) -> list[str]:
        """Mount the user's host runtime as /mnt.

        When Max AI itself runs inside a devcontainer, paths such as
        ``/max_ai/user123`` are container paths. The Docker daemon sees
        the host path behind that bind mount instead, so translate before
        handing the source to ``docker run``.
        """
        source = self._docker_visible_path(host_runtime)
        return [
            "-v",
            f"{source}:{self.CONTAINER_WORKSPACE}",
        ]

    def _docker_visible_path(self, path: Path) -> Path:
        """Return the path Docker daemon can see for a local path."""
        resolved = path.expanduser().resolve()
        best_mount: tuple[Path, Path] | None = None

        try:
            lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
        except OSError:
            return resolved

        for line in lines:
            if " - " not in line:
                continue
            before, _sep, after = line.partition(" - ")
            fields = before.split()
            if len(fields) < 5:
                continue
            mount_root = Path(fields[3].replace("\\040", " "))
            mount_point = Path(fields[4].replace("\\040", " "))
            try:
                rel = resolved.relative_to(mount_point)
            except ValueError:
                continue
            if best_mount is None or len(mount_point.parts) > len(best_mount[0].parts):
                best_mount = (mount_point, mount_root / rel)

        if best_mount is None:
            return resolved
        return best_mount[1]

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

        shell_command, expand_error = self._expand_internal_bash_command(
            shell_command, host_runtime
        )
        if expand_error is not None:
            # Malformed read_skill invocation or unknown skill name —
            # return the catalog/usage so the model self-corrects instead
            # of getting a raw shell failure or inventing a path.
            return ToolResult.invalid_parameters(record.id, expand_error)

        timeout = record.parameters.get("timeout_seconds") or tool.timeout_seconds
        container_name = await self._get_or_create_bash_container(
            tool_context=tool_context,
            host_runtime=host_runtime,
        )
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin,
            "exec",
            "--workdir",
            self.CONTAINER_WORKSPACE,
            container_name,
            "bash",
            "-lc",
            shell_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        task = asyncio.create_task(proc.communicate())
        if cancellation_token is not None:
            cancellation_token.link_future(task)

        try:
            stdout_b, stderr_b = await asyncio.wait_for(task, timeout=float(timeout))
        except asyncio.TimeoutError:
            await self._kill_process(proc)
            return ToolResult.timeout(record.id, timeout_seconds=float(timeout))
        except asyncio.CancelledError:
            await self._kill_process(proc)
            return ToolResult.cancelled_during_execution(record.id)

        session_key = self._bash_session_key(tool_context)
        if session_key in self._bash_sessions:
            self._bash_sessions[session_key].last_used_at = time.monotonic()
            self._schedule_bash_session_cleanup(session_key)

        max_output_chars = int(getattr(tool, "max_output_chars", 20000))
        stdout, stdout_truncated, stdout_chars = self._truncate_text(
            stdout_b.decode(errors="replace"),
            max_output_chars,
        )
        stderr, stderr_truncated, stderr_chars = self._truncate_text(
            stderr_b.decode(errors="replace"),
            max_output_chars,
        )
        payload = {
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "cwd": self.CONTAINER_WORKSPACE,
            "command": shell_command,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "stdout_original_chars": stdout_chars,
            "stderr_original_chars": stderr_chars,
        }
        metadata = {"name": tool.name, "executor": "docker"}
        if proc.returncode != 0:
            detail = stderr.strip() or stdout.strip() or "Bash command failed."
            return ToolResult(
                success=False,
                error=f"Bash command failed with exit code {proc.returncode}: {detail}",
                result=payload,
                failure_reason=FailureReason.EXECUTION_ERROR,
                tool_call_id=record.id,
                metadata=metadata,
            )

        return ToolResult.success_result(
            record.id,
            payload,
            metadata=metadata,
        )

    async def _get_or_create_bash_container(
        self,
        tool_context: ToolContext,
        host_runtime: Path,
    ) -> str:
        key = self._bash_session_key(tool_context)
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
            "--network",
            "none",
            "--workdir",
            self.CONTAINER_WORKSPACE,
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
        self._schedule_bash_session_cleanup(key)
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
            task = self._bash_cleanup_tasks.pop(key, None)
            if task is not None:
                task.cancel()
            if session is not None:
                await self._remove_container(session.name)

    async def _close_all_bash_sessions(self) -> None:
        for task in self._bash_cleanup_tasks.values():
            task.cancel()
        self._bash_cleanup_tasks.clear()
        sessions = list(self._bash_sessions.values())
        self._bash_sessions.clear()
        for session in sessions:
            await self._remove_container(session.name)

    def _schedule_bash_session_cleanup(self, key: str) -> None:
        previous = self._bash_cleanup_tasks.pop(key, None)
        if previous is not None:
            previous.cancel()
        self._bash_cleanup_tasks[key] = asyncio.create_task(
            self._cleanup_bash_session_after_ttl(key)
        )

    async def _cleanup_bash_session_after_ttl(self, key: str) -> None:
        try:
            await asyncio.sleep(self.bash_ttl_seconds)
            session = self._bash_sessions.get(key)
            if session is None:
                return
            if time.monotonic() - session.last_used_at < self.bash_ttl_seconds:
                self._schedule_bash_session_cleanup(key)
                return
            self._bash_sessions.pop(key, None)
            self._bash_cleanup_tasks.pop(key, None)
            await self._remove_container(session.name)
        except asyncio.CancelledError:
            return

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
    def _clean_user_id(user_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", user_id):
            raise ValueError(
                "user_id must contain only letters, numbers, underscore, dot, or dash."
            )
        return user_id

    def _bash_session_key(self, tool_context: ToolContext) -> str:
        return tool_context.session_id or tool_context.run_id or tool_context.user_id

    def _bash_container_name(self, tool_context: ToolContext) -> str:
        raw = self._bash_session_key(tool_context)
        clean = re.sub(r"[^a-z0-9_.-]+", "-", raw.lower()).strip(".-")
        return f"maxai-bash-{clean or 'runtime'}"

    def _container_path(self, name: str) -> str:
        return f"{self.CONTAINER_WORKSPACE.rstrip('/')}/{name.strip('/')}"

    def _expand_internal_bash_command(
        self, command: str, host_runtime: Path
    ) -> tuple[str, str | None]:
        """Expand ``read_skill <name>`` into a cat of that skill's SKILL.md.

        Returns ``(command_to_run, error_message)``. Mirrors
        ``BashTool``: a malformed invocation or an unknown skill name
        returns an actionable error (listing the available skills)
        instead of letting the shell fail or the model invent a path.

        The skill is checked on the *host* runtime (which is bind-mounted
        as ``/mnt``), while the resulting ``cat`` targets the *container*
        path.
        """
        if not _READ_SKILL_PREFIX_RE.match(command):
            return command, None

        match = _READ_SKILL_RE.match(command)
        if match is None:
            return command, (
                "read_skill takes exactly one skill name and no extra arguments. "
                "Usage: read_skill <skill-name>. To run scripts or chain commands, "
                "issue them as a separate bash call after loading the skill."
            )

        skill_name = match.group(1)
        host_skills = host_runtime / setting.skill_dir
        if not (host_skills / skill_name / "SKILL.md").is_file():
            available = (
                sorted(p.name for p in host_skills.iterdir() if p.is_dir())
                if host_skills.is_dir()
                else []
            )
            return command, (
                f"Skill {skill_name!r} not found. "
                f"Available skills: {available or 'none'}. "
                "Use one of the available skill names exactly as listed; "
                "do not invent skill names or paths."
            )

        skill_file = (
            f"{self._container_path(setting.skill_dir)}/{skill_name}/SKILL.md"
        )
        preface = (
            "Skill instructions loaded only. No user file has been created yet. "
            "Next, run the script shown below with bash and then verify the output file.\n\n"
        )
        return f"printf %s {preface!r}; cat {skill_file!r}", None

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> tuple[str, bool, int]:
        """Truncate to ``max_chars``, reporting the original length.

        Returns ``(text, was_truncated, original_chars)`` so the model
        knows how much output it did not see and does not reason over a
        partial result as if it were complete.
        """
        original = len(value)
        if original <= max_chars:
            return value, False, original
        keep = max(max_chars, 0)
        return value[:keep] + "\n[output truncated]", True, original