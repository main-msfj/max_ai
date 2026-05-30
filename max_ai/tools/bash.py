"""Shell tool for running commands inside the sandboxed runtime."""

from __future__ import annotations

import re
import os
import asyncio
import logging
import typing as t
from pathlib import Path

from ..config import setting
from ..loggers.scope import ScopedLogger
from ..termination import CancellationToken
from ..base.tools import CoreRuntimeTool, ToolContext
from ..types.tool_call import ToolCallRecord, ToolResult
from ..types.tools import (
    CoreToolParameters,
    DockerToolRef,
    ToolApprovalMode,
    RuntimeDirs,
)

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["DockerSanbox"])

_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
# Strict form: exactly `read_skill <name>` with no extra arguments.
_READ_SKILL_RE = re.compile(r"^\s*read_skill\s+([A-Za-z0-9_-]+)\s*$")
# Loose detector: the command *intends* to be a read_skill invocation,
# even if it's malformed (extra args, quotes, chaining). Used to give a
# helpful error instead of letting the shell fail with "command not found".
_READ_SKILL_PREFIX_RE = re.compile(r"^\s*read_skill\b")


# =====================================================================
# DANGEROUS COMMAND PATTERNS
# =====================================================================
# Commands blocked at the BashTool layer. The Docker container is the
# primary security boundary (read-only filesystem outside /mnt, non-root
# user, dropped capabilities). This list is a second line of defense for
# things that either:
#
#   (a) Would damage the container or destabilize the runtime, OR
#   (b) Are clearly malicious patterns we want to reject with a clear
#       message instead of letting the kernel return a cryptic error.
#
# Inside /mnt the model has full freedom — read, write, install, delete,
# run scripts. This list does NOT restrict what the model can do with
# its own workspace.
#
# NOTE: these patterns are bypassable by a determined adversary (string
# splitting, command substitution, env indirection). They exist to give
# the model a clear, actionable error — NOT to contain a hostile actor.
# Containment is the sandbox's job; never run skills without it.
# =====================================================================

_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # -------- Privilege escalation --------
    (
        re.compile(r"(?:^|[\s;&|()`])(?:sudo|su|doas)\b"),
        "privilege escalation (sudo / su / doas)",
    ),
    # -------- Host / container lifecycle --------
    (
        re.compile(r"(?:^|[\s;&|()`])(?:shutdown|reboot|halt|poweroff)\b"),
        "container lifecycle command (shutdown / reboot / halt / poweroff)",
    ),
    (
        re.compile(r"(?:^|[\s;&|()`])kill\s+(?:-\w+\s+)?1(?:\s|$)"),
        "killing PID 1 (terminates the container)",
    ),
    (
        re.compile(r":\(\)\s*\{\s*:\s*\|\s*:&"),
        "forkbomb pattern",
    ),
    # -------- Block devices / filesystem manipulation --------
    (
        re.compile(r"(?:^|[\s;&|()`])(?:mkfs|fdisk|parted|mkswap)\b"),
        "filesystem creation / partitioning",
    ),
    (
        re.compile(r"(?:^|[\s;&|()`])(?:mount|umount)\b"),
        "mount / umount",
    ),
    (
        re.compile(r"(?:^|[\s;&|()`])dd\s+[^;|&]*\bof=/dev/"),
        "dd writing to a block device",
    ),
    # -------- Kernel / module manipulation --------
    (
        re.compile(r"(?:^|[\s;&|()`])(?:modprobe|insmod|rmmod|depmod)\b"),
        "kernel module manipulation",
    ),
    (
        re.compile(r"(?:^|[\s;&|()`])sysctl\s+-w\b"),
        "sysctl write",
    ),
    # -------- Network reconfiguration --------
    (
        re.compile(r"(?:^|[\s;&|()`])(?:iptables|ip6tables|nft|nftables)\b"),
        "firewall reconfiguration (iptables / nft)",
    ),
    (
        re.compile(
            r"(?:^|[\s;&|()`])ip\s+(?:route|rule|link)\s+(?:add|del|change|replace)\b"
        ),
        "network route / rule / link modification",
    ),
    # -------- Writes outside /mnt to system paths --------
    (
        re.compile(
            r"(?:^|[\s;&|()`])"
            r"(?:rm|mv|cp|chmod|chown|truncate|tee)\s+"
            r"(?:-[A-Za-z]+\s+)*"
            r"(?:/(?:etc|usr|bin|sbin|lib|lib64|boot|root|var|opt|proc|sys|dev)(?:/|\s|$))"
        ),
        "destructive operation on a system path outside /mnt",
    ),
    (
        re.compile(
            r"(?:>|>>)\s*"
            r"/(?:etc|usr|bin|sbin|lib|lib64|boot|root|var|opt|proc|sys|dev)/"
        ),
        "redirect writing into a system path outside /mnt",
    ),
    # -------- Docker / container escape attempts --------
    (
        re.compile(r"(?:^|[\s;&|()`])docker\b"),
        "docker command from inside the container",
    ),
    (
        re.compile(r"/var/run/docker\.sock"),
        "access to the Docker socket",
    ),
    (
        re.compile(r"(?:^|[\s;&|()`])(?:nsenter|unshare|chroot)\b"),
        "namespace / chroot manipulation",
    ),
    # -------- Process injection --------
    (
        re.compile(r"(?:^|[\s;&|()`])(?:strace|ltrace|gdb)\s+-p\b"),
        "attaching a debugger to a running process",
    ),
]


def _find_dangerous(command: str) -> str | None:
    """Return the human-readable reason if the command is blocked, else None."""
    for pattern, reason in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            log.warning("malicious Command Found", pattern=pattern, reason=reason)
            return reason
    return None


class BashTool(CoreRuntimeTool):
    """Execute shell commands inside the sandboxed runtime.

    The tool runs inside a Docker container whose only writable area is
    /mnt (bind-mounted from the host). Everything else is read-only and
    runs as a non-root user with dropped capabilities, so the model has
    full freedom inside /mnt and the kernel enforces the boundary.

    Inside /mnt the layout is:
      /mnt/skills/    — skill packages (SKILL.md + scripts/)
      /mnt/artifacts/ — user-facing outputs you create
      /mnt/tools/     — helper code

    The BashTool layer adds a small list of pattern-based blocks for
    commands that would damage the container or escape the sandbox, so
    the model gets a clear error message instead of a cryptic kernel
    failure.
    """

    _DESCRIPTION = (
        "Run a shell command in the sandboxed runtime for Skill Tasks. Skills live in "
        "/mnt/skills and you save user-facing outputs to /mnt/artifacts. "
        "Inside /mnt you have full freedom: read, write, install packages "
        "(pip, uv), run scripts in any interpreter, chain commands with "
        "&& / || / ;, use pipes, redirects, loops, subshells — do whatever "
        "the task needs. The only rule is stay inside /mnt; the container "
        "blocks writes elsewhere."
    )

    def __init__(
        self,
        name: str = "bash",
        description: str | None = None,
        timeout_seconds: float = 120,
        max_output_chars: int = 20000,
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
    ) -> None:
        super().__init__(
            name=name,
            description=description or self._DESCRIPTION,
            approval_mode=approval_mode,
            timeout_seconds=timeout_seconds,
        )
        self.max_output_chars = max_output_chars

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command to run Skill Tasks from /mnt. Standard shell "
                        "syntax supported: pipes, redirects, chaining "
                        "(&& / || / ;), loops, subshells, and so on."
                    ),
                },
                "timeout_seconds": {
                    "type": ["integer", "null"],
                    "description": "Optional timeout for this command in seconds.",
                    "minimum": 1,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        validation = super().validate_parameters(tool_request)
        if not validation.is_tool_valid:
            return validation

        command = t.cast(str, tool_request.parameters["command"])
        if not command.strip():
            return CoreToolParameters(
                is_tool_valid=False,
                msg_error="command cannot be empty.",
            )

        reason = _find_dangerous(command)
        if reason is not None:
            # Surface the actual reason so the model can self-correct on the
            # next turn instead of blindly retrying or inventing a workaround.
            return CoreToolParameters(
                is_tool_valid=False,
                msg_error=(
                    f"Command blocked: {reason}. This action is not allowed. "
                    "Stay inside /mnt and avoid system-level operations; adjust "
                    "the command and try again."
                ),
            )
        return validation

    def docker_ref(self) -> DockerToolRef:
        return DockerToolRef(
            kind="class",
            module=__name__,
            qualname=type(self).__qualname__,
            config={
                "timeout_seconds": self.timeout_seconds,
                "max_output_chars": self.max_output_chars,
                "approval_mode": self.approval_mode,
            },
        )

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        validation = self.validate_parameters(tool_request)
        if not validation.is_tool_valid:
            msg_error = validation.msg_error or "Invalid bash parameters."
            log.info(msg_error)
            return ToolResult.invalid_parameters(tool_request.id, msg_error)

        if tool_context is None:
            msg_error = "bash requires ToolContext with user_id."
            log.error(msg_error)
            return ToolResult.execution_error(tool_request.id, msg_error)

        command = t.cast(str, tool_request.parameters["command"])
        timeout = tool_request.parameters.get("timeout_seconds") or self.timeout_seconds
        if timeout <= 0:
            msg_error = "timeout_seconds must be greater than zero."
            log.info(msg_error)
            return ToolResult.invalid_parameters(tool_request.id, msg_error)

        try:
            runtime = self._runtime_dirs(tool_context)
            runtime.root.mkdir(parents=True, exist_ok=True)
            runtime.tools.mkdir(parents=True, exist_ok=True)
            runtime.skills.mkdir(parents=True, exist_ok=True)
            runtime.artifacts.mkdir(parents=True, exist_ok=True)

            command, expand_error = self._expand_internal_command(command, runtime)
            if expand_error is not None:
                # Malformed read_skill invocation or unknown skill name.
                # Returning the catalog/usage keeps the model anchored to
                # real skills instead of guessing paths.
                log.info(expand_error)
                return ToolResult.invalid_parameters(tool_request.id, expand_error)

            env = os.environ.copy()

            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=runtime.root,
                env=env,
            )

            task = asyncio.create_task(proc.communicate())
            if cancellation_token is not None:
                cancellation_token.link_future(task)
            stdout_b, stderr_b = await asyncio.wait_for(task, timeout=float(timeout))

            stdout = stdout_b.decode(errors="replace")
            stderr = stderr_b.decode(errors="replace")
            stdout, stdout_truncated, stdout_chars = self._truncate(stdout)
            stderr, stderr_truncated, stderr_chars = self._truncate(stderr)

            return ToolResult.success_result(
                tool_request.id,
                {
                    "exit_code": proc.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "cwd": str(runtime.root),
                    "command": command,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "stdout_original_chars": stdout_chars,
                    "stderr_original_chars": stderr_chars,
                },
                metadata={"name": self.name},
            )

        except asyncio.TimeoutError:
            if "proc" in locals() and proc.returncode is None:
                proc.kill()
                await proc.communicate()
            return ToolResult.timeout(tool_request.id, timeout_seconds=float(timeout))

        except asyncio.CancelledError:
            if "proc" in locals() and proc.returncode is None:
                proc.kill()
                await proc.communicate()
            return ToolResult.cancelled_during_execution(tool_request.id)

        except Exception as e:
            return ToolResult.execution_error(tool_request.id, str(e))

    def _truncate(self, value: str) -> tuple[str, bool, int]:
        """Truncate to max_output_chars, reporting the original length.

        Returns ``(text, was_truncated, original_chars)``. Surfacing the
        original size lets the model know how much it did NOT see, so it
        doesn't reason over a partial output as if it were complete.
        """
        original = len(value)
        if original <= self.max_output_chars:
            return value, False, original
        keep = max(self.max_output_chars, 0)
        return value[:keep] + "\n[output truncated]", True, original

    @staticmethod
    def _expand_internal_command(
        command: str, runtime: RuntimeDirs
    ) -> tuple[str, str | None]:
        """Expand ``read_skill <name>`` into a cat of that skill's SKILL.md.

        Returns ``(command_to_run, error_message)``:

        - Not a read_skill invocation → ``(command, None)`` unchanged.
        - Looks like read_skill but malformed (extra args, quotes,
          chaining) → ``(command, usage_error)``.
        - Valid form but the skill isn't materialized → ``(command,
          not_found_error)`` listing the available skills.
        - Valid and present → ``(cat_command, None)``.

        The error branches exist so the model gets an actionable message
        and stays anchored to the real skill catalog, instead of a raw
        "command not found" or an invented path.
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
        skill_file = runtime.skills / skill_name / "SKILL.md"
        if not skill_file.is_file():
            available = (
                sorted(p.name for p in runtime.skills.iterdir() if p.is_dir())
                if runtime.skills.is_dir()
                else []
            )
            return command, (
                f"Skill {skill_name!r} not found. "
                f"Available skills: {available or 'none'}. "
                "Use one of the available skill names exactly as listed; "
                "do not invent skill names or paths."
            )

        return f"cat {skill_file.as_posix()!r}", None

    @classmethod
    def _runtime_dirs(cls, tool_context: ToolContext) -> RuntimeDirs:
        deps = tool_context.deps or {}
        root = cls._path_from_deps_or_env(deps, "runtime_root", "RUNTIME_DIR")
        tools = cls._path_from_deps_or_env(deps, "tools_dir", "TOOLS_DIR")
        skills = cls._path_from_deps_or_env(deps, "skills_dir", "SKILLS_DIR")
        artifacts = cls._path_from_deps_or_env(deps, "artifacts_dir", "ARTIFACTS_DIR")

        if root is None:
            # Fallback layout matches the workspace registry (no `tmp`
            # segment): <root_dir>/<user_id>.
            user_id = cls._safe_user_id(tool_context.user_id)
            root = setting.root_dir / user_id
        if tools is None:
            tools = root / "tools"
        if skills is None:
            skills = root / "skills"
        if artifacts is None:
            artifacts = root / "artifacts"

        root = root.expanduser().resolve()
        tools = tools.expanduser().resolve()
        skills = skills.expanduser().resolve()
        artifacts = artifacts.expanduser().resolve()
        for child in (tools, skills, artifacts):
            child.relative_to(root)
        return RuntimeDirs(root=root, tools=tools, skills=skills, artifacts=artifacts)

    @staticmethod
    def _path_from_deps_or_env(
        deps: dict[str, t.Any],
        dep_key: str,
        env_key: str,
    ) -> Path | None:
        value = deps.get(dep_key) or os.environ.get(env_key)
        if value is None:
            return None
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{dep_key} must be a string or Path.")
        return Path(value)

    @staticmethod
    def _safe_user_id(user_id: str) -> str:
        if not isinstance(user_id, str) or not _VALID_NAME_RE.match(user_id):
            raise ValueError("Invalid user_id  Allowed characters")
        return user_id