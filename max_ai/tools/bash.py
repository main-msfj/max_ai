"""Shell tool for navigating the user's runtime directory."""

from __future__ import annotations

import asyncio
import os
import re
import typing as t
from pathlib import Path

from ..base.tools import CoreRuntimeTool, ToolContext
from ..config import setting
from ..termination import CancellationToken
from ..types.tool_call import ToolCallRecord, ToolResult
from ..types.tools import DockerToolRef, ToolApprovalMode, RuntimeDirs


_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class BashTool(CoreRuntimeTool):
    """Execute shell commands from the user's runtime root."""

    _DESCRIPTION = (
        "Run a shell command from the user's runtime root. The command starts "
        "in a directory containing tools/, skills/, and artifacts/. Use "
        "relative paths: inspect skills/ for skill packages, use tools/ for "
        "helper code, and write user-facing outputs to artifacts/."
    )

    def __init__(
        self,
        timeout_seconds: float = 120,
        max_output_chars: int = 20000,
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.ASK_APPROVED,
    ) -> None:
        super().__init__(
            name="bash",
            description=self._DESCRIPTION,
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
                        "Shell command to run from the runtime root. Use relative "
                        "paths such as skills/, tools/, and artifacts/."
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
            return ToolResult.invalid_parameters(
                tool_request.id,
                validation.msg_error or "Invalid bash parameters.",
            )
        if tool_context is None:
            return ToolResult.execution_error(
                tool_request.id,
                "bash requires ToolContext with user_id.",
            )

        command = t.cast(str, tool_request.parameters["command"])
        if not command.strip():
            return ToolResult.invalid_parameters(
                tool_request.id,
                "command cannot be empty.",
            )

        timeout = tool_request.parameters.get("timeout_seconds") or self.timeout_seconds
        if timeout <= 0:
            return ToolResult.invalid_parameters(
                tool_request.id,
                "timeout_seconds must be greater than zero.",
            )

        try:
            runtime = self._runtime_dirs(tool_context)
            runtime.root.mkdir(parents=True, exist_ok=True)
            runtime.tools.mkdir(parents=True, exist_ok=True)
            runtime.skills.mkdir(parents=True, exist_ok=True)
            runtime.artifacts.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env.update(
                {
                    "RUNTIME_DIR": str(runtime.root),
                    "TOOLS_DIR": str(runtime.tools),
                    "SKILLS_DIR": str(runtime.skills),
                    "ARTIFACTS_DIR": str(runtime.artifacts),
                    "WORKSPACE_DIR": str(runtime.artifacts),
                }
            )

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
            stdout, stdout_truncated = self._truncate(stdout)
            stderr, stderr_truncated = self._truncate(stderr)

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

    def _truncate(self, value: str) -> tuple[str, bool]:
        if len(value) <= self.max_output_chars:
            return value, False
        keep = max(self.max_output_chars, 0)
        return value[:keep] + "\n[output truncated]", True

    @classmethod
    def _runtime_dirs(cls, tool_context: ToolContext) -> RuntimeDirs:
        deps = tool_context.deps or {}
        root = cls._path_from_deps_or_env(deps, "runtime_root", "RUNTIME_DIR")
        tools = cls._path_from_deps_or_env(deps, "tools_dir", "TOOLS_DIR")
        skills = cls._path_from_deps_or_env(deps, "skills_dir", "SKILLS_DIR")
        artifacts = cls._path_from_deps_or_env(
            deps, "artifacts_dir", "ARTIFACTS_DIR", fallback_env="WORKSPACE_DIR"
        )

        if root is None:
            user_id = cls._safe_user_id(tool_context.user_id)
            root = setting.root_dir / "tmp" / user_id
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
        fallback_env: str | None = None,
    ) -> Path | None:
        value = deps.get(dep_key) or os.environ.get(env_key)
        if value is None and fallback_env is not None:
            value = os.environ.get(fallback_env)
        if value is None:
            return None
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{dep_key} must be a string or Path.")
        return Path(value)

    @staticmethod
    def _safe_user_id(user_id: str) -> str:
        if not isinstance(user_id, str) or not _VALID_NAME_RE.match(user_id):
            raise ValueError(
                f"Invalid user_id {user_id!r}. Allowed characters: "
                "letters, digits, underscores, hyphens."
            )
        return user_id

