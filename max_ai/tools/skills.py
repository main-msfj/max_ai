"""Tools for discovering and running materialized skills."""

from __future__ import annotations

import asyncio
import os
import re
import typing as t
from pathlib import Path

from ..base.tools import CoreTool, ToolContext
from ..config import setting
from ..types.tool_call import ToolCallRecord, ToolResult
from ..types.tools import DockerToolRef, ToolApprovalMode


_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class SearchSkillsTool(CoreTool):
    """Search the agent's available skill catalog."""

    def __init__(
        self,
        skills: list[dict[str, str]],
        timeout_seconds: float = 30,
    ) -> None:
        super().__init__(
            name="search_skills",
            description=(
                "Search the available skills by task, name, or description. "
                "Use this before skill_bash when deciding which skill applies."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            timeout_seconds=timeout_seconds,
        )
        self.skills = [dict(skill) for skill in skills]

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Task or keywords to match against skill metadata.",
                },
                "limit": {
                    "type": ["integer", "null"],
                    "description": "Maximum number of matching skills to return.",
                    "minimum": 1,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def docker_ref(self) -> DockerToolRef:
        return DockerToolRef(
            kind="class",
            module=__name__,
            qualname=type(self).__qualname__,
            config={
                "skills": self.skills,
                "timeout_seconds": self.timeout_seconds,
            },
        )

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: t.Any | None = None,
    ) -> ToolResult:
        validation = self.validate_parameters(tool_request)
        if not validation.is_tool_valid:
            return ToolResult.invalid_parameters(
                tool_request.id,
                validation.msg_error or "Invalid search_skills parameters.",
            )

        query = t.cast(str, tool_request.parameters["query"]).strip().lower()
        limit = tool_request.parameters.get("limit") or 5
        matches = self._search(query=query, limit=int(limit))
        return ToolResult.success_result(
            tool_request.id,
            {"skills": matches},
            metadata={"name": self.name},
        )

    def _search(self, query: str, limit: int) -> list[dict[str, str]]:
        if not query:
            return self.skills[:limit]

        terms = [term for term in re.split(r"\s+", query) if term]
        scored: list[tuple[int, dict[str, str]]] = []

        for skill in self.skills:
            haystack = " ".join(
                [
                    skill.get("name", ""),
                    skill.get("description", ""),
                    skill.get("path", ""),
                ]
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            if query in haystack:
                score += 3
            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda item: (-item[0], item[1].get("name", "")))
        return [skill for _, skill in scored[:limit]]


class SkillBashTool(CoreTool):
    """Run shell commands inside the current user's materialized skills dir."""

    _DESCRIPTION = (
        "Execute a shell command in the current user's skills directory. "
        "The directory is available as $SKILLS_DIR. Use this to read "
        "SKILL.md files, inspect references/assets, and run skill scripts. "
        "Write generated files to $WORKSPACE_DIR."
    )

    def __init__(self, timeout_seconds: float = 120) -> None:
        super().__init__(
            name="skill_bash",
            description=self._DESCRIPTION,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            timeout_seconds=timeout_seconds,
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run from the skills directory.",
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
            config={"timeout_seconds": self.timeout_seconds},
        )

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: t.Any | None = None,
    ) -> ToolResult:
        validation = self.validate_parameters(tool_request)
        if not validation.is_tool_valid:
            return ToolResult.invalid_parameters(
                tool_request.id,
                validation.msg_error or "Invalid skill_bash parameters.",
            )

        if tool_context is None:
            return ToolResult.execution_error(
                tool_request.id,
                "skill_bash requires ToolContext with user_id.",
            )

        command = t.cast(str, tool_request.parameters["command"])
        timeout = tool_request.parameters.get("timeout_seconds") or self.timeout_seconds
        if timeout <= 0:
            return ToolResult.invalid_parameters(
                tool_request.id,
                "timeout_seconds must be greater than zero.",
            )

        skills_dir = self._skills_dir_for(tool_context.user_id)
        if not skills_dir.is_dir():
            return ToolResult.execution_error(
                tool_request.id,
                (
                    f"User skills directory does not exist: {skills_dir}. "
                    "Materialize the skill registry for this user before "
                    "calling skill_bash."
                ),
            )

        env = os.environ.copy()
        env["SKILLS_DIR"] = str(skills_dir)
        env.setdefault("WORKSPACE_DIR", str(self._workspace_dir_for(tool_context.user_id)))

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=skills_dir,
                env=env,
            )

            task = asyncio.create_task(proc.communicate())
            if cancellation_token is not None:
                cancellation_token.link_future(task)
            stdout_b, stderr_b = await asyncio.wait_for(task, timeout=float(timeout))

            return ToolResult.success_result(
                tool_request.id,
                {
                    "exit_code": proc.returncode,
                    "stdout": stdout_b.decode(errors="replace"),
                    "stderr": stderr_b.decode(errors="replace"),
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

    @staticmethod
    def _skills_dir_for(user_id: str) -> Path:
        env_dir = os.environ.get("SKILLS_DIR")
        if env_dir:
            return Path(env_dir).expanduser().resolve()
        if not isinstance(user_id, str) or not _VALID_NAME_RE.match(user_id):
            raise ValueError(
                f"Invalid user_id {user_id!r}. Allowed characters: "
                "letters, digits, underscores, hyphens."
            )
        return setting.get_or_create_server_tmp_dir() / user_id / "skills"

    @staticmethod
    def _workspace_dir_for(user_id: str) -> Path:
        env_dir = os.environ.get("WORKSPACE_DIR")
        if env_dir:
            return Path(env_dir).expanduser().resolve()
        if not isinstance(user_id, str) or not _VALID_NAME_RE.match(user_id):
            raise ValueError(
                f"Invalid user_id {user_id!r}. Allowed characters: "
                "letters, digits, underscores, hyphens."
            )
        workspace_dir = setting.get_or_create_server_tmp_dir() / user_id / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir
