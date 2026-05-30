"""Tool for inspecting generated artifacts (read-only)."""

from __future__ import annotations

import os
import re
import typing as t
from pathlib import Path

from ..base.tools import CoreTool, ToolContext
from ..config import setting
from ..types.tool_call import ToolCallRecord, ToolResult
from ..types.tools import ToolApprovalMode


_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class WorkspaceTool(CoreTool):
    """List and read files under the user's ``<user_id>/artifacts`` directory.

    Read-only on purpose: skills write their outputs to artifacts/ through
    bash, so this tool exists only so the model can inspect what it has
    already produced. It cannot write or delete — use bash for that.
    """

    def __init__(self, timeout_seconds: float = 60) -> None:
        super().__init__(
            name="workspace",
            description=(
                "List and read files under the user's artifacts/ directory only. "
                "Use this to inspect documents and outputs you already created "
                "FOR the user (reports, generated files, deliverables). "
                "It is read-only: to create or edit files, use the bash tool. "
                "Do NOT use this to read skill instructions or skill scripts — "
                "those live under $SKILLS_DIR and are only accessible via bash. "
                "Paths are relative to artifacts/."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            timeout_seconds=timeout_seconds,
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "read"],
                    "description": "Artifacts operation to perform.",
                },
                "path": {
                    "type": ["string", "null"],
                    "description": "Relative file or directory path inside artifacts.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

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
                validation.msg_error or "Invalid workspace parameters.",
            )
        if tool_context is None:
            return ToolResult.execution_error(
                tool_request.id,
                "workspace requires ToolContext with user_id.",
            )

        try:
            artifacts = self._artifacts_dir_for(tool_context)
            artifacts.mkdir(parents=True, exist_ok=True)
            action = tool_request.parameters["action"]

            if action == "list":
                rel = tool_request.parameters.get("path") or "."
                target = self._resolve_inside(artifacts, rel)
                return ToolResult.success_result(
                    tool_request.id,
                    self._list(target, artifacts),
                    metadata={"name": self.name},
                )

            path = tool_request.parameters.get("path")
            if not isinstance(path, str) or not path.strip():
                return ToolResult.invalid_parameters(
                    tool_request.id,
                    "path is required for the read action.",
                )

            skill_redirect = self._maybe_redirect_skill_path(tool_request.id, path)
            if skill_redirect is not None:
                return skill_redirect

            target = self._resolve_inside(artifacts, path)

            if action == "read":
                if not target.is_file():
                    return ToolResult.execution_error(
                        tool_request.id,
                        f"Artifacts file does not exist: {path}",
                    )
                rel_path = self._relative(target, artifacts)
                try:
                    content = target.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    return ToolResult.success_result(
                        tool_request.id,
                        {
                            "path": rel_path,
                            "exists": True,
                            "type": "file",
                            "bytes": target.stat().st_size,
                            "binary": True,
                        },
                        metadata={"name": self.name},
                    )

                return ToolResult.success_result(
                    tool_request.id,
                    {
                        "path": rel_path,
                        "exists": True,
                        "type": "file",
                        "bytes": target.stat().st_size,
                        "content": content,
                    },
                    metadata={"name": self.name},
                )

            return ToolResult.invalid_parameters(
                tool_request.id,
                f"Unsupported workspace action: {action}",
            )

        except Exception as e:
            return ToolResult.execution_error(tool_request.id, str(e))

    @classmethod
    def _artifacts_dir_for(cls, tool_context: ToolContext) -> Path:
        deps = tool_context.deps or {}
        value = (
            deps.get("artifacts_dir")
            or os.environ.get("ARTIFACTS_DIR")
            or os.environ.get("WORKSPACE_DIR")
        )
        if value is not None:
            if not isinstance(value, (str, Path)):
                raise TypeError("artifacts_dir must be a string or Path.")
            return Path(value).expanduser().resolve()
        # Fallback layout matches the workspace registry (no `tmp` segment):
        # <root_dir>/<user_id>/artifacts.
        user_id = cls._safe_user_id(tool_context.user_id)
        return (setting.root_dir / user_id / "artifacts").resolve()

    @staticmethod
    def _safe_user_id(user_id: str) -> str:
        if not isinstance(user_id, str) or not _VALID_NAME_RE.match(user_id):
            raise ValueError(
                f"Invalid user_id {user_id!r}. Allowed characters: "
                "letters, digits, underscores, hyphens."
            )
        return user_id

    @staticmethod
    def _resolve_inside(root: Path, relative_path: str | Path) -> Path:
        relative_path = WorkspaceTool._normalize_artifacts_path(relative_path)
        target = (root / relative_path).expanduser().resolve()
        root_resolved = root.resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError:
            raise ValueError("path must stay inside the artifacts directory.") from None
        return target

    @staticmethod
    def _normalize_artifacts_path(relative_path: str | Path) -> str | Path:
        if isinstance(relative_path, Path):
            return relative_path
        clean = relative_path.strip()
        if clean == "artifacts":
            return "."
        if clean.startswith("artifacts/"):
            return clean.removeprefix("artifacts/") or "."
        return relative_path

    @staticmethod
    def _relative(path: Path, root: Path) -> str:
        return path.resolve().relative_to(root.resolve()).as_posix()

    def _list(self, target: Path, artifacts: Path) -> dict[str, t.Any]:
        if not target.exists():
            return {
                "path": self._relative(target, artifacts),
                "exists": False,
                "items": [],
            }
        if target.is_file():
            return {
                "path": self._relative(target, artifacts),
                "exists": True,
                "items": [
                    {
                        "path": self._relative(target, artifacts),
                        "type": "file",
                        "bytes": target.stat().st_size,
                    }
                ],
            }
        items = []
        for child in sorted(target.iterdir(), key=lambda p: p.name):
            items.append(
                {
                    "path": self._relative(child, artifacts),
                    "type": "directory" if child.is_dir() else "file",
                    "bytes": None if child.is_dir() else child.stat().st_size,
                }
            )
        return {
            "path": self._relative(target, artifacts) if target != artifacts else ".",
            "exists": True,
            "items": items,
        }

    @staticmethod
    def _maybe_redirect_skill_path(
        tool_call_id: str, path: str
    ) -> ToolResult | None:
        """Redirect skill-path reads to the bash tool.

        The LLM occasionally tries to read SKILL.md or skill scripts through
        workspace. Skills live under $SKILLS_DIR, which workspace cannot
        access. Return a helpful error pointing at bash so the model
        self-corrects on retry.
        """
        normalized = path.strip().lstrip("./").replace("\\", "/")
        parts = normalized.split("/")
        first = parts[0] if parts else ""

        looks_like_skill = (
            normalized.endswith("SKILL.md")
            or first == "skills"
            or "/scripts/" in normalized
        )
        if not looks_like_skill:
            return None

        # Best-effort guess at the skill package name for the hint.
        if first == "skills" and len(parts) >= 2:
            skill_name = parts[1]
        elif first and first != "skills":
            skill_name = first
        else:
            skill_name = "<skill>"

        return ToolResult.execution_error(
            tool_call_id,
            (
                f"workspace cannot access skill files — they live under "
                f"$SKILLS_DIR, not artifacts/. Use the bash tool instead:\n"
                f'  cat "$SKILLS_DIR/{skill_name}/SKILL.md"\n'
                f"workspace is only for reading files under artifacts/."
            ),
        )