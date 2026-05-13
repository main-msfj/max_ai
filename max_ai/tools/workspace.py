"""Tool for reading and editing generated artifacts."""

from __future__ import annotations

import os
import re
import typing as t
from pathlib import Path

from ..base.tools import CoreTool, ToolContext
from ..config import setting
from ..types.tool_call import ToolCallRecord, ToolResult
from ..types.tools import DockerToolRef, ToolApprovalMode


_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class WorkspaceTool(CoreTool):
    """Read and edit files under ``tmp/<user_id>/artifacts``."""

    def __init__(self, timeout_seconds: float = 60) -> None:
        super().__init__(
            name="workspace",
            description=(
                "List, read, write, and delete files in the current user's "
                "artifacts directory. Use this for documents and outputs "
                "created for the user."
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
                    "enum": ["list", "read", "write", "delete"],
                    "description": "Artifacts operation to perform.",
                },
                "path": {
                    "type": ["string", "null"],
                    "description": "Relative file or directory path inside artifacts.",
                },
                "content": {
                    "type": ["string", "null"],
                    "description": "Text content for write operations.",
                },
                "overwrite": {
                    "type": ["boolean", "null"],
                    "description": "Whether write may replace an existing file.",
                },
            },
            "required": ["action"],
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
                    "path is required for read, write, and delete actions.",
                )
            target = self._resolve_inside(artifacts, path)

            if action == "read":
                if not target.is_file():
                    return ToolResult.execution_error(
                        tool_request.id,
                        f"Artifacts file does not exist: {path}",
                    )
                return ToolResult.success_result(
                    tool_request.id,
                    {
                        "path": self._relative(target, artifacts),
                        "content": target.read_text(encoding="utf-8"),
                    },
                    metadata={"name": self.name},
                )

            if action == "write":
                content = tool_request.parameters.get("content")
                if not isinstance(content, str):
                    return ToolResult.invalid_parameters(
                        tool_request.id,
                        "content is required for write action.",
                    )
                overwrite = bool(tool_request.parameters.get("overwrite", True))
                if target.exists() and not overwrite:
                    return ToolResult.execution_error(
                        tool_request.id,
                        f"Artifacts file already exists: {path}",
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return ToolResult.success_result(
                    tool_request.id,
                    {
                        "path": self._relative(target, artifacts),
                        "bytes": len(content.encode("utf-8")),
                    },
                    metadata={"name": self.name},
                )

            if action == "delete":
                if not target.exists():
                    return ToolResult.success_result(
                        tool_request.id,
                        {"path": self._relative(target, artifacts), "deleted": False},
                        metadata={"name": self.name},
                    )
                if target.is_dir():
                    return ToolResult.execution_error(
                        tool_request.id,
                        "delete only supports files.",
                    )
                target.unlink()
                return ToolResult.success_result(
                    tool_request.id,
                    {"path": self._relative(target, artifacts), "deleted": True},
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
        user_id = cls._safe_user_id(tool_context.user_id)
        return (setting.root_dir / "tmp" / user_id / "artifacts").resolve()

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
        target = (root / relative_path).expanduser().resolve()
        root_resolved = root.resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError:
            raise ValueError("path must stay inside the artifacts directory.") from None
        return target

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
