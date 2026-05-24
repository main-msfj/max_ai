"""Local filesystem workspace registry."""

from __future__ import annotations

import shutil
from pathlib import Path

from ...base.workspace import WorkSpaceRegistry


class WorkspaceLocal(WorkSpaceRegistry):
    """Workspace registry backed by the local filesystem.

    Owns the per-user runtime layout under the configured server root:

        {root}/{user_id}/tools
        {root}/{user_id}/skills
        {root}/{user_id}/artifacts

    The artifacts directory is where generated outputs and editable user
    files live. Skills and executors ask this registry for the layout
    instead of each one constructing paths independently.
    """

    def __init__(self, root: str | Path | None = None, tag: str = "local") -> None:
        super().__init__(tag=tag, root=root)

    def save_or_upload(self, user_id: str, file_path: str | Path) -> str:
        """Copy a local file into the user's artifacts and return its relative path."""
        directory = self.materialize(user_id)
        source = Path(file_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Workspace source file does not exist: {source}")

        target = self._resolve_inside(directory.artifacts_dir, source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target:
            shutil.copy2(source, target)
        return self._relative(target, directory.artifacts_dir)

    def get_or_download(self, user_id: str, filename: str) -> Path:
        """Return the local path of an artifact, raising if it doesn't exist.

        On local backends nothing is downloaded — the file is already on
        disk. Remote backends override this method to fetch from storage.
        """
        directory = self.materialize(user_id)
        target = self._resolve_inside(directory.artifacts_dir, filename)
        if not target.is_file():
            raise FileNotFoundError(f"Workspace file does not exist: {filename}")
        return target

    def list_files(self, user_id: str) -> list[str]:
        """List files under the user's artifacts as relative POSIX paths."""
        directory = self.materialize(user_id)
        workspace = directory.artifacts_dir.resolve()
        return [
            self._relative(path, workspace)
            for path in sorted(workspace.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        ]

    def get_artifacts_dir(self, user_id: str) -> Path:
        """Return the materialized artifacts directory for this user."""
        return self.materialize(user_id).artifacts_dir

    def get_skills_dir(self, user_id: str) -> Path:
        """Return the materialized skills directory for this user."""
        return self.materialize(user_id).skill_dir

    def get_tools_dir(self, user_id: str) -> Path:
        """Return the materialized tools directory for this user."""
        return self.materialize(user_id).tool_dir

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
