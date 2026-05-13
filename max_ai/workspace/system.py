"""
File System Workspace
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..base.workspace import WorkSpaceRegistry
from ..errors.workspace import WorkSpaceError
from ..loggers import ScopedLogger
from ..types.workspace import WorkspaceDirectory


logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["LocalWorkSpace"])


class LocalWorkSpace(WorkSpaceRegistry):
    """Local workspace serving as the runtime tmp area for files and skills."""

    def __init__(self) -> None:
        super().__init__(tag="LocalWorkSpace")

    def materialize_directories(
        self,
        user_id: str,
        root: str | Path | None = None,
    ) -> WorkspaceDirectory:
        """Create base runtime directories for this user."""
        return self.get_or_create_dirs(user_id=user_id, root=root)

    def save_or_upload(self, user_id: str, file_path: str | Path) -> str:
        """Copy a local file into the user's artifacts directory."""
        directory = self.materialize_directories(user_id)
        source = Path(file_path).expanduser().resolve()

        if not source.is_file():
            raise WorkSpaceError.no_file_exist(source)

        target = self._resolve_inside(directory.artifacts_dir, source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target.relative_to(directory.artifacts_dir).as_posix()

    def get_or_download(self, user_id: str, filename: str) -> Path:
        """Return a local artifacts file path."""
        directory = self.materialize_directories(user_id)
        target = self._resolve_inside(directory.artifacts_dir, filename)

        if not target.is_file():
            raise WorkSpaceError.no_workspace_file_exist(filename)
        return target

    def list_files(self, user_id: str) -> list[str]:
        """List files in the user's artifacts directory."""
        directory = self.materialize_directories(user_id)
        artifacts = directory.artifacts_dir.resolve()
        return [
            path.relative_to(artifacts).as_posix()
            for path in sorted(artifacts.rglob("*"))
            if path.is_file()
        ]

    @staticmethod
    def _resolve_inside(root: Path, relative_path: str | Path) -> Path:
        target = (root / relative_path).expanduser().resolve()
        root_resolved = root.resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError:
            raise WorkSpaceError.out_of_workspace() from None
        return target
