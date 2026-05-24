"""Core contract for agent Workspace."""

from __future__ import annotations

import logging
import typing as t
from pathlib import Path
from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..config import setting
from ..loggers import ScopedLogger
from ..types.workspace import WorkspaceDirectory
from .capability import CoreAgentCapabilities


logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["Workspace"])


class WorkSpaceRegistry(CoreAgentCapabilities[BaseModel], ABC):
    """Abstract base for workspace registries.

    Layout contract — the host and container share the same subdirectory
    names so paths translate cleanly across the bind mount:

        HOST                                  CONTAINER
        <root>/<user_id>/             ──►     /mnt/
        <root>/<user_id>/tools/       ──►     /mnt/tools/
        <root>/<user_id>/skills/      ──►     /mnt/skills/
        <root>/<user_id>/artifacts/   ──►     /mnt/artifacts/

    The user_id segment exists only on the host (multi-tenant filesystem).
    The container is single-tenant per session, so the mount drops the
    user_id and exposes the subdirectories directly under /mnt.
    """

    CATEGORIES = {"tools", "skills", "artifacts"}

    def __init__(self, tag: str, root: str | Path | None = None) -> None:
        super().__init__()
        self.tag = tag
        self.base_root = Path(root).expanduser().resolve() if root else setting.root_dir
        self.root: Path | None = None
        self.skills_dir: Path | None = None
        self.tool_dir: Path | None = None
        self.artifacts_dir: Path | None = None

    def materialize(self, user_id: str) -> WorkspaceDirectory:
        """Create and return the per-user runtime directories.

        Default implementation defers to get_or_create_dirs. Subclasses
        that need to do additional work (uploading to remote storage,
        seeding files, etc.) can override.
        """
        return self.get_or_create_dirs(user_id, self.base_root)

    def map_directory(
        self, user_id: str, root: Path | str | None = None
    ) -> WorkspaceDirectory:
        """Compute the per-user runtime layout on the host."""
        user_id = user_id
        base = self.base_root

        runtime_root = base / user_id

        directory = WorkspaceDirectory(
            root=runtime_root,
            tool_dir=runtime_root / setting.tool_dir,
            skill_dir=runtime_root / setting.skill_dir,
            artifacts_dir=runtime_root / setting.artifacts_dir,
        )

        # Cache the last layout for convenience accessors.
        self.root = directory.root
        self.tool_dir = directory.tool_dir
        self.skills_dir = directory.skill_dir
        self.artifacts_dir = directory.artifacts_dir
        return directory

    def get_or_create_dirs(
        self, user_id: str, root: str | Path | None = None
    ) -> WorkspaceDirectory:
        """Compute the layout and ensure all directories exist."""
        base = self.map_directory(user_id, root)
        kwargs = dict(parents=True, exist_ok=True)
        base.root.mkdir(**kwargs)
        base.tool_dir.mkdir(**kwargs)
        base.skill_dir.mkdir(**kwargs)
        base.artifacts_dir.mkdir(**kwargs)
        return base

    @abstractmethod
    def save_or_upload(self, user_id: str, file_path: str | Path) -> str:
        """Persist a local file into the user's artifacts area."""
        ...

    @abstractmethod
    def get_or_download(self, user_id: str, filename: str) -> t.Any:
        """Retrieve an artifact by filename."""
        ...

    @abstractmethod
    def list_files(self, user_id: str) -> list[str]:
        """List the user's artifact filenames."""
        ...
