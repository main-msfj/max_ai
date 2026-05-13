"""
Core Contract for agent Workspace

"""

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
    """Abstract Base for Workspace"""

    CAGTEGORIES = {"tools", "skills", "artifacts"}

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag
        self.root: Path | None = None
        self.skills_dir: Path | None = None
        self.tool_dir: Path | None = None
        self.artifacts_dir: Path | None = None

    def map_directory(
        self, user_id: str, root: Path | str | None
    ) -> WorkspaceDirectory:
        """Mapppung tmp folder"""
        root = Path(root) if root else setting.root_dir
        user_id = user_id if user_id else "default"

        # Encapsulate DIR
        directory = WorkspaceDirectory(
            root=root / "tmp" / user_id,
            tool_dir=root / setting.tool_dir.format(user_id=user_id),
            skill_dir=root / setting.skill_dir.format(user_id=user_id),
            artifacts_dir=root / setting.artifacts_dir.format(user_id=user_id),
        )

        # Keep on Memory
        self.root = directory.root
        self.tool_dir = directory.tool_dir
        self.skills_dir = directory.skill_dir
        self.artifacts_dir = directory.artifacts_dir
        return directory

    def get_or_create_dirs(
        self, user_id: str , root: str | Path | None
    ) -> WorkspaceDirectory:
        """Create Directory"""
        kwargs = dict(parents=True, exist_ok=True)
        base = self.map_directory(user_id, root)
        base.tool_dir.mkdir(**kwargs)
        base.skill_dir.mkdir(**kwargs)
        base.artifacts_dir.mkdir(**kwargs)
        return base
    
    @abstractmethod
    def save_or_upload(self, user_id: str, file_path: str | Path) -> str:
        """Function that Save or Upload Files"""
        ...

    @abstractmethod
    def get_or_download(self, user_id: str, filename: str) -> t.Any:
        """"Download target file from Workspace"""
        ...

    @abstractmethod
    def list_files(self, user_id: str) -> list[str]:
        """List all files in workspace"""
