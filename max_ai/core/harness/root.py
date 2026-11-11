"""A project's ``.agent/`` folder:

    <project>/.agent/
        setting.json
        skills/
        workspace/
"""

from __future__ import annotations

import json
import typing as t
from pathlib import Path

from ...config import setting


class Layout:
    """Where a project's ``.agent/`` lives and what it holds. Creating it
    never overwrites what already exists."""

    SETTINGS: t.ClassVar[str] = "setting.json"

    def __init__(self, project: Path | None = None) -> None:
        self.root = Path(project or setting.root_dir).expanduser().resolve() / ".agent"

    @property
    def settings(self) -> Path:
        return self.root / self.SETTINGS

    @property
    def skills(self) -> Path:
        return self.root / "skills"

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    def create_project(self) -> Path:
        """Create ``.agent/`` with ``setting.json``, ``skills/`` and ``workspace/``."""
        for folder in (self.skills, self.workspace):
            folder.mkdir(parents=True, exist_ok=True)
        # "x" fails if the file exists: two starts at once never overwrite.
        try:
            with self.settings.open("x", encoding="utf-8") as file:
                file.write(json.dumps({}, indent=2) + "\n")
        except FileExistsError:
            pass
        return self.root
