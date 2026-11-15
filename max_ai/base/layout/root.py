"""The ``.agent/`` folder inside an environment:

    <where the sandbox puts it>/.agent/
        setting.json
        skills/
        workspace/
        tmp/
"""

from __future__ import annotations

import typing as t
from pathlib import Path, PurePosixPath

# Read once from the package, then sent to the environment.
SCRIPT = Path(__file__).with_name("script.py").read_text()


class Layout:
    """What ``.agent/`` holds. The sandbox says where it goes (``base``) and
    runs ``SCRIPT`` with ``skeleton()`` to create it."""

    FOLDER: t.ClassVar[str] = ".agent"
    SETTINGS: t.ClassVar[str] = "setting.json"

    def __init__(self, base: str | PurePosixPath) -> None:
        self.root = PurePosixPath(base) / self.FOLDER

    @property
    def settings(self) -> PurePosixPath:
        return self.root / self.SETTINGS

    @property
    def skills(self) -> PurePosixPath:
        return self.root / "skills"

    @property
    def workspace(self) -> PurePosixPath:
        return self.root / "workspace"

    @property
    def tmp(self) -> PurePosixPath:
        return self.root / "tmp"

    def skeleton(self) -> dict[str, t.Any]:
        """What ``SCRIPT`` needs to create ``.agent/``: the folders, and the
        settings file with its first content."""
        return {
            "folders": [str(self.skills), str(self.workspace), str(self.tmp)],
            "settings": str(self.settings),
            "content": {"skills": {}},
        }
