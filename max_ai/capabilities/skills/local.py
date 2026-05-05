"""
Local-filesystem skill registry.

Source is a directory on the same machine. Skills are subdirectories
inside that source, each one a self-contained package with at least
a SKILL.md.

This is the simplest backend and the one used in dev environments
where you keep skills next to your code, or as the destination of a
``git clone`` you manage outside the registry.

Caching behavior: skills are copied from ``source`` to the shared
cache (``SKILLS_CACHE_DIR``) on first use. Subsequent registries
pointing at the same source skip the copy. To pick up changes from
source after editing, clear the cache directory manually — the
registry treats cache as authoritative once populated.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ...base.skills import CoreSkillRegistry


class LocalSkillRegistry(CoreSkillRegistry):
    """Skill registry backed by a local directory.

    Expects ``source`` to be an existing directory whose immediate
    children are skill folders:

        source/
        ├── docx-creator/
        │   ├── SKILL.md
        │   └── scripts/
        ├── pdf-creator/
        │   └── SKILL.md
        └── ...

    Each declared skill must exist as a direct subdirectory of
    ``source``; nested layouts are not supported.
    """

    def __init__(self, source: str | Path, skills: list[str]) -> None:
        super().__init__(source=source, skills=skills)
        self._source_path: Path = self._resolve_source_path(self.source)

    @staticmethod
    def _resolve_source_path(source: str) -> Path:
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Source directory does not exist: {path}"
            )
        if not path.is_dir():
            raise NotADirectoryError(
                f"Source must be a directory, got a file: {path}"
            )
        return path

    async def _download_if_not_exist(
        self, skill_name: str, target_dir: Path
    ) -> None:
        """Copy ``{source}/{skill_name}/`` into the cache.

        Called by the base class only when ``target_dir`` does not
        exist yet. The base class verifies the result afterwards
        (directory present, SKILL.md present), so this method can
        focus on the copy itself.
        """
        skill_src = self._source_path / skill_name

        if not skill_src.exists():
            available = sorted(
                p.name for p in self._source_path.iterdir() if p.is_dir()
            )
            raise FileNotFoundError(
                f"Skill {skill_name!r} not found in source {self._source_path}. "
                f"Available skills in source: {available}"
            )
        if not skill_src.is_dir():
            raise NotADirectoryError(
                f"Skill {skill_name!r} at {skill_src} is not a directory."
            )

        # copytree expects the destination to NOT exist; the base
        # class guarantees this.
        shutil.copytree(skill_src, target_dir)