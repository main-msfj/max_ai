"""
Filesystem-backed skill registry.

Reads skill packages from a local source directory. To stay
behaviorally identical to remote-backed registries (S3, Git, HTTP),
each ``load()`` call materializes the requested skills into a
dedicated temporary directory and hands that path to the loader. The
loader never touches the original source — it only sees the tmp copy.

This makes ``LocalSkillRegistry`` a faithful stand-in during
development for what production registries do: download to tmp, parse
from tmp, serve resources from tmp.

The tmp directory is created on ``connect()`` and is NOT cleaned up
by the registry itself. Cleanup is the caller's responsibility (or a
separate pipeline). The agent doesn't need to know or care.
"""

from __future__ import annotations

import shutil
import tempfile
import typing as t
from pathlib import Path

from ...base.skill import CoreSkillRegistry
from .skill_loader import load_skill_from_dir

if t.TYPE_CHECKING:
    from ...types.skills import Skill, ResourceMeta



class LocalSkillRegistry(CoreSkillRegistry):
    """Resolve skills from a local source directory via tmp staging.

    Source layout (read-only)::

        source_path/
            pr_review/
                SKILL.md
                scripts/...
                references/...
            bug_triage/
                ...

    On ``load(names)`` each requested skill is copied to::

        {tmp_root}/{name}/    ← isolated, owned by this registry instance

    and the loader operates on the tmp copy. Resource paths recorded in
    each ``Skill`` therefore point at tmp files, not source files —
    which is consistent with what remote registries produce.
    """

    def __init__(
        self,
        name: str,
        source_path: str | Path,
        skills: list[str],
    ) -> None:
        super().__init__(name=name, skills=skills)
        self.source_path: Path = Path(source_path).expanduser().resolve()
        if not self.source_path.is_dir():
            raise FileNotFoundError(
                f"Skill source directory does not exist: {self.source_path}"
            )
        self._tmp_root: Path | None = None

    # -------- LIFECYCLE -----------------------------------------------------------
    async def connect(self) -> None:
        """Create the dedicated tmp directory for this registry instance."""
        if self._tmp_root is None:
            self._tmp_root = Path(tempfile.mkdtemp(prefix="maxai_skills_"))

    async def disconnect(self) -> None:
        # Cleanup is intentionally NOT performed here. A separate pipeline
        # owns lifecycle of the tmp space — the agent must not assume the
        # files are gone after disconnect.
        return None

    # -------- READ OPERATIONS -----------------------------------------------------------
    async def load(self, names: list[str]) -> list["Skill"]:
        """Stage each requested skill into tmp and load it.

        For every name in ``names`` this method:
          1. Validates the skill exists in ``source_path``.
          2. Copies the entire skill directory to ``{tmp_root}/{name}/``.
          3. Calls the generic loader against the tmp copy.

        The result is a list of ``Skill`` objects whose internal
        resource paths point at tmp files.
        """
        await self._ensure_connected()
        assert self._tmp_root is not None  # set by connect()

        if not isinstance(names, list):
            raise TypeError(f"names must be a list, got {type(names).__name__}")
        if not all(isinstance(n, str) and n for n in names):
            raise ValueError("names must be a list of non-empty strings")

        loaded: list["Skill"] = []
        for name in names:
            source_skill = self.source_path / name
            if not source_skill.is_dir():
                raise FileNotFoundError(
                    f"Skill {name!r} not found in source: {source_skill}"
                )

            tmp_skill = self._tmp_root / name
            # If a previous load left this tmp dir behind, replace it.
            # Keeps load() idempotent within a single registry lifecycle.
            if tmp_skill.exists():
                shutil.rmtree(tmp_skill)
            shutil.copytree(source_skill, tmp_skill)

            loaded.append(load_skill_from_dir(tmp_skill))
        return loaded

    # -------- RESOURCE ACCESS -----------------------------------------------------------
    async def _read_resource_impl(
        self, skill_name: str, resource: "ResourceMeta"
    ) -> str:
        """Read a resource file from the staged tmp copy.

        ``resource.path`` was already resolved to a tmp path by the
        loader at load time, so no additional staging is needed here.
        Validation that the resource belongs to a known skill happens
        upstream in the closure built by ``make_read_resource_tool``.
        """
        return resource.path.read_text(encoding="utf-8")
