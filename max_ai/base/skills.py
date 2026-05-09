"""
Contracts for skill registries.

Skills are self-contained packages (instructions + scripts + resources)
that live in an external source — a local directory, a Git repo, a
blob storage container, or anywhere else. A registry knows how to:

  1. Resolve which skills exist in its source.
  2. Download them to a shared on-disk cache (idempotent).
  3. Materialize them into a per-user directory that the agent's
     sandbox can mount read-only.

Two filesystem locations matter. They default under the shared server
workspace and can be overridden with environment variables:

  SKILLS_CACHE_DIR   shared cache, persistent across sessions
  SERVER_DIR/tmp     root for ephemeral per-user runtime dirs

Skills are cached under ``serverWorkspace/var/skills-cache/{registry_key}/{skill}/``
where ``registry_key`` is derived from the child registry class name.
Runtime copies live under ``serverWorkspace/tmp/{user_id}/skills/{skill}/``.

Validation is eager: instantiating a registry verifies that every
declared skill is reachable from its source and lands correctly in
the cache. A broken or unreachable skill fails at construction time,
never silently at runtime.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from pydantic import BaseModel

from ..config import setting
from .tools import CoreTool
from ..core.blocks import SkillBlock
from .capability import CoreAgentCapabilities
from ..tools.skills import SearchSkillsTool, SkillBashTool

if t.TYPE_CHECKING:
    pass


_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class CoreSkillRegistry(CoreAgentCapabilities[BaseModel], ABC):
    """Abstract base class for skill registries.

    A registry is bound to a single ``source`` (local dir, Git URL,
    blob container, ...) and a fixed list of skill names it manages.
    Subclasses implement ``_download_if_not_exist`` with backend-
    specific logic (file copy, git clone, blob download, ...).

    The base class owns the lifecycle (validation, cache layout,
    session materialization, cleanup) so every backend behaves
    identically from the agent's point of view.
    """

    def __init__(self, source: str | Path, skills: list[str]) -> None:
        super().__init__()
        self.source: str = self._validate_source(source)
        self.skills: list[str] = self._validate_skill_names(skills)

        self._cache_root: Path = self._resolve_cache_root()
        self._runtime_root: Path = self._resolve_runtime_root()
        self._registry_cache_dir: Path = self._cache_root / self._registry_key()

    # -------- VALIDATION ---------------------------------------------------------------

    @staticmethod
    def _validate_source(source: str | Path) -> str:
        if not isinstance(source, (str, Path)) or not str(source):
            raise TypeError("source must be a non-empty string or Path")
        return str(source)

    @staticmethod
    def _validate_skill_names(skills: list[str]) -> list[str]:
        if not isinstance(skills, list):
            raise TypeError(f"skills must be a list, got {type(skills).__name__}")
        if not skills:
            raise ValueError("skills list cannot be empty")
        if not all(isinstance(s, str) and s for s in skills):
            raise ValueError("skills must be a list of non-empty strings")
        for s in skills:
            if not _VALID_NAME_RE.match(s):
                raise ValueError(
                    f"Invalid skill name {s!r}. Allowed characters: "
                    "letters, digits, underscores, hyphens."
                )
        dupes = {s for s in skills if skills.count(s) > 1}
        if dupes:
            raise ValueError(f"Duplicate skill names: {sorted(dupes)}")
        return list(skills)

    # -------- ENV-DRIVEN PATHS ---------------------------------------------------------

    @staticmethod
    def _resolve_cache_root() -> Path:
        override = os.environ.get("SKILLS_CACHE_DIR")
        if override:
            path = Path(override).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path
        return setting.get_or_create_skill_cache_dir()

    @staticmethod
    def _resolve_runtime_root() -> Path:
        server_dir = os.environ.get("SERVER_DIR")
        if server_dir:
            path = Path(server_dir).expanduser().resolve() / "tmp"
            path.mkdir(parents=True, exist_ok=True)
            return path
        return setting.get_or_create_server_tmp_dir()

    def _registry_key(self) -> str:
        """Stable, filesystem-safe key derived from the child registry class.

        Groups cache entries by backend type (LocalSkillRegistry,
        GitSkillRegistry, etc.).
        """
        return (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", type(self).__name__).strip("_")
            or "SkillRegistry"
        )

    # -------- LIFECYCLE ----------------------------------------------------------------

    async def connect(self) -> None:
        """Eager validation: ensure every declared skill is in cache.

        Called once via ``CoreAgentCapabilities._ensure_connected``.
        Failure here means the registry cannot serve any session, so
        we raise loudly rather than defer the error to materialize().
        """
        self._registry_cache_dir.mkdir(parents=True, exist_ok=True)

        for skill_name in self.skills:
            cache_path = self._registry_cache_dir / skill_name
            if not cache_path.exists():
                await self._download_if_not_exist(skill_name, cache_path)

            if not cache_path.is_dir():
                raise RuntimeError(
                    f"Skill {skill_name!r} did not materialize correctly "
                    f"in cache at {cache_path}. Backend "
                    f"{type(self).__name__} returned without error but "
                    "the directory is missing."
                )

            skill_md = cache_path / "SKILL.md"
            if not skill_md.is_file():
                raise RuntimeError(
                    f"Skill {skill_name!r} is missing SKILL.md "
                    f"at {skill_md}. Every skill must declare one."
                )

    async def disconnect(self) -> None:
        """No-op by default. Cache is shared and persists across runs."""
        return None

    async def list_skill_blocks(self) -> list[SkillBlock]:
        """Return lightweight ``SkillBlock`` entries for prompt discovery.

        Only ``name`` and ``description`` are read from each skill's
        ``SKILL.md`` frontmatter. ``instructions`` is intentionally empty:
        the model should inspect the full ``SKILL.md`` on demand through
        ``skill_bash`` when it decides a skill applies.
        """
        await self._ensure_connected()

        blocks: list[SkillBlock] = []
        for skill_name in self.skills:
            skill_md = self._registry_cache_dir / skill_name / "SKILL.md"
            meta = self._read_skill_frontmatter(skill_md)

            name = meta.get("name") or skill_name
            description = meta.get("description") or ""
            if not isinstance(name, str) or not name:
                raise RuntimeError(f"Skill {skill_name!r} has invalid name metadata.")
            if not isinstance(description, str):
                raise RuntimeError(
                    f"Skill {skill_name!r} has invalid description metadata."
                )

            blocks.append(SkillBlock(name=name, description=description))

        return blocks

    @staticmethod
    def _read_skill_frontmatter(skill_md: Path) -> dict[str, t.Any]:
        text = skill_md.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return {}

        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}

        data = yaml.safe_load(parts[1]) or {}
        if not isinstance(data, dict):
            return {}
        return data

    # -------- BACKEND HOOK -------------------------------------------------------------

    @abstractmethod
    async def _download_if_not_exist(
        self, skill_name: str, target_dir: Path
    ) -> None:
        """Bring one skill from ``self.source`` into ``target_dir``.

        Called by the base class only when ``target_dir`` does not
        already exist. Implementations must materialize the full
        skill (SKILL.md + scripts + resources) under ``target_dir``.

        Args:
            skill_name: Name of the skill to fetch.
            target_dir: Absolute path where the skill should land.
                Will not exist when this is called; the implementation
                is responsible for creating it.

        Raises:
            Any backend-specific error if the skill cannot be fetched.
            The base class will surface these as registry failures.
        """
        ...

    # -------- USER MATERIALIZATION --------------------------------------------------

    def materialize(
        self,
        user_id: str,
        session_id: str | None = None,
        skills: list[str] | None = None,
    ) -> Path:
        """Copy a subset of cached skills into a per-user runtime dir.

        The returned path is meant to be bind-mounted read-only into
        the user's sandbox and exposed as ``$SKILLS_DIR``.

        Args:
            user_id: Identifier of the user whose session is being set
                up. Must be filesystem-safe.
            session_id: Ignored. Kept temporarily for backwards compatibility
                while runtime layout moves from session scoped to user scoped.
            skills: Optional subset of ``self.skills`` to expose to
                this user. Defaults to all of them. Useful for
                multi-tenant setups where each user only sees their
                own skills.

        Returns:
            Absolute path to the user's skills directory.
        """
        if not isinstance(user_id, str) or not _VALID_NAME_RE.match(user_id):
            raise ValueError(
                f"Invalid user_id {user_id!r}. Allowed characters: "
                "letters, digits, underscores, hyphens."
            )
        wanted = skills if skills is not None else self.skills
        unknown = [s for s in wanted if s not in self.skills]
        if unknown:
            raise ValueError(
                f"Skills {unknown} are not part of this registry. "
                f"Available: {self.skills}"
            )

        user_skills_dir = self._user_skills_dir(user_id)
        user_skills_dir.mkdir(parents=True, exist_ok=True)

        for skill_name in wanted:
            src = self._registry_cache_dir / skill_name
            dst = user_skills_dir / skill_name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

        return user_skills_dir

    def _user_skills_dir(self, user_id: str) -> Path:
        if not isinstance(user_id, str) or not _VALID_NAME_RE.match(user_id):
            raise ValueError(
                f"Invalid user_id {user_id!r}. Allowed characters: "
                "letters, digits, underscores, hyphens."
            )
        return self._runtime_root / user_id / "skills"

    def make_skill_bash_tool(self) -> "CoreTool":
        """Build the user-scoped ``skill_bash`` tool.

        The tool is intentionally owned by the skill registry: agents
        without a ``CoreSkillRegistry`` never receive it. It exposes the
        current user's materialized skills directory via ``$SKILLS_DIR``
        and runs commands from that directory so the model can inspect
        ``SKILL.md`` files, references, scripts, and other packaged assets.
        """
        return SkillBashTool()

    def make_search_skills_tool(self) -> "CoreTool":
        """Build the searchable skill catalog tool."""
        return SearchSkillsTool(self._skill_entries())

    def _skill_entries(self) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for skill_name in self.skills:
            skill_md = self._registry_cache_dir / skill_name / "SKILL.md"
            meta = self._read_skill_frontmatter(skill_md) if skill_md.is_file() else {}
            name = meta.get("name") or skill_name
            description = meta.get("description") or ""
            entries.append(
                {
                    "name": str(name),
                    "description": str(description),
                    "path": f"{skill_name}/SKILL.md",
                }
            )
        return entries

    @property
    def tools(self) -> list["CoreTool"]:
        """Tools exposed by the skill registry."""
        return [self.make_search_skills_tool(), self.make_skill_bash_tool()]

    def cleanup(self, user_id: str) -> None:
        """Remove a user's session directory entirely.

        Cache is left intact — only the per-session copies are
        deleted. Safe to call multiple times; missing dirs are
        ignored.
        """
        if not isinstance(user_id, str) or not _VALID_NAME_RE.match(user_id):
            raise ValueError(f"Invalid user_id {user_id!r}.")

        user_root = self._runtime_root / user_id
        if user_root.exists():
            shutil.rmtree(user_root, ignore_errors=True)
