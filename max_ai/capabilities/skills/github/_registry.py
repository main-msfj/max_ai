"""Skill registry backed by a GitHub repository.

The whole repository is shallow-cloned once per registry instance (into the
skills cache, alongside the per-skill copies ``CoreSkillBase`` already
manages); each selected skill is then copied out of that clone exactly like
``LocalSkillRegistry`` copies from a local directory. A private repo needs
only an environment variable name (``token_env``) — the token itself is
never stored, serialized, or placed on a command line.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from pathlib import Path

from ....base.skills import CoreSkillBase
from ....core.executor.process import run_process
from ._model import GithubSkillRegistryConfig


class GithubSkillRegistry(CoreSkillBase):
    """Skills cloned from a GitHub repository with ``git``.

    ``source`` is the repository: a full URL (``https://github.com/org/repo``,
    optionally ending in ``.git``) or the short ``org/repo`` form, expanded
    against github.com. Each name in ``skills`` must be a top-level directory
    in that repo containing a ``SKILL.md``, exactly like ``LocalSkillRegistry``.
    When the skills live deeper, ``path`` names their folder inside the repo
    (e.g. ``path="skills"`` for ``anthropics/skills``).

    ``ref`` must be a branch or tag (``git clone --branch`` doesn't accept an
    arbitrary commit); pin a tag for reproducibility.

    Private repos: set ``token_env`` to the name of an environment variable
    holding a token with read access to the repo's contents (a fine-grained
    GitHub PAT is enough). The token is read once, at clone time, from the
    environment — never from a constructor argument, never written to the
    clone's git config, and never part of the process's command line (it's
    passed to ``git`` through ``GIT_CONFIG_*`` environment variables, which
    aren't visible to other users the way argv is).
    """

    component_schema = GithubSkillRegistryConfig
    component_type = "skills"
    component_provider_override = "max_ai.capabilities.skills.github.GithubSkillRegistry"

    def __init__(
        self, source: str, skills: list[str], *, ref: str = "main",
        path: str = "", token_env: str | None = None,
    ) -> None:
        """Initialize ``GithubSkillRegistry``.

Parameters
----------
source : str
    Value supplied for ``source``.
skills : list[str]
    Value supplied for ``skills``.
ref : str
    Value supplied for ``ref``.
path : str
    Value supplied for ``path``.
token_env : str | None
    Value supplied for ``token_env``."""
        super().__init__(source=source, skills=skills)
        self.ref = self.require_type(ref, str, "ref")
        if not self.ref.strip():
            raise ValueError("ref cannot be empty")
        self.path = self._validate_path(path)
        self.token_env = token_env
        # The base keyed the cache on source alone; ref and path must not share it.
        key = f"{self._registry_key()}-{self._variant()}"
        self._registry_cache_dir = self._cache_root / key
        self._clone_dir = self._cache_root / f"{key}-repo"
        self._clone_lock = asyncio.Lock()
        self._cloned = False

    def _to_config(self) -> GithubSkillRegistryConfig:
        """Build the serializable configuration for ``GithubSkillRegistry``."""
        return GithubSkillRegistryConfig(
            source=self.source, skills=list(self.skills),
            ref=self.ref, path=self.path, token_env=self.token_env,
        )

    @classmethod
    def _from_config(cls, config: GithubSkillRegistryConfig) -> "GithubSkillRegistry":
        """Create an instance from its configuration for ``GithubSkillRegistry``.

Parameters
----------
config : GithubSkillRegistryConfig
    Value supplied for ``config``."""
        return cls(
            source=config.source, skills=config.skills,
            ref=config.ref, path=config.path, token_env=config.token_env,
        )

    def _variant(self) -> str:
        """Short, filesystem-safe fingerprint of ``ref`` and ``path``."""
        return hashlib.sha256(f"{self.ref}\0{self.path}".encode()).hexdigest()[:10]

    @staticmethod
    def _validate_path(path: str) -> str:
        """Normalize ``path`` to a relative POSIX path that stays inside the repo."""
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
        if path.startswith(("/", "\\")) or ".." in parts:
            raise ValueError(f"path must be relative and stay inside the repo, got {path!r}")
        return "/".join(parts)

    @staticmethod
    def _repo_url(source: str) -> str:
        """Perform the internal ``repo url`` operation for ``GithubSkillRegistry``.

Parameters
----------
source : str
    Value supplied for ``source``."""
        if source.startswith(("http://", "https://", "git@", "ssh://", "file://")):
            return source
        if "/" not in source or source.count("/") != 1:
            raise ValueError(
                f"source must be a URL or 'org/repo', got {source!r}"
            )
        return f"https://github.com/{source}.git"

    async def _clone(self) -> Path:
        """Shallow-clone the repo once; later calls reuse the same checkout."""
        async with self._clone_lock:
            if self._cloned:
                return self._clone_dir
            if shutil.which("git") is None:
                raise RuntimeError("git is required for GithubSkillRegistry but isn't installed")
            if self._clone_dir.exists():
                shutil.rmtree(self._clone_dir)

            env = os.environ.copy()
            if self.token_env:
                token = os.environ.get(self.token_env)
                if not token or not token.strip():
                    raise ValueError(f"Set {self.token_env} to a GitHub token")
                # Passed via GIT_CONFIG_* env vars, never argv or a config file.
                env["GIT_CONFIG_COUNT"] = "1"
                env["GIT_CONFIG_KEY_0"] = "http.extraheader"
                env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: bearer {token}"

            result = await run_process(
                ["git", "clone", "--depth", "1", "--branch", self.ref, "--single-branch",
                 self._repo_url(self.source), str(self._clone_dir)],
                env=env, timeout=120,
            )
            if result.exit_code != 0:
                raise RuntimeError(f"git clone failed: {result.stderr or 'unknown error'}")
            self._cloned = True
            return self._clone_dir

    async def _download_skill(self, skill_name: str, target_dir: Path) -> None:
        """Copy ``{skill_name}/`` out of the cloned repo into target_dir."""
        skills_root = (await self._clone()) / self.path
        skill_src = skills_root / skill_name
        where = f"{self.source}@{self.ref}" + (f" under {self.path!r}" if self.path else "")

        if not skills_root.is_dir():
            raise FileNotFoundError(f"Folder {self.path!r} not found in {self.source}@{self.ref}.")
        if not skill_src.exists():
            available = sorted(p.name for p in skills_root.iterdir() if p.is_dir() and p.name != ".git")
            raise FileNotFoundError(
                f"Skill {skill_name!r} not found in {where}. Available directories: {available}"
            )
        if not skill_src.is_dir():
            raise NotADirectoryError(f"Skill {skill_name!r} at {skill_src} is not a directory.")

        shutil.copytree(skill_src, target_dir)


__all__ = ["GithubSkillRegistry", "GithubSkillRegistryConfig"]
