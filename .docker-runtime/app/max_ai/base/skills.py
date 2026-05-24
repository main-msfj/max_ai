"""
Contracts for skill registries.
"""

from __future__ import annotations

import re
import shutil
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from pydantic import BaseModel

from .capability import CoreAgentCapabilities
from ..config import setting
from ..core.blocks import SkillBlock
from ..types.workspace import WorkspaceDirectory


_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class CoreSkillRegistry(CoreAgentCapabilities[BaseModel], ABC):
    """Abstract base class for skill registries.

    A registry is bound to one source and a selected list of skills.
    It only fetches the selected skills, not the entire source.

    Attributes:
        source: The source location (path or URL) for skills.
        skills: List of skill names to be fetched from the source.

    The registry manages a local cache of skills and provides methods to:
    - Download and validate skills from the source
    - Load skill metadata
    - Materialize skills into a runtime workspace
    """

    def __init__(self, source: str | Path, skills: list[str]) -> None:
        """Initialize the skill registry.

        Args:
            source: Source location (file path or URL) where skills are stored.
            skills: List of skill names to fetch. Must be non-empty with valid names.

        Raises:
            TypeError: If source is not a string/Path or skills is not a list.
            ValueError: If skills list is empty, contains invalid names, or has duplicates.
        """
        super().__init__()
        self.source: str = self._validate_source(source)
        self.skills: list[str] = self._validate_skill_names(skills)
        self._skill_blocks: list[SkillBlock] = []
        self._prepared: bool = False

        self._cache_root: Path = self._resolve_cache_root()
        self._registry_cache_dir: Path = self._cache_root / self._registry_key()

    async def prepare(self) -> None:
        """Download/cache selected skills and load metadata.

        Downloads each skill from the source into the registry cache directory,
        validates the skill structure, and loads metadata from SKILL.md files.
        This method is idempotent - calling it multiple times has no effect
        after the first successful preparation.

        Raises:
            FileNotFoundError: If a skill's SKILL.md file is missing.
            NotADirectoryError: If a skill target is not a directory.
        """
        if self._prepared:
            return

        self._registry_cache_dir.mkdir(parents=True, exist_ok=True)

        blocks: list[SkillBlock] = []
        for skill_name in self.skills:
            target_dir = self._registry_cache_dir / skill_name

            if target_dir.exists():
                shutil.rmtree(target_dir)

            await self._download_skill(skill_name, target_dir)
            self._validate_skill_dir(skill_name, target_dir)
            blocks.append(self._read_skill_block(skill_name, target_dir))

        self._skill_blocks = blocks
        self._prepared = True

    async def get_skills(self) -> list[SkillBlock]:
        """Return selected skill metadata.

        Ensures the registry is prepared before returning skill metadata.

        Returns:
            List of SkillBlock objects containing metadata for each skill.
        """
        await self.prepare()
        return list(self._skill_blocks)

    def materialize(self, directory: WorkspaceDirectory) -> Path:
        """Copy prepared selected skills into the runtime skill directory.

        Copies all prepared skills from the cache into the specified workspace
        directory, making them available for runtime execution.

        Args:
            directory: Workspace directory containing the target skill_dir.

        Returns:
            Path to the materialized skill directory.

        Raises:
            RuntimeError: If called before prepare() completes.
        """
        if not self._prepared:
            raise RuntimeError("Skill registry must be prepared before materialize().")

        directory.skill_dir.mkdir(parents=True, exist_ok=True)

        for skill_name in self.skills:
            source_dir = self._registry_cache_dir / skill_name
            target_dir = directory.skill_dir / skill_name

            if target_dir.exists():
                shutil.rmtree(target_dir)

            shutil.copytree(source_dir, target_dir)

        return directory.skill_dir

    @abstractmethod
    async def _download_skill(self, skill_name: str, target_dir: Path) -> None:
        """Download/copy one selected skill from source into target_dir.

        Subclasses must implement this to define how skills are retrieved
        from their specific source type (e.g., filesystem, git, HTTP).

        Args:
            skill_name: Name of the skill to download.
            target_dir: Destination directory for the skill files.
        """
        ...

    @staticmethod
    def _validate_source(source: str | Path) -> str:
        """Validate and normalize the source parameter.

        Args:
            source: Source location to validate.

        Returns:
            String representation of the source.

        Raises:
            TypeError: If source is not a string or Path, or is empty.
        """
        if not isinstance(source, (str, Path)) or not str(source):
            raise TypeError("source must be a non-empty string or Path")
        return str(source)

    @staticmethod
    def _validate_skill_names(skills: list[str]) -> list[str]:
        """Validate the list of skill names.

        Checks that:
        - skills is a non-empty list of strings
        - Each skill name contains only allowed characters (letters, digits, _, -)
        - No duplicate skill names exist

        Args:
            skills: List of skill names to validate.

        Returns:
            The validated list of skill names.

        Raises:
            TypeError: If skills is not a list.
            ValueError: If skills is empty, contains invalid names, or has duplicates.
        """
        if not isinstance(skills, list):
            raise TypeError(f"skills must be a list, got {type(skills).__name__}")
        if not skills:
            raise ValueError("skills list cannot be empty")
        if not all(isinstance(s, str) and s for s in skills):
            raise ValueError("skills must be a list of non-empty strings")

        for skill in skills:
            if not _VALID_NAME_RE.match(skill):
                raise ValueError(
                    f"Invalid skill name {skill!r}. Allowed characters: "
                    "letters, digits, underscores, hyphens."
                )

        dupes = {skill for skill in skills if skills.count(skill) > 1}
        if dupes:
            raise ValueError(f"Duplicate skill names: {sorted(dupes)}")

        return list(skills)

    @staticmethod
    def _validate_skill_dir(skill_name: str, path: Path) -> None:
        """Validate that a skill directory has the required structure.

        Checks that:
        - The directory exists and is a directory
        - A SKILL.md file is present

        Args:
            skill_name: Name of the skill being validated.
            path: Path to the skill directory.

        Raises:
            FileNotFoundError: If the directory or SKILL.md doesn't exist.
            NotADirectoryError: If path is not a directory.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Skill {skill_name!r} was not downloaded into {path}."
            )
        if not path.is_dir():
            raise NotADirectoryError(
                f"Skill {skill_name!r} target is not a directory: {path}"
            )

        skill_file = path / "SKILL.md"
        if not skill_file.is_file():
            raise FileNotFoundError(
                f"Skill {skill_name!r} is missing SKILL.md at {skill_file}."
            )

    def _read_skill_block(self, skill_name: str, skill_dir: Path) -> SkillBlock:
        """Read and parse a skill's metadata from SKILL.md.

        Extracts YAML frontmatter from the SKILL.md file to create a SkillBlock.
        Falls back to the skill_name if no name is in the metadata.

        Args:
            skill_name: Name of the skill.
            skill_dir: Directory containing the skill files.

        Returns:
            SkillBlock containing the skill's metadata.
        """
        skill_file = skill_dir / "SKILL.md"
        raw = skill_file.read_text(encoding="utf-8")
        metadata = self._parse_frontmatter(raw)

        return SkillBlock(
            name=metadata.get("name") or skill_name,
            description=metadata.get("description") or "",
        )

    @staticmethod
    def _parse_frontmatter(text: str) -> dict[str, t.Any]:
        """Parse YAML frontmatter from markdown text.

        Extracts and parses the YAML block delimited by --- markers
        at the beginning of the text.

        Args:
            text: Markdown text potentially containing frontmatter.

        Returns:
            Dictionary of frontmatter data, or empty dict if none found.
        """
        if not text.startswith("---"):
            return {}

        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}

        data = yaml.safe_load(parts[1]) or {}
        if not isinstance(data, dict):
            return {}

        return data

    @staticmethod
    def _resolve_cache_root() -> Path:
        """Resolve and create the root cache directory for skills.

        Returns:
            Path to the skills cache directory (creates if needed).
        """
        cache_root = setting.root_dir / "var" / "skills"
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root

    def _registry_key(self) -> str:
        """Generate a unique cache key for this registry instance.

        Creates a key based on the registry class name and source,
        sanitizing the source string to be filesystem-safe.

        Returns:
            String key suitable for use as a directory name.
        """
        source_key = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.source).strip("-")
        return f"{type(self).__name__}-{source_key or 'default'}"
