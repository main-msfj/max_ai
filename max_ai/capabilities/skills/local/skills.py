"""LocalSkills: skills that are folders on this machine."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

from pydantic import BaseModel, Field

from ....base.session_store import validate_id
from ....base.skills import SkillsCore
from ....errors.skills import SkillError


class LocalSkillsConfig(BaseModel):
    path: str
    skills: list[str]
    skill_env: dict[str, str] = Field(default_factory=dict)


class LocalSkills(SkillsCore):
    """The skills ``<path>/<name>/``, each with its ``SKILL.md``. The key of
    a skill is the hash of its files, so any change is a new key."""

    component_schema = LocalSkillsConfig
    component_provider_override = "max_ai.capabilities.skills.local.LocalSkills"

    def __init__(
        self, path: str, skills: list[str], skill_env: dict[str, str] | None = None
    ) -> None:
        super().__init__(skills, skill_env)
        self.path = path

    def _to_config(self) -> LocalSkillsConfig:
        return LocalSkillsConfig(path=self.path, skills=self.skills, skill_env=self.skill_env)

    @classmethod
    def _from_config(cls, config: LocalSkillsConfig) -> LocalSkills:
        return cls(path=config.path, skills=config.skills, skill_env=config.skill_env)

    async def get_or_update_key(self, name: str) -> str:
        digest = hashlib.sha256()
        for file in self._files(name):
            # Name, permission and content: changing any of them changes the key.
            relative = file.relative_to(self._folder(name)).as_posix()
            executable = bool(file.stat().st_mode & 0o100)
            digest.update(f"{relative}\0{executable}\0{file.stat().st_size}\0".encode())
            digest.update(file.read_bytes())
        return digest.hexdigest()

    async def pack(self, name: str) -> bytes:
        folder = self._folder(name)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for file in self._files(name):
                tar.add(file, arcname=file.relative_to(folder).as_posix())
        return buffer.getvalue()

    def _folder(self, name: str) -> Path:
        folder = Path(self.path).expanduser().resolve() / validate_id("skill name", name)
        if not folder.is_dir():
            raise SkillError(f"skill {name} not found in {self.path}")
        return folder

    def _files(self, name: str) -> list[Path]:
        """The skill's files, always in the same order. Links are left out:
        the sandbox would not install them."""
        folder = self._folder(name)
        return sorted(p for p in folder.rglob("*") if p.is_file() and not p.is_symlink())


__all__ = ["LocalSkills", "LocalSkillsConfig"]
