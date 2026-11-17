"""Skills: where they come from. The sandbox is where they are installed."""

from __future__ import annotations

import io
import os
import tarfile
from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..errors.skills import SkillError
from .component import ComponentBase
from .sandbox import SandboxCore


class SkillsCore(ComponentBase[BaseModel], ABC):
    """A place skills come from. A child says how to get a skill's key and
    its files; comparing and installing is written here, once."""

    component_type = "skills"

    def __init__(self, skills: list[str], skill_env: dict[str, str] | None = None) -> None:
        self.skills = list(skills)
        # {"NAME_INSIDE_THE_SKILL": "NAME_IN_MY_ENVIRONMENT"}
        self.skill_env = dict(skill_env or {})

    @abstractmethod
    async def get_or_update_key(self, name: str) -> str:
        """What changes whenever the skill changes: a content hash, git's id."""

    @abstractmethod
    async def pack(self, name: str) -> bytes:
        """The skill's folder as a tar.gz, ``SKILL.md`` at its top."""

    async def keys(self) -> dict[str, str]:
        """The key of every skill. A child that can ask for all of them at
        once writes it here."""
        return {name: await self.get_or_update_key(name) for name in self.skills}

    def variables(self) -> dict[str, str]:
        """The values the skills need, under the names they use. They go to
        ``sandbox.bind(identifier, variables)``."""
        found: dict[str, str] = {}
        missing: list[str] = []
        for inside, mine in self.skill_env.items():
            value = os.getenv(mine)
            if value is None:
                missing.append(mine)
            else:
                found[inside] = value
        if missing:
            raise SkillError(f"not set in the environment: {', '.join(missing)}")
        return found

    async def sync(self, sandbox: SandboxCore) -> list[str]:
        """Install in ``sandbox`` the skills that are missing or changed, and
        return their names."""
        keys = await self.keys()
        outdated = await sandbox.outdated_skills(keys)
        for name in outdated:
            archive = await self.pack(name)
            await sandbox.install_skill(name, keys[name], archive, describe(name, archive))
        return outdated

    async def blocks(self, sandbox: SandboxCore) -> str:
        """What the prompt says about these skills, as installed in ``sandbox``."""
        installed = await sandbox.installed_skills()
        return "\n".join(
            f"- {name}: {installed[name]['description']} ({sandbox.layout.skills / name}/SKILL.md)"
            for name in self.skills
            if name in installed
        )


def describe(name: str, archive: bytes) -> str:
    """The ``description:`` line at the top of the skill's ``SKILL.md``."""
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        try:
            file = tar.extractfile("SKILL.md")
        except KeyError:
            file = None
        if file is None:
            raise SkillError(f"skill {name} has no SKILL.md")
        text = file.read().decode(errors="replace")
    for line in text.splitlines():
        if line.startswith("description:"):
            return line.removeprefix("description:").strip().strip("\"'")
    raise SkillError(f"skill {name}: SKILL.md has no description")


__all__ = ["SkillError", "SkillsCore", "describe"]
