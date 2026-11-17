"""GitHubSkills: skills that are folders of a GitHub repository."""

from __future__ import annotations

import io
import os
import tarfile
import typing as t

import httpx
from pydantic import BaseModel, Field, SecretStr

from ....base.session_store import validate_id
from ....base.skills import SkillsCore
from ....errors.skills import SkillError

API = "https://api.github.com"
TOKEN_ENV = "GITHUB_TOKEN"  # where the token is looked for when it is not passed
EXECUTABLE = "100755"  # git's mode of a file that can be run


class GitHubSkillsConfig(BaseModel):
    url: str
    skills: list[str]
    token_env: str = TOKEN_ENV
    skill_env: dict[str, str] = Field(default_factory=dict)


class GitHubSkills(SkillsCore):
    """The skills ``<name>/`` of a repository, each downloaded on its own.

    ``url`` is the repository, or the folder in it that holds the skills:
    ``org/repo``, ``https://github.com/org/repo`` or
    ``https://github.com/org/repo/tree/<branch>/<folder>``.

    The key of a skill is git's id of its folder: it changes only when
    something inside that folder does.

    ``token`` is needed for a private repository; without it, ``$GITHUB_TOKEN``
    is tried. It is never serialized: only the name of that variable.
    """

    component_schema = GitHubSkillsConfig
    component_provider_override = "max_ai.capabilities.skills.github.GitHubSkills"

    def __init__(
        self,
        url: str,
        skills: list[str],
        token: str | SecretStr | None = None,
        skill_env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(skills, skill_env)
        self.url = url
        token = token or os.getenv(TOKEN_ENV)
        self.token = SecretStr(token) if isinstance(token, str) else token
        self.repository, self.ref, self.folder = parse(url)
        self._keys: dict[str, str] = {}

    def _to_config(self) -> GitHubSkillsConfig:
        return GitHubSkillsConfig(
            url=self.url, skills=self.skills, token_env=TOKEN_ENV, skill_env=self.skill_env
        )

    @classmethod
    def _from_config(cls, config: GitHubSkillsConfig) -> GitHubSkills:
        return cls(
            url=config.url,
            skills=config.skills,
            token=os.getenv(config.token_env),
            skill_env=config.skill_env,
        )

    async def keys(self) -> dict[str, str]:
        """One request lists the folder that holds the skills, with the id
        of each."""
        query = {"ref": self.ref} if self.ref else {}
        listing = (await self._get(f"contents/{self.folder}".rstrip("/"), **query)).json()
        found = {entry["name"]: entry["sha"] for entry in listing if entry["type"] == "dir"}
        if missing := [name for name in self.skills if name not in found]:
            raise SkillError(f"not found in {self.url}: {', '.join(missing)}")
        self._keys = {name: found[name] for name in self.skills}
        return self._keys

    async def get_or_update_key(self, name: str) -> str:
        return (await self.keys())[name]

    async def pack(self, name: str) -> bytes:
        validate_id("skill name", name)
        # Download the folder with the id that was compared: key and files match.
        key = self._keys.get(name) or await self.get_or_update_key(name)
        tree = (await self._get(f"git/trees/{key}", recursive="1")).json()
        if tree.get("truncated"):
            raise SkillError(f"skill {name} has too many files to list")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for entry in tree["tree"]:
                if entry["type"] != "blob" or entry["mode"] not in ("100644", EXECUTABLE):
                    continue  # folders, links and submodules
                content = (await self._get(f"git/blobs/{entry['sha']}", raw=True)).content
                info = tarfile.TarInfo(entry["path"])
                info.size = len(content)
                info.mode = 0o755 if entry["mode"] == EXECUTABLE else 0o644
                tar.addfile(info, io.BytesIO(content))
        return buffer.getvalue()

    async def _get(self, path: str, raw: bool = False, **query: str) -> httpx.Response:
        """One GET to the repository's API."""
        headers = {
            "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token is not None:
            headers["Authorization"] = f"Bearer {self.token.get_secret_value()}"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{API}/repos/{self.repository}/{path}", params=query, headers=headers
            )
        if response.status_code != 200:
            reason = _reason(response)
            raise SkillError(f"GitHub answered {response.status_code} for {self.url}: {reason}")
        return response


def parse(url: str) -> tuple[str, str | None, str]:
    """``url`` as (``org/repo``, branch or None, folder of the skills)."""
    parts = url.removeprefix("https://").removeprefix("github.com/").strip("/").split("/")
    parts[-1] = parts[-1].removesuffix(".git")
    if len(parts) < 2 or not all(parts[:2]):
        raise SkillError(f"not a GitHub repository: {url}")
    repository = "/".join(parts[:2])
    if len(parts) >= 4 and parts[2] == "tree":
        return repository, parts[3], "/".join(parts[4:])
    return repository, None, "/".join(parts[2:])


def _reason(response: httpx.Response) -> str:
    try:
        body: t.Any = response.json()
    except ValueError:
        return response.text[:200]
    return str(body.get("message", "")) if isinstance(body, dict) else ""


__all__ = ["GitHubSkills", "GitHubSkillsConfig"]
