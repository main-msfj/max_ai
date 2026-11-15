"""GitHubSkills, with GitHub's answers replaced by a small repository in memory."""

import httpx
import pytest

from max_ai.base.skills import SkillError
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.skills.github import GitHubSkills
from max_ai.capabilities.skills.github.skills import parse

SKILL = b"---\nname: pdf\ndescription: Work with pdf files\n---\n"


class Repository:
    """Answers the three requests GitHubSkills makes, and counts them."""

    def __init__(self):
        self.skills = {"pdf": "tree-1", "docx": "tree-2"}
        self.calls: list[str] = []
        self.headers: dict[str, str] = {}

    async def get(self, path, raw=False, **query):
        self.calls.append(path)
        if path.startswith("contents"):
            body = [{"name": n, "sha": s, "type": "dir"} for n, s in self.skills.items()]
            return httpx.Response(200, json=[*body, {"name": "README.md", "sha": "x", "type": "file"}])
        if path.startswith("git/trees"):
            return httpx.Response(200, json={"truncated": False, "tree": [
                {"path": "SKILL.md", "type": "blob", "mode": "100644", "sha": "b1"},
                {"path": "scripts", "type": "tree", "mode": "040000", "sha": "t"},
                {"path": "scripts/run.sh", "type": "blob", "mode": "100755", "sha": "b2"},
                {"path": "link", "type": "blob", "mode": "120000", "sha": "b3"},
            ]})
        return httpx.Response(200, content=SKILL if path.endswith("b1") else b"echo hi")


@pytest.fixture
def repository(monkeypatch):
    repository = Repository()
    monkeypatch.setattr(GitHubSkills, "_get", lambda self, path, raw=False, **q: repository.get(path, raw, **q))
    return repository


@pytest.fixture
async def sandbox(tmp_path):
    sandbox = LocalSandbox(root=str(tmp_path)).bind("user-1")
    await sandbox.prepare()
    return sandbox


def test_the_url_names_the_repository_its_branch_and_the_folder():
    assert parse("org/repo") == ("org/repo", None, "")
    assert parse("https://github.com/org/repo.git") == ("org/repo", None, "")
    assert parse("https://github.com/org/repo/tree/main/skills/office") == ("org/repo", "main", "skills/office")
    with pytest.raises(SkillError):
        parse("repo")


async def test_one_request_gives_the_key_of_every_skill(repository):
    skills = GitHubSkills("org/repo", ["pdf", "docx"])
    assert await skills.keys() == {"pdf": "tree-1", "docx": "tree-2"}
    assert repository.calls == ["contents"]


async def test_a_skill_that_is_not_in_the_repository_fails_with_its_name(repository):
    with pytest.raises(SkillError, match="xlsx"):
        await GitHubSkills("org/repo", ["pdf", "xlsx"]).keys()


async def test_sync_downloads_only_the_skill_that_changed(tmp_path, repository, sandbox):
    skills = GitHubSkills("org/repo", ["pdf", "docx"])
    assert await skills.sync(sandbox) == ["pdf", "docx"]
    script = tmp_path / ".agent/skills/pdf/scripts/run.sh"
    assert script.read_text() == "echo hi" and script.stat().st_mode & 0o100
    assert not (tmp_path / ".agent/skills/pdf/link").exists()

    repository.calls.clear()
    assert await skills.sync(sandbox) == []
    assert repository.calls == ["contents"]

    repository.skills["docx"] = "tree-9"
    repository.calls.clear()
    assert await skills.sync(sandbox) == ["docx"]
    assert repository.calls == ["contents", "git/trees/tree-9", "git/blobs/b1", "git/blobs/b2"]


def test_the_token_is_read_from_its_variable_and_never_serialized(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    assert GitHubSkills("org/repo", ["pdf"]).token.get_secret_value() == "ghp_secret"
    passed = GitHubSkills("org/repo", ["pdf"], token="ghp_other")
    assert passed.token.get_secret_value() == "ghp_other"
    dumped = passed.serialize()
    assert "ghp_" not in dumped.model_dump_json() and "GITHUB_TOKEN" in dumped.model_dump_json()
    assert GitHubSkills.deserialize(dumped).token.get_secret_value() == "ghp_secret"


@pytest.mark.integration
async def test_a_real_public_repository(tmp_path, sandbox):
    skills = GitHubSkills("https://github.com/anthropics/skills/tree/main/skills", ["brand-guidelines"])
    assert await skills.sync(sandbox) == ["brand-guidelines"]
    assert (tmp_path / ".agent/skills/brand-guidelines/SKILL.md").is_file()
    assert "brand-guidelines" in await skills.blocks(sandbox)
