"""GithubSkillRegistry: skills cloned from a repository over git."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from max_ai.capabilities.skills.github import GithubSkillRegistry
from max_ai.types.workspace import WorkspaceDirectory


def make_repo(tmp_path: Path) -> Path:
    """A tiny local git repo standing in for a GitHub remote."""
    repo = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "a@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "a"], check=True)

    skill_dir = repo / "greet"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: greet\ndescription: Say hello.\n---\n\nSay hello.\n"
    )
    other = repo / "farewell"
    other.mkdir()
    (other / "SKILL.md").write_text("---\nname: farewell\ndescription: Say bye.\n---\n")

    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "skills"], check=True, capture_output=True)
    return repo


def directory(tmp_path: Path, name: str) -> WorkspaceDirectory:
    root = tmp_path / "server" / name
    return WorkspaceDirectory(
        root=root, workspace_dir=root / "workspace",
        skill_dir=root / "skills", artifacts_dir=root / "workspace",
    )


async def test_clones_and_materializes_a_skill(tmp_path):
    repo = make_repo(tmp_path)
    registry = GithubSkillRegistry(source=f"file://{repo}", skills=["greet"])

    async with registry:
        blocks = await registry.get_skills()
        assert [b.name for b in blocks] == ["greet"]

        target = registry.materialize(directory(tmp_path, "u1"))
        assert (target / "greet" / "SKILL.md").is_file()
        assert "farewell" not in [p.name for p in target.iterdir()]  # only selected skills


async def test_unknown_skill_lists_what_is_actually_in_the_repo(tmp_path):
    repo = make_repo(tmp_path)
    registry = GithubSkillRegistry(source=f"file://{repo}", skills=["nope"])
    async with registry:
        with pytest.raises(FileNotFoundError, match="greet"):
            await registry.get_skills()


async def test_private_repo_needs_the_token_env_set(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.delenv("MY_GH_TOKEN", raising=False)
    registry = GithubSkillRegistry(source=f"file://{repo}", skills=["greet"], token_env="MY_GH_TOKEN")
    async with registry:
        with pytest.raises(ValueError, match="MY_GH_TOKEN"):
            await registry.get_skills()


def test_config_stores_the_env_var_name_never_a_token():
    registry = GithubSkillRegistry(source="org/repo", skills=["x"], token_env="MY_GH_TOKEN")
    stored = registry.serialize().model_dump_json()
    assert "MY_GH_TOKEN" in stored
    back = GithubSkillRegistry.deserialize(stored)
    assert (back.source, back.skills, back.token_env) == ("org/repo", ["x"], "MY_GH_TOKEN")


@pytest.mark.parametrize("source, expected", [
    ("org/repo", "https://github.com/org/repo.git"),
    ("https://github.com/org/repo.git", "https://github.com/org/repo.git"),
    ("git@github.com:org/repo.git", "git@github.com:org/repo.git"),
])
def test_short_form_expands_against_github(source, expected):
    assert GithubSkillRegistry._repo_url(source) == expected


def test_rejects_a_source_that_is_neither_a_url_nor_org_repo():
    with pytest.raises(ValueError, match="org/repo"):
        GithubSkillRegistry._repo_url("not-a-repo-reference")
