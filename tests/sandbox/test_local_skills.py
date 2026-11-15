"""SkillsCore (variables, sync, blocks) and LocalSkills, on a LocalSandbox."""

import json

import pytest

from max_ai.base.skills import SkillError
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.skills.local import LocalSkills


@pytest.fixture
def source(tmp_path):
    """A folder with two skills, outside the sandbox."""
    for name in ("pdf", "docx"):
        folder = tmp_path / "source" / name
        (folder / "scripts").mkdir(parents=True)
        (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Work with {name} files\n---\n")
        (folder / "scripts/run.sh").write_text("echo hi")
    return tmp_path / "source"


@pytest.fixture
async def sandbox(tmp_path):
    sandbox = LocalSandbox(root=str(tmp_path / "env")).bind("user-1")
    (tmp_path / "env").mkdir()
    await sandbox.prepare()
    return sandbox


# -------- key --------------------------------------------------------------------
async def test_the_key_changes_with_content_name_or_permission(source):
    skills = LocalSkills(str(source), ["pdf"])
    first = await skills.get_or_update_key("pdf")
    assert await skills.get_or_update_key("pdf") == first

    (source / "pdf/scripts/run.sh").write_text("echo bye")
    second = await skills.get_or_update_key("pdf")
    (source / "pdf/scripts/run.sh").rename(source / "pdf/scripts/go.sh")
    third = await skills.get_or_update_key("pdf")
    (source / "pdf/scripts/go.sh").chmod(0o755)
    fourth = await skills.get_or_update_key("pdf")
    assert len({first, second, third, fourth}) == 4


async def test_a_skill_that_is_not_there_fails_with_its_name(source):
    with pytest.raises(SkillError, match="xlsx"):
        await LocalSkills(str(source), ["xlsx"]).get_or_update_key("xlsx")


# -------- sync -------------------------------------------------------------------
async def test_sync_installs_only_what_is_missing_or_changed(tmp_path, source, sandbox):
    skills = LocalSkills(str(source), ["pdf", "docx"])
    assert await skills.sync(sandbox) == ["pdf", "docx"]
    installed = tmp_path / "env/.agent/skills"
    assert (installed / "pdf/scripts/run.sh").read_text() == "echo hi"
    assert await skills.sync(sandbox) == []

    (source / "docx/scripts/run.sh").write_text("echo new")
    assert await skills.sync(sandbox) == ["docx"]
    assert (installed / "docx/scripts/run.sh").read_text() == "echo new"


async def test_sync_installs_only_the_skills_asked_for(tmp_path, source, sandbox):
    await LocalSkills(str(source), ["pdf"]).sync(sandbox)
    assert [p.name for p in (tmp_path / "env/.agent/skills").iterdir()] == ["pdf"]


async def test_the_settings_keep_key_and_description(tmp_path, source, sandbox):
    skills = LocalSkills(str(source), ["pdf"])
    await skills.sync(sandbox)
    settings = json.loads((tmp_path / "env/.agent/setting.json").read_text())
    assert settings["skills"]["pdf"] == {
        "key": await skills.get_or_update_key("pdf"),
        "description": "Work with pdf files",
    }


async def test_a_skill_without_description_is_not_installed(tmp_path, source, sandbox):
    (source / "pdf/SKILL.md").write_text("no front matter")
    with pytest.raises(SkillError, match="description"):
        await LocalSkills(str(source), ["pdf"]).sync(sandbox)
    assert list((tmp_path / "env/.agent/skills").iterdir()) == []


# -------- prompt -----------------------------------------------------------------
async def test_blocks_name_each_skill_with_its_description_and_where_it_is(tmp_path, source, sandbox):
    skills = LocalSkills(str(source), ["pdf", "docx"])
    await skills.sync(sandbox)
    where = tmp_path / "env/.agent/skills"
    assert await skills.blocks(sandbox) == (
        f"- pdf: Work with pdf files ({where}/pdf/SKILL.md)\n"
        f"- docx: Work with docx files ({where}/docx/SKILL.md)"
    )


# -------- variables --------------------------------------------------------------
async def test_variables_reach_the_sandbox_under_the_skill_name(source, sandbox, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "secret-1")
    skills = LocalSkills(str(source), ["pdf"], skill_env={"ANTHROPIC_API": "ANT_KEY"})
    assert skills.variables() == {"ANTHROPIC_API": "secret-1"}

    bound = sandbox.bind("user-1", skills.variables())
    result = await bound.execute_command("echo $ANTHROPIC_API")
    assert result.stdout.strip() == "secret-1"


def test_a_missing_variable_fails_naming_it(source, monkeypatch):
    monkeypatch.delenv("ANT_KEY", raising=False)
    with pytest.raises(SkillError, match="ANT_KEY"):
        LocalSkills(str(source), ["pdf"], skill_env={"ANTHROPIC_API": "ANT_KEY"}).variables()


def test_serialized_skills_carry_names_never_values(source, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "secret-1")
    skills = LocalSkills(str(source), ["pdf"], skill_env={"ANTHROPIC_API": "ANT_KEY"})
    dumped = skills.serialize().model_dump_json()
    assert "ANT_KEY" in dumped and "secret-1" not in dumped
