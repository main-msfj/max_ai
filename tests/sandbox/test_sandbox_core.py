"""What every sandbox inherits from SandboxCore, run on LocalSandbox:
bind, prepare, upload_file, outdated_skills and install_skill."""

import hashlib
import io
import json
import shlex
import tarfile

import pytest

from max_ai.base.layout import SCRIPT
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.errors.sandbox import SandboxError


@pytest.fixture
async def sandbox(tmp_path):
    sandbox = LocalSandbox(root=str(tmp_path)).bind("user-1")
    await sandbox.prepare()
    return sandbox


def settings(tmp_path):
    return json.loads((tmp_path / ".agent" / "setting.json").read_text())


# -------- bind -------------------------------------------------------------------
def test_bind_gives_a_copy_and_leaves_the_shared_sandbox_alone(tmp_path):
    shared = LocalSandbox(root=str(tmp_path))
    bound = shared.bind("user-1", {"TOKEN": "x"})
    assert (bound.identifier, bound.variables) == ("user-1", {"TOKEN": "x"})
    assert (shared.identifier, shared.variables) == (None, {})


def test_variables_are_never_serialized(tmp_path):
    bound = LocalSandbox(root=str(tmp_path)).bind("user-1", {"TOKEN": "secret"})
    assert "secret" not in bound.serialize().model_dump_json()


# -------- prepare ----------------------------------------------------------------
async def test_prepare_builds_the_agent_folder(tmp_path, sandbox):
    root = tmp_path / ".agent"
    assert all((root / name).is_dir() for name in ("skills", "workspace", "tmp"))
    assert settings(tmp_path) == {"skills": {}}


async def test_prepare_never_overwrites_the_settings(tmp_path, sandbox):
    (tmp_path / ".agent" / "setting.json").write_text('{"skills": {"pdf": "k1"}}')
    await sandbox.prepare()
    assert settings(tmp_path) == {"skills": {"pdf": "k1"}}


# -------- upload_file ------------------------------------------------------------
async def test_upload_puts_the_file_in_the_session_folder(tmp_path, sandbox):
    where = await sandbox.upload_file("s1", "docs/photo.png", b"\x89PNG\x00\xff")
    assert where == str(tmp_path / ".agent/workspace/s1/docs/photo.png")
    assert (tmp_path / ".agent/workspace/s1/docs/photo.png").read_bytes() == b"\x89PNG\x00\xff"


@pytest.mark.parametrize(
    ("session_id", "path"),
    [("s1", "../../escape.txt"), ("..", "x.txt"), ("s1", "/etc/x.txt")],
)
async def test_upload_never_leaves_the_session_folder(tmp_path, sandbox, session_id, path):
    with pytest.raises((SandboxError, ValueError)):
        await sandbox.upload_file(session_id, path, b"x")
    assert not (tmp_path / ".agent" / "escape.txt").exists()


async def test_content_that_arrives_incomplete_changes_nothing(tmp_path, sandbox):
    target = tmp_path / ".agent/workspace/s1"
    target.mkdir(parents=True)
    (target / "a.bin").write_bytes(b"OLD")
    # The request carries the sha256 of the whole content; only half arrives.
    request = {
        "op": "upload", "folder": str(target), "path": "a.bin",
        "sha256": hashlib.sha256(b"half and the rest").hexdigest(),
    }
    result = await sandbox.execute_command(
        f"python3 -c {shlex.quote(SCRIPT)} {shlex.quote(json.dumps(request))}", stdin=b"half"
    )
    assert "incomplete" in json.loads(result.stdout)["error"]
    assert (target / "a.bin").read_bytes() == b"OLD"
    assert [p.name for p in target.iterdir()] == ["a.bin"]


# -------- skills -----------------------------------------------------------------
async def test_outdated_skills_are_the_missing_ones_and_those_with_another_key(sandbox, skill_archive):
    wanted = {"pdf": "k1", "docx": "k2"}
    assert await sandbox.outdated_skills(wanted) == ["pdf", "docx"]
    await sandbox.install_skill("pdf", "k1", skill_archive({"SKILL.md": b"pdf"}))
    assert await sandbox.outdated_skills(wanted) == ["docx"]
    assert await sandbox.outdated_skills({"pdf": "k9"}) == ["pdf"]


async def test_install_unpacks_the_skill_and_records_its_key(tmp_path, sandbox, skill_archive):
    await sandbox.install_skill("pdf", "k1", skill_archive({"SKILL.md": b"pdf", "scripts/run.sh": b"echo"}))
    skill = tmp_path / ".agent/skills/pdf"
    assert (skill / "SKILL.md").read_bytes() == b"pdf"
    assert (skill / "scripts/run.sh").read_bytes() == b"echo"
    kept = settings(tmp_path)["skills"]["pdf"]
    assert len(kept.pop("hash")) == 64 and kept == {"key": "k1", "description": ""}


async def test_a_new_version_replaces_the_whole_folder(tmp_path, sandbox, skill_archive):
    await sandbox.install_skill("pdf", "k1", skill_archive({"SKILL.md": b"v1", "old.txt": b"x"}))
    await sandbox.install_skill("pdf", "k2", skill_archive({"SKILL.md": b"v2"}))
    assert [p.name for p in (tmp_path / ".agent/skills/pdf").iterdir()] == ["SKILL.md"]
    assert settings(tmp_path)["skills"]["pdf"]["key"] == "k2"


async def test_a_skill_that_writes_outside_is_refused_and_leaves_nothing(tmp_path, sandbox, skill_archive):
    def escape(tar):
        info = tarfile.TarInfo("../../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(SandboxError, match="leaves"):
        await sandbox.install_skill("bad", "k1", skill_archive({"SKILL.md": b"x"}, also=escape))
    assert not (tmp_path / ".agent" / "escape.txt").exists()
    assert list((tmp_path / ".agent/skills").iterdir()) == []
    assert settings(tmp_path) == {"skills": {}}


async def test_links_inside_a_skill_are_skipped(tmp_path, sandbox, skill_archive):
    def link(tar):
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    await sandbox.install_skill("pdf", "k1", skill_archive({"SKILL.md": b"x"}, also=link))
    assert [p.name for p in (tmp_path / ".agent/skills/pdf").iterdir()] == ["SKILL.md"]


async def test_a_cut_archive_keeps_the_installed_version(tmp_path, sandbox, skill_archive):
    await sandbox.install_skill("pdf", "k1", skill_archive({"SKILL.md": b"v1"}))
    with pytest.raises(SandboxError):
        await sandbox.install_skill("pdf", "k2", skill_archive({"SKILL.md": b"v2"})[:20])
    assert (tmp_path / ".agent/skills/pdf/SKILL.md").read_bytes() == b"v1"
    # Never a key without its complete folder: k1, or none (reinstalled next run).
    assert settings(tmp_path)["skills"].get("pdf", {}).get("key") in ("k1", None)


async def test_a_skill_name_cannot_be_a_path(sandbox):
    with pytest.raises(ValueError):
        await sandbox.install_skill("../x", "k1", b"")
