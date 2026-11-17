"""read_file reads code, notebooks and PDFs from the session's folders, in parts."""

import importlib.util
import json
import os

import pytest

from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.filesystem import ReadFileTool
from max_ai.core.tool import ToolDispatcher, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord


@pytest.fixture
async def sandbox(tmp_path):
    """A user's sandbox; session ``s1`` has a few files, session ``s2`` a secret."""
    bound = LocalSandbox(root=str(tmp_path)).bind("user-1")
    await bound.prepare()
    await bound.upload_file("s1", "app.py", b"import os\n\nprint(os.getcwd())\n")
    await bound.upload_file("s1", "src/index.js", b"console.log('hi')\n")
    await bound.upload_file("s1", "long.txt", "".join(f"line {n}\n" for n in range(1, 31)).encode())
    await bound.upload_file("s1", "photo.bin", b"\x89PNG\x00\x00binary")
    await bound.upload_file("s1", "big.txt", b"result", temporary=True)
    await bound.upload_file("s2", "secret.txt", b"another session")
    (tmp_path / "host.txt").write_text("outside the sandbox folders")
    return bound


async def read(sandbox, **parameters):
    record = ToolCallRecord(tool_name="read_file", parameters=parameters)
    ctx = RunContext(user_id="user-1", session_id="s1")
    return await ToolDispatcher(ToolRegistry([ReadFileTool()]), sandbox).dispatch(record, ctx)


async def test_code_comes_back_with_its_line_numbers(sandbox):
    result = await read(sandbox, path="app.py")
    assert result.success
    assert result.result == "     1\timport os\n     2\t\n     3\tprint(os.getcwd())"
    assert (await read(sandbox, path="src/index.js")).result.endswith("console.log('hi')")


async def test_a_long_file_is_read_in_parts(sandbox):
    result = await read(sandbox, path="long.txt", offset=11, limit=5)
    lines = result.result.splitlines()
    assert lines[0] == "    11\tline 11" and lines[4] == "    15\tline 15"
    assert lines[5] == "[lines 11-15 of 30; continue with offset=16]"
    assert (await read(sandbox, path="long.txt", offset=99)).result == "long.txt has 30 lines."


async def test_an_absolute_path_inside_the_session_folders_works(sandbox):
    tmp = sandbox.session_paths("s1")["tmp"]
    assert (await read(sandbox, path=f"{tmp}/big.txt")).result.endswith("result")


async def test_a_notebook_is_read_cell_by_cell(sandbox):
    cells = [
        {"cell_type": "markdown", "source": ["# Sales\n", "By quarter."]},
        {"cell_type": "code", "source": ["print(1 + 1)"],
         "outputs": [{"output_type": "stream", "text": ["2\n"]}]},
    ]
    await sandbox.upload_file("s1", "report.ipynb", json.dumps({"cells": cells}).encode())
    result = await read(sandbox, path="report.ipynb")
    assert [line.split("\t", 1)[1] for line in result.result.splitlines()] == [
        "--- cell 1 (markdown) ---", "# Sales", "By quarter.",
        "--- cell 2 (code) ---", "print(1 + 1)", "[output]", "2",
    ]


async def test_a_pdf_says_what_it_needs_when_pypdf_is_missing(sandbox):
    if importlib.util.find_spec("pypdf") is not None:
        pytest.skip("pypdf is installed here")
    await sandbox.upload_file("s1", "paper.pdf", b"%PDF-1.4\n")
    result = await read(sandbox, path="paper.pdf")
    assert not result.success and "needs the pypdf package" in result.error


@pytest.mark.parametrize(
    "path, reason",
    [
        ("missing.py", "does not exist"),
        ("src", "is a folder"),
        ("photo.bin", "not a text file"),
        ("../s2/secret.txt", "outside the folders"),
        ("../../../host.txt", "outside the folders"),
        ("/etc/passwd", "outside the folders"),
    ],
)
async def test_what_cannot_be_read_says_why(sandbox, path, reason):
    result = await read(sandbox, path=path)
    assert not result.success and reason in result.error


async def test_a_link_pointing_outside_is_refused(sandbox, tmp_path):
    os.symlink(tmp_path / "host.txt", f"{sandbox.session_paths('s1')['workspace']}/shortcut.txt")
    result = await read(sandbox, path="shortcut.txt")
    assert not result.success and "outside the folders" in result.error
