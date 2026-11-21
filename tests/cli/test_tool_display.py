"""How a tool call is shown to a person: its name and the lines it changes."""

from max_ai.cli.blocks import ToolBlock
from max_ai.core.tool.display import changes, tool_label

EDIT = {
    "path": "config.py",
    "old_text": "limit: int = Field(\n    default=50,\n    ge=1,",
    "new_text": "limit: int = Field(\n    default=80,\n    ge=1,",
}


def test_an_edit_gives_the_lines_that_leave_and_enter_with_the_ones_that_stay():
    assert changes("edit_file", EDIT) == [
        (" ", "limit: int = Field("), ("-", "    default=50,"), ("+", "    default=80,"), (" ", "    ge=1,"),
    ]


def test_a_written_file_is_all_new_lines_and_other_tools_change_nothing():
    assert changes("write_file", {"path": "a.md", "content": "one\ntwo"}) == [("+", "one"), ("+", "two")]
    assert changes("bash", {"command": "ls"}) == []
    assert changes("edit_file", {"path": "a.md"}) == []


def test_a_tool_is_named_as_a_person_reads_it():
    assert [tool_label(name) for name in ("edit_file", "write_file", "bash", "weather")] == [
        "Edit", "Write", "Bash", "weather",
    ]


def test_the_block_of_an_edit_counts_its_lines_and_shows_them():
    block = ToolBlock("edit_file", EDIT)
    assert block._title().plain.endswith("Edit  config.py  +1 −1")
    assert block._input().plain.splitlines() == [
        "  limit: int = Field(", "−     default=50,", "+     default=80,", "      ge=1,",
    ]


def test_a_long_change_is_cut_until_the_block_is_opened():
    content = "\n".join(f"line {n}" for n in range(20))
    block = ToolBlock("write_file", {"path": "a.md", "content": content})
    assert block._input().plain.splitlines()[-1] == "… (+8 lines)"
    assert block._title().plain.endswith("Write  a.md  +20")


def test_the_lines_are_numbered_once_the_tool_says_where_the_change_begins():
    from max_ai.types.tool_call import ToolResult

    block = ToolBlock("edit_file", EDIT)
    block.result = ToolResult.success_result("c1", "Edited", metadata={"line": 38})
    assert block._input().plain.splitlines() == [
        "  38   limit: int = Field(", "  39 −     default=50,", "  39 +     default=80,", "  40       ge=1,",
    ]
    # A written file starts at its first line, known before it runs.
    written = ToolBlock("write_file", {"path": "a.md", "content": "one\ntwo"})
    assert written._input().plain.splitlines() == ["   1 + one", "   2 + two"]


def test_the_card_of_a_call_that_asks_shows_what_goes_in_and_why():
    command = "if [ -e .env ]; then printf 'a; b\\n'; cat .env; fi && ls"
    card = ToolBlock.approval("bash", {"command": command, "description": "Read the keys"}, "it names a file of keys")
    assert card.plain.splitlines() == [
        "◆ Bash  Read the keys", "",
        "  IN   if [ -e .env ];", "       then printf 'a; b\\n';", "       cat .env;", "       fi && ls", "",
        "  WHY  it names a file of keys",
    ]
    # A file tool shows its lines; without a reason there is no WHY.
    edit = ToolBlock.approval("edit_file", EDIT, None)
    assert "−     default=50," in edit.plain and "WHY" not in edit.plain


def test_a_command_is_cut_in_lines_only_outside_quotes_and_a_script_is_left_as_it_is():
    from max_ai.core.tool.display import command_lines

    assert command_lines("cd app && echo 'a; b' || exit 1; ls") == ["cd app && echo 'a; b' || exit 1;", "ls"]
    script = "python3 - <<'PY'\nfor n in range(2):\n    print(n); print('x')\nPY"
    assert command_lines(script) == script.splitlines()


def test_the_workspace_is_listed_where_it_is_and_drawn_as_a_tree(tmp_path):
    import subprocess

    from max_ai.cli.widgets import WorkspaceTree

    (tmp_path / "output" / "sheets").mkdir(parents=True)
    (tmp_path / "output" / "sheets" / "flights.xlsx").touch()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").touch()
    (tmp_path / "index.html").touch()
    printed = subprocess.run(
        WorkspaceTree.command(str(tmp_path)), shell=True, capture_output=True, text=True, check=True
    ).stdout
    assert WorkspaceTree.read(printed) == [
        ("output", True), ("index.html", False), ("output/sheets", True),
        ("output/sheets/flights.xlsx", False),
    ]


def test_the_theme_is_the_one_tried_then_the_saved_one_then_gray(tmp_path, monkeypatch):
    from max_ai.cli.theme import Theme
    from max_ai.config import setting

    monkeypatch.setattr(setting, "max_ai_home", tmp_path)
    monkeypatch.delenv(Theme.ENV, raising=False)
    assert Theme.chosen() == "gray"
    Theme.save("blue")
    assert Theme.chosen() == "blue" and (tmp_path / "cli.json").read_text() == '{\n  "theme": "blue"\n}\n'
    monkeypatch.setenv(Theme.ENV, "teal")
    assert Theme.chosen() == "teal"
    # A name that is no theme, or a file that cannot be read, is gray.
    monkeypatch.setenv(Theme.ENV, "pink")
    assert Theme.chosen() == "gray"
    monkeypatch.delenv(Theme.ENV)
    (tmp_path / "cli.json").write_text("not json")
    assert Theme.chosen() == "gray"
    import pytest

    with pytest.raises(ValueError, match="pink is not a theme"):
        Theme.save("pink")


def test_a_compaction_whose_summary_failed_says_what_was_lost():
    from max_ai.cli.blocks import CompactionBlock
    from max_ai.core.event_type import CompactionEvent

    def header(**given) -> str:
        block = CompactionBlock("SummaryCompaction")
        block.done = True
        block.event = CompactionEvent(
            source="loop", strategy="SummaryCompaction", changed=True, kept_message_count=6,
            tokens_before=90_000, tokens_after=12_000, **given,
        )
        return block._header().plain

    assert "failed" not in header(summary="Goal: build fizz", removed=14)
    assert "the summary failed: 14 messages left without one" in header(removed=14, summary_failed=True)
