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
        "  IN   if [ -e .env ];", "       then printf 'a; b\\n';", "       cat .env;", "       fi &&", "       ls", "",
        "  WHY  it names a file of keys",
    ]
    # A file tool shows its lines; without a reason there is no WHY.
    edit = ToolBlock.approval("edit_file", EDIT, None)
    assert "−     default=50," in edit.plain and "WHY" not in edit.plain


def test_a_command_is_cut_in_lines_only_outside_quotes():
    from max_ai.core.tool.display import command_lines

    assert command_lines("cd app && echo 'a && b; c' || exit 1") == [
        "cd app &&", "echo 'a && b; c' ||", "exit 1",
    ]
    assert command_lines('echo "x; y"; ls') == ['echo "x; y";', "ls"]


def test_the_file_tree_hides_what_the_tools_keep_for_themselves(tmp_path):
    from max_ai.cli.widgets import FileTree

    for name in (".git", ".venv", "__pycache__", "src", ".env", "README.md"):
        (tmp_path / name).mkdir() if "." not in name[1:] else (tmp_path / name).touch()
    shown = FileTree.filter_paths(FileTree, tmp_path.iterdir())  # no app needed: it only reads HIDDEN
    assert sorted(path.name for path in shown) == [".env", "README.md", "src"]
