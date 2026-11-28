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
        "IN     limit: int = Field(", "     −     default=50,", "     +     default=80,", "           ge=1,",
    ]


def test_a_long_change_is_cut_until_the_block_is_opened():
    content = "\n".join(f"line {n}" for n in range(20))
    block = ToolBlock("write_file", {"path": "a.md", "content": content})
    folded = block._input().plain.splitlines()
    assert len(folded) == ToolBlock.DIFF_LINES + 1 and folded[-1] == "     … +16 lines · click to open"
    assert block._title().plain.endswith("Write  a.md  +20")
    block.add_class("-expanded")
    assert len(block._input().plain.splitlines()) == 20


def test_a_folded_change_starts_just_before_the_first_line_that_changes():
    same = "\n".join(f"same {n}" for n in range(10))
    edit = {"path": "a.py", "old_text": f"{same}\nold", "new_text": f"{same}\nnew"}
    assert ToolBlock("edit_file", edit)._input().plain.splitlines() == [
        "IN     same 9", "     − old", "     + new", "     … +9 lines · click to open",
    ]


def test_the_lines_are_numbered_once_the_tool_says_where_the_change_begins():
    from max_ai.types.tool_call import ToolResult

    block = ToolBlock("edit_file", EDIT)
    block.result = ToolResult.success_result("c1", "Edited", metadata={"line": 38})
    assert block._input().plain.splitlines() == [
        "IN     38   limit: int = Field(", "       39 −     default=50,",
        "       39 +     default=80,", "       40       ge=1,",
    ]
    # A written file starts at its first line, known before it runs.
    written = ToolBlock("write_file", {"path": "a.md", "content": "one\ntwo"})
    assert written._input().plain.splitlines() == ["IN      1 + one", "        2 + two"]


def test_a_command_and_what_it_printed_show_a_few_lines_until_opened():
    from max_ai.types.tool_call import ToolResult

    script = "python3 - <<'EOF'\n" + "\n".join(f"step_{n}()" for n in range(8)) + "\nEOF"
    block = ToolBlock("bash", {"command": script, "description": "Run the steps"})
    assert block._out().plain == "OUT  running… · ctrl+g to send to background"
    block.result = ToolResult.success_result("c1", "exit code: 3\n" + "\n".join(f"out {n}" for n in range(9)))
    block._output = "\n".join(f"out {n}" for n in range(9))
    assert block._input().plain.splitlines() == [
        "IN   python3 - <<'EOF'", "     step_0()", "     step_1()", "     … +7 lines · click to open",
    ]
    assert block._out().plain.splitlines() == [
        "OUT  out 0", "     out 1", "     out 2", "     … +6 lines · click to open",
    ]
    # How it went is at the end of the title, not in a line of its own.
    assert block._title().plain.startswith("◆ Bash  Run the steps  exit 3")
    block.add_class("-expanded")
    assert len(block._input().plain.splitlines()) == 10 and len(block._out().plain.splitlines()) == 9


def test_a_failure_is_one_line_and_a_read_file_only_counts_its_lines():
    from max_ai.types.tool_call import ToolResult

    failed = ToolBlock("edit_file", EDIT)
    failed.complete(ToolResult.execution_error("c1", "a.py: it resolves outside\nmore detail"))
    assert failed._out().plain.splitlines() == [
        "OUT  Error: a.py: it resolves outside", "     … +1 line · click to open",
    ]
    read = ToolBlock("read_file", {"path": "a.py"})
    read.complete(ToolResult.success_result("c2", "one\ntwo\nthree"))
    assert read._out().plain == "OUT  3 lines · click to open"


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


def test_a_background_command_and_the_task_tools_say_what_they_do():
    from max_ai.core.tool.display import tool_summary

    command = {"command": "pytest -q", "description": "Run the tests"}
    assert tool_summary("bash", {**command, "background": True}) == "in background · Run the tests"
    assert tool_summary("bash", {**command, "background": False}) == "Run the tests"
    assert tool_summary("task_output", {"task_id": "2"}) == "task 2"
    assert tool_summary("task_output", {}) == "all tasks"
    assert [tool_label(name) for name in ("task_output", "stop_task")] == ["Task output", "Stop task"]
