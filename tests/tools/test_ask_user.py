import pytest

from max_ai.capabilities.tools.ask_user import AskUserTool
from max_ai.core.harness import messages as harness
from max_ai.types.tool_call import ToolCallRecord


def options(*labels: str) -> list[dict]:
    return [{"label": label} for label in labels]


def is_valid(questions: list[dict]) -> bool:
    tool = AskUserTool()
    record = ToolCallRecord(id="r", tool_name=tool.name, parameters={"questions": questions})
    return tool.validate_parameters(record).is_tool_valid


@pytest.mark.parametrize("questions", [
    [{"question": "Format?"}],
    [{"question": "Format?", "options": options("CSV", "JSON")}],
])
def test_good_questions_pass(questions):
    assert is_valid(questions)


@pytest.mark.parametrize("questions", [
    [],
    [{"question": str(n)} for n in range(5)],
    [{"question": "   "}],
    [{"question": "Format?"}, {"question": " format? "}],
    [{"question": "Format?", "options": options("CSV")}],
    [{"question": "Format?", "options": options("CSV", " ")}],
    [{"question": "A?", "options": options("x", "y")}, {"question": "B?", "options": options("z", "Z")}],
])
def test_bad_questions_cost_a_retry(questions):
    assert not is_valid(questions)


def test_an_answer_covers_every_question_and_nothing_else():
    questions = [{"question": "A?"}, {"question": "B?"}]
    record = ToolCallRecord(id="r", tool_name=AskUserTool.TOOL_NAME, parameters={"questions": questions})
    answers = AskUserTool.answer(record, {"A?": "yes", "Z?": "ignored"}).result["answers"]
    assert answers == {"A?": "yes", "B?": harness.QUESTION_NOT_ANSWERED}


def test_multi_select_must_be_a_boolean():
    question = {"question": "Files?", "options": options("a", "b")}
    assert is_valid([{**question, "multi_select": True}])
    assert not is_valid([{**question, "multi_select": "yes"}])


def test_several_answers_reach_the_model_as_a_list():
    questions = [{"question": "Files?", "multi_select": True, "options": options("a, b", "c")}]
    record = ToolCallRecord(id="r", tool_name=AskUserTool.TOOL_NAME, parameters={"questions": questions})
    assert AskUserTool.answer(record, {"Files?": ["a, b", "c"]}).result == {"answers": {"Files?": ["a, b", "c"]}}
