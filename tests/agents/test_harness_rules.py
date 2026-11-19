import logging

import pytest

from max_ai.agents.agent import Agent
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.capabilities.compaction import SummaryCompaction
from max_ai.capabilities.memory import Memory
from max_ai.capabilities.middleware.tracing import TracingMiddleware
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.stacks.custom_layer import CustomLayer
from max_ai.capabilities.tools.bash import BashTool
from max_ai.capabilities.tools.decorator import tool as as_tool
from max_ai.config import Settings, setting
from max_ai.core.harness.rules import Harness
from max_ai.core.model.llm import ModelConfig
from max_ai.errors.harness import HarnessError


def weather(city: str) -> str:
    """The weather in a city."""
    return city


def names(tools):
    return [tool.name for tool in tools]


def test_a_long_tool_description_is_refused_and_a_short_one_passes():
    Harness.tool_rules([as_tool(weather)])
    long = as_tool(weather, description="Tells the weather. " * 40)
    with pytest.raises(HarnessError, match="'weather' takes about .* the most is 100"):
        Harness.tool_rules([long])


def test_the_prompt_warns_when_short_and_is_refused_over_a_tenth_of_the_window(caplog):
    with caplog.at_level(logging.WARNING):
        Harness.prompt_rules(500, 128_000)
    assert "cache is not used well" in caplog.text
    Harness.prompt_rules(12_800, 128_000)
    with pytest.raises(HarnessError, match="the most is 12800"):
        Harness.prompt_rules(12_801, 128_000)


def test_long_instructions_are_sent_to_a_skill_and_a_long_description_is_refused():
    Harness.instruction_rules("Be brief.")
    with pytest.raises(HarnessError, match="instructions take about .* the most is 100"):
        Harness.instruction_rules("word " * 600)
    Harness.custom_prompt_rules([CustomLayer("Answer in Spanish.")])
    with pytest.raises(HarnessError, match="Write a skill"):
        Harness.custom_prompt_rules([CustomLayer("word " * 8000)])
    with pytest.raises(HarnessError, match="the most is 100"):
        Harness.description_rules("word " * 600)


def test_no_tool_may_run_longer_than_the_sandbox_lives(monkeypatch):
    assert Harness.sandbox_rules() == 600
    monkeypatch.setattr(setting, "tool_timeout_seconds", 900)
    with pytest.raises(HarnessError, match="lower the first or raise the second"):
        Harness.sandbox_rules()


def test_our_tools_are_added_and_the_developers_win(tmp_path):
    assert names(Harness.tool_combination([], None, None)) == ["ask_user", "write_todos"]
    sandbox = LocalSandbox(str(tmp_path))
    mine = as_tool(weather, name="read_file")
    got = names(Harness.tool_combination([mine, BashTool()], sandbox, None))
    assert got == [
        "read_file", "bash", "ask_user", "write_todos", "write_file", "edit_file",
        "enter_worktree", "exit_worktree",  # they come with bash
    ]
    assert "enter_worktree" not in names(Harness.tool_combination([], sandbox, None))


def test_skills_need_a_sandbox_and_bash(tmp_path):
    skills = object.__new__(type("S", (), {}))  # only its presence matters here
    sandbox = LocalSandbox(str(tmp_path))
    # The rules only check; the list of tools is made after them.
    with pytest.raises(HarnessError, match="pass sandbox"):
        Harness.combination_rules([], None, skills)
    with pytest.raises(HarnessError, match="bash runs its commands in a sandbox"):
        Harness.combination_rules([BashTool()], None, None)
    with pytest.raises(HarnessError, match="memory is kept in a sandbox"):
        Harness.combination_rules([], None, None, Memory())
    with pytest.raises(HarnessError, match="add BashTool"):
        Harness.combination_rules([], sandbox, skills)
    Harness.combination_rules([BashTool()], sandbox, skills)
    # An agent with skills gets the tool that reads them, before the worktree ones.
    got = names(Harness.tool_combination([BashTool()], sandbox, skills))
    assert got[0] == "bash" and got.index("read_skill") < got.index("enter_worktree")
    assert "read_skill" not in names(Harness.tool_combination([BashTool()], sandbox, None))


def test_tracing_is_added_once():
    assert isinstance(Harness.middleware_rules([])[0], TracingMiddleware)
    mine = TracingMiddleware(capture_content=False)
    assert Harness.middleware_rules([mine]) == [mine]


def test_the_summary_uses_its_own_model_only_with_a_model_and_a_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(setting, "summary_model", "openai/gpt-5-nano")
    assert Harness.compaction_rules().client is None
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    made = Harness.compaction_rules()
    assert isinstance(made, SummaryCompaction) and made.client.model == "openai/gpt-5-nano"


def test_the_agent_applies_the_rules_when_it_is_built():
    client = OpenRouterChatCompletionClient("openai/gpt-test", api_key="sk-test")
    made = Agent("helper", "Answers questions.", "Be brief.", client, toolset=[as_tool(weather)])
    assert names(made.tools) == ["weather", "ask_user", "write_todos"] and names(made.toolset) == ["weather"]
    assert isinstance(made.middlewares[0], TracingMiddleware)
    with pytest.raises(HarnessError, match="instructions take about"):
        Agent("helper", "Answers questions.", "word " * 600, client)


def test_the_harness_sets_the_models_window_and_output(caplog):
    config = ModelConfig(max_context_window=1_000_000, max_output_tokens=64_000)
    client = OpenRouterChatCompletionClient("openai/gpt-test", api_key="sk-test", config=config)
    with caplog.at_level(logging.WARNING):
        Agent("helper", "Answers questions.", "Be brief.", client)
    assert (client.config.max_context_window, client.config.max_output_tokens) == (128_000, 12_800)
    assert "was 1000000" in caplog.text


def test_the_most_of_the_prompt_can_be_changed_from_the_environment(monkeypatch):
    monkeypatch.setenv("HARNESS_PROMPT_MAX_SHARE", "0.2")
    assert Settings().harness_prompt_max_share == 0.2


def test_all_the_tools_together_stay_under_the_limit():
    Harness.tools_total_rules(20_000)
    with pytest.raises(HarnessError, match="the most is 20000"):
        Harness.tools_total_rules(20_001)


def test_a_model_call_is_tried_three_times_at_most_and_the_waits_are_kept():
    from max_ai.core.retry import RetryPolicy

    made = Harness.retry_rules(RetryPolicy(attempts=10, base_delay=0.5))
    assert (made.attempts, made.base_delay) == (3, 0.5)
