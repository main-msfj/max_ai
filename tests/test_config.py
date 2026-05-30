from __future__ import annotations


from max_ai.config import Settings


def test_compaction_settings_defaults():
    settings = Settings(_env_file=None)

    assert settings.compaction_prompt_budget_tokens == 6000
    assert settings.compaction_summary_budget_tokens == 2000
    assert settings.compaction_safety_margin_ratio == 0.05
    assert settings.compaction_live_message_threshold == 0.8
    assert settings.compaction_live_message_keep_ratio == 0.2
    assert settings.compaction_min_output_tokens == 1024


def test_compaction_settings_can_be_overridden_from_env(monkeypatch):
    monkeypatch.setenv("COMPACTION_PROMPT_BUDGET_TOKENS", "3100")
    monkeypatch.setenv("COMPACTION_SUMMARY_BUDGET_TOKENS", "2100")
    monkeypatch.setenv("COMPACTION_SAFETY_MARGIN_RATIO", "0.1")
    monkeypatch.setenv("COMPACTION_LIVE_MESSAGE_THRESHOLD", "0.35")
    monkeypatch.setenv("COMPACTION_LIVE_MESSAGE_KEEP_RATIO", "0.22")
    monkeypatch.setenv("COMPACTION_MIN_OUTPUT_TOKENS", "1200")

    settings = Settings(_env_file=None)

    assert settings.compaction_prompt_budget_tokens == 3100
    assert settings.compaction_summary_budget_tokens == 2100
    assert settings.compaction_safety_margin_ratio == 0.1
    assert settings.compaction_live_message_threshold == 0.35
    assert settings.compaction_live_message_keep_ratio == 0.22
    assert settings.compaction_min_output_tokens == 1200
