"""Global settings for MaxAI.

Every field can be set in the environment or ``.env`` with its uppercase
name, e.g. ``HARNESS_MAX_ITERATIONS=30``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global Settings for Max AI Framework"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------- HOST -------------------------------------------------------------
    max_ai_home: Path = Field(default_factory=lambda: Path.home() / ".max_ai")
    default_tokenizer: str = Field(
        default="o200k_base",
        description="tiktoken encoding; a model's tokenizer_base overrides it.",
    )

    # -------- TOOLS ------------------------------------------------------------
    tool_timeout_seconds: float = Field(
        default=360, gt=0, description="Max seconds one bash command may run."
    )

    # -------- HARNESS ----------------------------------------------------------
    harness_max_iterations: int = Field(
        default=50,
        ge=1,
        validation_alias="HARNESS_MAX_ITERATIONS",
        description="Model calls one turn may make, in the loop the harness creates.",
    )
    harness_parallel_tools: int = Field(
        default=8,
        ge=1,
        validation_alias="HARNESS_PARALLEL_TOOLS",
        description="Read-only calls of one reply that may run at once.",
    )
    harness_tool_result_chars: int = Field(
        default=40_000,
        ge=1_000,
        validation_alias="HARNESS_TOOL_RESULT_CHARS",
        description="A tool result longer than this is saved to a file and shown cut.",
    )
    harness_tool_description_max_tokens: int = Field(default=100, ge=1)
    harness_tools_max_tokens: int = Field(
        default=20_000,
        ge=1,
        validation_alias="HARNESS_TOOLS_MAX_TOKENS",
        description="Most all the loaded tools together may take in a request.",
    )
    harness_deferred_tools_min_tokens: int = Field(
        default=5_000,
        ge=0,
        validation_alias="HARNESS_DEFERRED_TOOLS_MIN_TOKENS",
        description="Deferred tools that take less than this together are loaded directly.",
    )
    harness_prompt_min_tokens: int = Field(
        default=2000, ge=0, description="Below this the provider's cache is not used well."
    )
    harness_prompt_max_share: float = Field(
        default=0.10,
        gt=0,
        le=1,
        validation_alias="HARNESS_PROMPT_MAX_SHARE",
        description="Most of the context window the prompt may take.",
    )
    harness_custom_prompt_max_tokens: int = Field(
        default=1500,
        ge=1,
        validation_alias="HARNESS_CUSTOM_PROMPT_MAX_TOKENS",
        description="Most the developer's own prompt layers may take; more belongs in a skill.",
    )
    harness_instructions_max_tokens: int = Field(default=100, ge=1)
    harness_description_max_tokens: int = Field(default=100, ge=1)
    harness_retry_attempts: int = Field(
        default=3,
        ge=1,
        validation_alias="HARNESS_RETRY_ATTEMPTS",
        description="Tries in total for one model call: the first one plus its retries.",
    )
    harness_sandbox_seconds: float = Field(
        default=600,
        gt=0,
        validation_alias="HARNESS_SANDBOX_SECONDS",
        description="How long a sandbox lives; no tool may run longer.",
    )
    summary_model: str = Field(
        default="",
        validation_alias="SUMMARY_MODEL",
        description="OpenRouter model that writes the summaries; empty: the agent's own.",
    )
    harness_memory_at_compaction: bool = Field(
        default=True,
        validation_alias="HARNESS_MEMORY_AT_COMPACTION",
        description="When messages leave, what the user said that is worth remembering "
        "is saved to the memory.",
    )
    harness_judge_model: str = Field(
        default="",
        validation_alias="HARNESS_JUDGE_MODEL",
        description="OpenRouter decisions model that judges bash commands without a rule; "
        "empty: no judge, the user is asked.",
    )
    harness_judge_min_safe: float = Field(
        default=0.9,
        ge=0.5,
        le=1,
        validation_alias="HARNESS_JUDGE_MIN_SAFE",
        description="How sure the judge must be to run a command.",
    )
    harness_judge_seconds: float = Field(
        default=3,
        gt=0,
        validation_alias="HARNESS_JUDGE_SECONDS",
        description="Most the judge may take; after it the user is asked.",
    )

    # -------- COMPACTION -------------------------------------------------------
    compaction_prompt_budget_tokens: int = Field(default=6000, gt=0)
    compaction_summary_max_share: float = Field(
        default=0.10,
        gt=0,
        le=1,
        description="Most of the context window the summary may take; the model is told.",
    )
    compaction_safety_margin_ratio: float = Field(default=0.05, ge=0, lt=1)
    compaction_live_message_threshold: float = Field(default=0.8, gt=0, lt=1)
    compaction_live_message_keep_ratio: float = Field(default=0.2, gt=0, lt=1)
    compaction_min_output_tokens: int = Field(default=1024, gt=0)
    compaction_keep_groups: int = Field(
        default=2, ge=1, description="Last exchanges a compaction never touches."
    )
    compaction_result_chars: int = Field(
        default=2_000,
        ge=100,
        description="An old tool result longer than this is cut when the context fills.",
    )

    def create_global(self) -> Path:
        """Create ``~/.max_ai/setting.global.json`` if it is missing; never
        overwrite it. The Agent calls it when it starts."""
        home = self.max_ai_home.expanduser().resolve()
        home.mkdir(parents=True, exist_ok=True)
        path = home / "setting.global.json"
        # "x" fails if the file exists: two starts at once never overwrite.
        try:
            with path.open("x", encoding="utf-8") as file:
                file.write(json.dumps({"skills": {}}, indent=2) + "\n")
        except FileExistsError:
            pass
        return path


setting = Settings()
