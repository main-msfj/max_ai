"""Global settings for MaxAI.

Every field can be set in the environment or ``.env`` with its uppercase
name, e.g. ``MAX_LOOP_ITERATIONS=30``. ``root_dir`` reads ``HOST_WORKSPACE``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global Settings for Max AI Framework"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------- HOST -------------------------------------------------------------
    root_dir: Path = Field(default_factory=Path.cwd, validation_alias="HOST_WORKSPACE")
    default_tokenizer: str = Field(
        default="o200k_base",
        description="tiktoken encoding; a model's tokenizer_base overrides it.",
    )

    # -------- LOOP -------------------------------------------------------------
    max_loop_iterations: int = Field(default=20, ge=1)
    environment_idle_timeout: float = Field(default=300, ge=0)

    # -------- TOOLS ------------------------------------------------------------
    max_parallel_tools: int = Field(
        default=8, ge=1, description="Read-only calls of one reply that may run at once."
    )
    max_tool_result_chars: int = Field(default=30_000, ge=1_000)
    tool_timeout_seconds: float = Field(
        default=360, gt=0, description="Max seconds one bash command may run."
    )
    tool_name_pattern: str = Field(default=r"[a-z][a-z0-9_]{0,63}")
    tool_description_min_tokens: int = Field(default=5, ge=1)
    tool_description_max_tokens: int = Field(default=150, ge=1)

    # -------- COMPACTION -------------------------------------------------------
    compaction_prompt_budget_tokens: int = Field(default=6000, gt=0)
    compaction_summary_budget_tokens: int = Field(default=2000, gt=0)
    compaction_safety_margin_ratio: float = Field(default=0.05, ge=0, lt=1)
    compaction_live_message_threshold: float = Field(default=0.8, gt=0, lt=1)
    compaction_live_message_keep_ratio: float = Field(default=0.2, gt=0, lt=1)
    compaction_min_output_tokens: int = Field(default=1024, gt=0)
    compaction_min_keep_groups: int = Field(
        default=3,
        ge=1,
        description="Recent assistant+tool groups always kept, so the model does not "
        "re-call tools whose results vanished.",
    )
    compaction_capacity_warning_tokens: int = Field(
        default=2000, ge=0, description="Warn when less room than this is left for messages."
    )
    compaction_summary_message_cap_tokens: int = Field(
        default=2000, gt=0, description="Each message is cut to this in the summary request."
    )

    @field_validator("root_dir")
    @classmethod
    def _resolve_root(cls, v: Path) -> Path:
        """Store root_dir absolute, so relative HOST_WORKSPACE paths work."""
        return v.expanduser().resolve()


setting = Settings()
