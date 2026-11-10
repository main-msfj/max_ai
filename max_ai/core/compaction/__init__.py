"""Compaction machinery shared by every strategy: models, budget math and
atomic message blocks. Token counting lives in ``core.tokenizer``."""

from ..model.compaction import (
    CompactionConfig,
    CompactionOutput,
    CompactionResult,
    CompactionState,
    MemoryFactUpdate,
    MemoryMaintenanceOutput,
    MessageGroup,
)
from .budget import (
    client_max_output_tokens,
    live_message_budget_tokens,
    live_message_capacity_tokens,
    live_message_threshold_tokens,
)
from .groups import (
    current_turn_start,
    group_atomic_messages,
    split_recent_messages,
    turn_starts,
)

__all__ = [
    "CompactionConfig",
    "CompactionState",
    "CompactionOutput",
    "CompactionResult",
    "MemoryFactUpdate",
    "MemoryMaintenanceOutput",
    "MessageGroup",
    "client_max_output_tokens",
    "live_message_budget_tokens",
    "live_message_capacity_tokens",
    "live_message_threshold_tokens",
    "current_turn_start",
    "group_atomic_messages",
    "split_recent_messages",
    "turn_starts",
]
