"""Compaction machinery shared by every strategy: models and atomic
message blocks. Token counting lives in ``core.tokenizer``."""

from ..model.compaction import (
    CompactionState,
    MessageGroup,
)
from .groups import (
    group_atomic_messages,
    split_recent_messages,
)

__all__ = [
    "CompactionState",
    "MessageGroup",
    "group_atomic_messages",
    "split_recent_messages",
]
