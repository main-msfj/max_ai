"""The one tokenizer: token counting shared by messages, prompts, compaction,
tools and the CLI.

Depends on messages only for typing: ``core.messages`` imports this module
(to count its own tokens), so a runtime import back would cycle.
"""

from __future__ import annotations

import json
import logging
import typing as t
import weakref
from functools import lru_cache

import tiktoken
from pydantic import BaseModel, Field, PrivateAttr

from ..config import setting

if t.TYPE_CHECKING:
    from .messages import CoreMessage


logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _encoding(name: str) -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        fallback = setting.default_tokenizer
        if name == fallback:
            raise
        logger.warning("Tokenizer %r not found; falling back to %r", name, fallback)
        return tiktoken.get_encoding(fallback)


class TokenCounter(BaseModel):
    """Counts tokens with a tiktoken encoding.

    ``tokenizer_base`` defaults to ``setting.default_tokenizer``
    (``DEFAULT_TOKENIZER`` in ``.env``). For non-OpenAI models it is an
    approximation, good enough for budgeting.
    """

    tokenizer_base: str = Field(default_factory=lambda: setting.default_tokenizer)
    message_overhead_tokens: int = Field(default=5, ge=0)

    _encoder: tiktoken.Encoding = PrivateAttr()
    # Messages are immutable: each one is counted once per counter. Keyed by
    # id(), dropped when the message is garbage-collected.
    _counts: dict[int, int] = PrivateAttr(default_factory=lambda: dict[int, int]())

    def model_post_init(self, __context: t.Any) -> None:
        self._encoder = _encoding(self.tokenizer_base)

    def encode(self, text: str) -> list[int]:
        return self._encoder.encode(text) if text else []

    def decode(self, tokens: list[int]) -> str:
        return self._encoder.decode(tokens)

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoder.encode(text))

    @staticmethod
    def estimate(text: str, counter: TokenCounter | None = None) -> int:
        """Tokens of a text: the counter's when given, else 4 characters each."""
        if counter is not None:
            return counter.count_text(text)
        return len(text) // 4

    def count_message(self, message: CoreMessage) -> int:
        if message.token_count > 0:
            return message.token_count
        key = id(message)
        cached = self._counts.get(key)
        if cached is not None:
            return cached
        count = self._count_message(message)
        self._counts[key] = count
        weakref.finalize(message, self._counts.pop, key, None)
        return count

    def _count_message(self, message: CoreMessage) -> int:
        # Tool traffic is counted serialized: arguments and ids weigh too.
        # Assistant thinking is never sent back to the model, so it is left out.
        if message.role == "tool" or getattr(message, "tool_calls", None):
            return self.count_text(
                message.model_dump_json(exclude_none=True, exclude={"thinking"})
            )

        if message.is_multimodal():
            return self.count_text(self._message_budget_payload(message))

        return self.count_text(message.text()) + self.message_overhead_tokens

    def count_messages(self, messages: t.Iterable[CoreMessage]) -> int:
        return sum(self.count_message(message) for message in messages)

    def _message_budget_payload(self, message: CoreMessage) -> str:
        data = message.model_dump(exclude_none=True, exclude={"thinking"})

        content = data.get("content")
        if isinstance(content, list):
            for item in t.cast("list[t.Any]", content):
                if not isinstance(item, dict):
                    continue
                part = t.cast("dict[str, t.Any]", item)
                if part.get("data") is not None:
                    part["data"] = f"<{part.get('type', 'binary')}_bytes>"

        return json.dumps(data, ensure_ascii=False, default=str)


@lru_cache(maxsize=None)
def default_counter() -> TokenCounter:
    """Shared counter for ``setting.default_tokenizer``."""
    return TokenCounter()


__all__ = ["TokenCounter", "default_counter"]
