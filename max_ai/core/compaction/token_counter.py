"""Token counting shared by messages, compaction and telemetry.

Depends on messages only for typing: ``core.messages`` imports this module
(``CoreMessage.with_token_count``), so a runtime import back would cycle.
"""

from __future__ import annotations

import json
import logging
import typing as t
from functools import lru_cache

import tiktoken
from pydantic import BaseModel, Field, PrivateAttr

from ...config import setting

if t.TYPE_CHECKING:
    from ..messages import CoreMessage


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

    def count_message(self, message: CoreMessage) -> int:
        if message.token_count > 0:
            return message.token_count

        # Tool traffic is counted serialized: arguments and ids weigh too.
        if message.role == "tool" or getattr(message, "tool_calls", None):
            return self.count_serialized(message)

        if message.is_multimodal():
            return self.count_text(self._message_budget_payload(message))

        return self.count_text(message.text()) + self.message_overhead_tokens

    def count_messages(self, messages: t.Iterable[CoreMessage]) -> int:
        return sum(self.count_message(message) for message in messages)

    def count_serialized(self, value: t.Any) -> int:
        if isinstance(value, BaseModel):
            return self.count_text(value.model_dump_json(exclude_none=True))

        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)

        return self.count_text(text)

    def _message_budget_payload(self, message: CoreMessage) -> str:
        data = message.model_dump(exclude_none=True)

        content = data.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("data") is not None:
                    part_type = part.get("type", "binary")
                    part["data"] = f"<{part_type}_bytes>"

        return json.dumps(data, ensure_ascii=False, default=str)


@lru_cache(maxsize=None)
def default_counter() -> TokenCounter:
    """Shared counter for ``setting.default_tokenizer``."""
    return TokenCounter()


__all__ = ["TokenCounter", "default_counter"]
