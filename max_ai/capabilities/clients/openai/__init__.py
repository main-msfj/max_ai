"""The OpenAI client."""
from ._model import OpenAIChatCompletionClientConfig
from .client import OpenAIChatCompletionClient

__all__ = ["OpenAIChatCompletionClient", "OpenAIChatCompletionClientConfig"]
