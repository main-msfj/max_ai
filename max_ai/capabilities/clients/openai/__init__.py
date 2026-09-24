"""Module providing capability implementations and supporting utilities."""
from ._model import OpenAIChatCompletionClientConfig
from .client import OpenAIChatCompletionClient

__all__ = ["OpenAIChatCompletionClient", "OpenAIChatCompletionClientConfig"]
