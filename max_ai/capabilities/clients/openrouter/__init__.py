"""Module providing capability implementations and supporting utilities."""
from ._model import OpenRouterChatCompletionClientConfig
from .client import OpenRouterChatCompletionClient

__all__ = ["OpenRouterChatCompletionClient", "OpenRouterChatCompletionClientConfig"]
