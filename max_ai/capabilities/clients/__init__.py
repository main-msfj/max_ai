"""Module providing capability implementations and supporting utilities."""
from .to_delete.openai import OpenAIChatCompletionClient
from .to_delete.openrouter import OpenRouterChatCompletionClient

__all__ = [
    "OpenAIChatCompletionClient",
    "OpenRouterChatCompletionClient",
]
