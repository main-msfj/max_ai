"""Module providing capability implementations and supporting utilities."""
from ._model import OllamaChatCompletionClientConfig
from .client import OllamaChatCompletionClient

__all__ = ["OllamaChatCompletionClient", "OllamaChatCompletionClientConfig"]
