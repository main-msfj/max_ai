"""
Core contract for chat completion clients.

A client wraps a single LLM provider's chat-completion endpoint
(Ollama, Anthropic, OpenAI, etc.) and exposes a normalized interface
the agent can use without knowing the provider's quirks.

What varies by provider — and lives in concrete subclasses:
  - How rendered prompt layers are arranged into the message list
    (some put everything in a system field, some interleave).
  - The wire format of messages, tools, and tool calls.
  - The shape of the usage / token-count payload.
  - The streaming chunk format.
  - Reasoning / "thinking" extraction (some surface it as a separate
    field, others embed it in the content with tags).
  - The actual HTTP/API call (``complete`` / ``stream``).

What stays the same — and is the framework's contract:
  - ``run()`` is the public entry point.
  - ``ChatCompletionResult`` and ``ChatCompletionChunk`` are the
    normalized return types.
  - ``Usage`` is the normalized token / call statistics object.
  - Tool inputs are ``CoreTool`` instances; the client converts to
    provider-native schema internally.
"""

from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, SecretStr

from .tools import CoreTool
from .component_config import ComponentBase

from ..types.stacks import PromptCtx
from ..core.blocks import CoreMessage
from ..core.models import ModelConfig
from ..errors.client import ClientError
from ..types.completions import ChatCompletionChunk, ChatCompletionResult, Usage

if t.TYPE_CHECKING:
    from ..types.run_context import RunContext

T = t.TypeVar("T")

class CoreChatCompletionClient(ComponentBase[BaseModel], ABC):
    """Abstract base class for all MaxAI chat completion clients.

    Subclasses implement six abstract methods that cover the
    provider-specific surface area. ``run()`` is concrete and shared:
    it validates inputs, runs the conversion pipeline, and dispatches
    to ``complete`` or ``stream`` based on the ``stream`` flag.
    """

    # -------- CONSTRUCTION -----------------------------------------------------------
    def __init__(
        self,
        model: str,
        api_key: str | SecretStr | None = None,
        config: ModelConfig | None = None,
        **kwargs: t.Any,
    ) -> None:
        """Initialize a client instance.

        Args:
            model: The provider-specific model identifier
                (e.g. ``"qwen3:4b"``, ``"claude-opus-4"``,
                ``"gpt-4o"``).
            api_key: Authentication key. Optional — providers like
                local Ollama need none. Concrete clients decide what
                to do when ``None`` (read from env var, fall back to
                no auth, etc.).
            config: Model capabilities and defaults. If None, an
                empty ``ModelConfig`` is used. The ``run()`` method
                consults ``config.supports_function_calling`` to
                decide whether to allow tool calls.
            **kwargs: Provider-specific defaults (e.g. ``temperature``,
                ``top_p``). Concrete clients pull what they need.
        """
        self.model: str = self._require_type(model, str, "model")
        self.api_key: SecretStr | None = self._resolve_api_key(api_key)
        self.config: ModelConfig = self._require_type(
            config or ModelConfig(), ModelConfig, "config"
        )

    @staticmethod
    def _require_type(value: t.Any, expected: type[T], field: str) -> T:
        if not isinstance(value, expected):
            raise TypeError(
                f"{field} must be {expected.__name__}, got {type(value).__name__}"
            )
        return value

    @staticmethod
    def _resolve_api_key(key: str | SecretStr | None) -> SecretStr | None:
        """Normalize an API key argument to ``SecretStr | None``."""
        if key is None:
            return None
        return key if isinstance(key, SecretStr) else SecretStr(key)

    # -------- PUBLIC ENTRY POINT -----------------------------------------------------------
    async def run(
        self,
        ctx: "RunContext",
        prompts: PromptCtx,
        tools: list["CoreTool"] | None = None,
        output_format: t.Type[BaseModel] | None = None,
        stream: bool = False,
        **kwargs: t.Any,
    ) -> t.Union[ChatCompletionResult, t.AsyncGenerator[ChatCompletionChunk, None]]:
        """Execute a chat completion request.

        Pipeline:
          1. Validate that tools, if any, are supported by the model.
          2. ``format_messages`` — combine ``ctx`` (chat history,
             user message) and ``prompts.rendered_layers`` into the
             provider-agnostic ``list[CoreMessage]``.
          3. ``build_api_messages`` — convert to the provider's wire
             format (``list[dict]``).
          4. ``build_tool_schema`` — convert ``CoreTool`` instances
             to the provider's tool schema (``list[dict]``).
          5. Dispatch to ``complete`` or ``stream``.

        Args:
            ctx: Per-turn runtime state (chat history, user message,
                anything else carried across the turn).
            prompts: Rendered prompt layers from ``agent.prepare()``.
            tools: Tools the LLM may call this turn. ``None`` or empty
                disables tool calling for the request.
            output_format: Optional Pydantic model for structured
                output. Concrete clients map this to their provider's
                JSON-mode / response-format mechanism.
            stream: If True, returns an async generator of chunks.
            **kwargs: Provider-specific per-call overrides
                (``temperature``, ``cache_control``, etc.).

        Returns:
            ``ChatCompletionResult`` when ``stream=False``,
            ``AsyncGenerator[ChatCompletionChunk, None]`` otherwise.

        Raises:
            ClientError: If tools are passed but the model's
                ``ModelConfig.supports_function_calling`` is False.
        """
        if tools and not self.config.supports_function_calling:
            raise ClientError.tools_not_supported(
                model=self.model, tool_count=len(tools)
            )

        messages = self.format_messages(ctx, prompts)
        api_messages = self.build_api_messages(messages)
        tool_schema = self.build_tool_schema(tools or [])

        if stream:
            return self.stream(api_messages, tool_schema, output_format, **kwargs)
        return await self.complete(api_messages, tool_schema, output_format, **kwargs)

    # -------- ABSTRACT — MESSAGE ASSEMBLY -----------------------------------------------------------
    @abstractmethod
    def format_messages(
        self, ctx: "RunContext", prompts: PromptCtx
    ) -> list[CoreMessage]:
        """Assemble the full message list for the LLM call.

        Each provider arranges ``prompts.rendered_layers`` differently:

        - Anthropic-style: collapse all layers into a single system
          field, then chat history and user message as messages.
        - OpenAI-style: similar, but the system prompt goes as the
          first message with ``role="system"``.
        - Ollama: model-dependent; usually OpenAI-style.

        Args:
            ctx: Per-turn runtime state.
            prompts: Rendered prompt layers (system content) plus
                whatever metadata the provider needs.

        Returns:
            Provider-agnostic ``list[CoreMessage]`` in the order the
            provider expects: typically system first, then chat
            history, then the current user message.
        """
        ...

    @abstractmethod
    def build_api_messages(self, messages: list[CoreMessage]) -> list[dict[str, t.Any]]:
        """Convert ``CoreMessage`` instances to the provider's wire format.

        Args:
            messages: The output of ``format_messages``.

        Returns:
            A list of dicts ready to be sent to the provider's API.
        """
        ...

    # -------- ABSTRACT — TOOL SCHEMA -----------------------------------------------------------
    @abstractmethod
    def build_tool_schema(self, tools: list["CoreTool"]) -> list[dict[str, t.Any]]:
        """Convert ``CoreTool`` instances to the provider's tool schema.

        Each provider has its own schema (Anthropic: ``input_schema``,
        OpenAI: ``parameters`` with ``type=function``, Ollama:
        OpenAI-style). The conversion includes turning each tool's
        Pydantic input model into JSON Schema and wrapping it as the
        provider expects.

        Args:
            tools: The ``CoreTool`` list passed to ``run()``. May be
                empty.

        Returns:
            A list of provider-native tool schemas. Empty when
            ``tools`` is empty.
        """
        ...

    # -------- ABSTRACT — USAGE -----------------------------------------------------------
    @abstractmethod
    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        """Normalize the provider's usage payload into framework ``Usage``.

        Each provider exposes token counts under different field names
        (Anthropic: ``input_tokens`` / ``output_tokens``, OpenAI:
        ``prompt_tokens`` / ``completion_tokens``, Ollama:
        ``prompt_eval_count`` / ``eval_count``).

        Args:
            usage: Whatever the provider returned. Type varies.

        Returns:
            A populated ``Usage`` instance.
        """
        ...

    # -------- ABSTRACT — API CALLS -----------------------------------------------------------
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]] | None,
        output_format: t.Type[BaseModel] | None,
        **kwargs: t.Any,
    ) -> ChatCompletionResult:
        """Execute a single (non-streaming) chat completion call.

        Args:
            messages: Provider-native message list (output of
                ``build_api_messages``).
            tools: Provider-native tool schemas (output of
                ``build_tool_schema``). May be empty or None.
            output_format: Optional structured-output model.
            **kwargs: Provider-specific per-call options.

        Returns:
            A normalized ``ChatCompletionResult``.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]] | None,
        output_format: t.Type[BaseModel] | None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[ChatCompletionChunk, None]:
        """Execute a streaming chat completion call.

        Args:
            messages: Provider-native message list.
            tools: Provider-native tool schemas.
            output_format: Optional structured-output model.
            **kwargs: Provider-specific per-call options.

        Yields:
            ``ChatCompletionChunk`` instances. The final chunk has
            ``is_complete=True`` and carries the cumulative ``usage``.
        """
        ...
