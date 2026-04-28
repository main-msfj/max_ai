"""
Ollama BaseChatCompletionClient Implementation.

This module provides integration with Ollama local LLM server using the
official ollama-python client library. Supports tool calling, thinking mode,
streaming, and structured outputs.
"""

import json
import time
import logging
import jsonref
import tiktoken
import typing as t
from pydantic import BaseModel, Field, SecretStr

try:
    from ollama import AsyncClient, ResponseError
    from ollama import ChatResponse
except ImportError:
    raise ImportError(
        "Ollama library not installed. "
        "Please install it with 'pip install ollama>=0.4.0' to use OllamaChatCompletionClient."
    )

from ._base import (
    AuthenticationError,
    BaseChatCompletionClient,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
)

from ..agent_config import ModelConfig
from ..context import AgentContext
from ..loggers import ScopedLogger

from .._common import PromptContext
from .._component_config import Component

from ..models import ChatCompletionChunk, ChatCompletionResult, Usage
from ..messages import (
    BaseMessage,
    SystemMessage,
    AssistantMessage,
    ToolCallRecord,
    MultiModalMessage,
    ToolMessage,
)

# -------- LOGGER -----------------------------------------------------------
logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="OllamaChatCompletionClient")


class OllamaChatCompletionClientConfig(BaseModel):
    """Configuration for OllamaChatCompletionClient serialization."""

    model: str
    host: str = "http://localhost:11434"
    api_key: SecretStr | None = None
    think: bool | None = None
    keep_alive: str | int | None = None
    config: dict[str, t.Any] = Field(default_factory=dict)


class OllamaChatCompletionClient(
    Component[OllamaChatCompletionClientConfig], BaseChatCompletionClient
):
    """
    Ollama implementation of BaseChatCompletionClient.

    Supports local models (Qwen3, Llama3, Mistral, Phi-4, Gemma3, etc.) with
    tool calling, thinking mode, streaming, and structured outputs.
    """

    component_config_schema = OllamaChatCompletionClientConfig
    component_type = "client"
    component_provider_override = "maxais.llm.OllamaChatCompletionClient"

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        api_key: SecretStr | None = None,
        model_config: ModelConfig | None = None,
        think: bool | None = None,
        keep_alive: str | int | None = None,
        **kwargs: t.Any,
    ):
        """
        Initialize OllamaChatCompletionClient.

        Args:
            model: The model identifier (e.g., "qwen3:4b-thinking-2507-q4_K_M", "llama3.2:3b")
            host: Ollama server URL (default: http://localhost:11434)
            api_key: Optional API key (only required for hosted Ollama or custom auth proxies)
            model_config: Optional metadata about the model being used
            think: Enable thinking mode for reasoning models. None = model default,
                   True = force enable, False = force disable
            keep_alive: How long the model stays loaded in memory (e.g., "5m", 0 to unload)
            **kwargs: Additional configuration parameters
        """
        super().__init__(model, api_key, model_config=model_config, **kwargs)

        self.host = host
        self.think = think
        self.keep_alive = keep_alive

        # Ollama local deployments don't require auth, but hosted/proxied ones might
        headers: dict[str, str] = {}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"

        self.client = AsyncClient(host=host, headers=headers or None)

        # Default settings
        self.VALIDATION_FREQUENCY = 5  # Number of loops before checking JSON tools

    # -------- PROVIDER HOOKS -----------------------------------------------------------
    def resolve_model_name(self) -> str:
        """Return the Ollama model name."""
        if self.model_config and self.model_config.name:
            return self.model_config.name
        return self.model

    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        """
        Normalize Ollama usage payload into internal Usage model.

        Ollama provides token counts directly in the response object:
            - prompt_eval_count: input tokens processed
            - eval_count: output tokens generated
            - total_duration: nanoseconds
            - load_duration: nanoseconds (model load time)
            - prompt_eval_duration: nanoseconds
            - eval_duration: nanoseconds

        Note: Ollama does not support cached tokens — always 0.

        Args:
            usage: ChatResponse object (or object with eval_count attrs)
        Returns:
            Usage: Normalized usage data in internal format
        """
        prompt_tokens = int(getattr(usage, "prompt_eval_count", 0) or 0)
        completion_tokens = int(getattr(usage, "eval_count", 0) or 0)

        # total_duration is in nanoseconds → convert to ms
        total_ns = int(getattr(usage, "total_duration", 0) or 0)
        duration_ms = total_ns // 1_000_000

        return Usage(
            duration_ms=duration_ms,
            tokens_input=prompt_tokens,
            tokens_output=completion_tokens,
            tokens_cached=0,  # Ollama doesn't support prompt caching
        )

    def build_api_messages(
        self, messages: list[BaseMessage]
    ) -> list[dict[str, t.Any]]:
        """
        Convert internal Messages to Ollama API format.

        Ollama uses an OpenAI-compatible message structure with some differences:
            - Images go in a top-level `images` array (list of base64 strings), NOT inline in content
            - Tool calls use OpenAI-like structure but arguments are objects, not JSON strings
            - Tool responses use role="tool" without tool_call_id (Ollama matches by order)

        Args:
            messages: List of internal Message objects

        Returns:
            List of messages formatted for Ollama API
        """
        api_messages: list[dict[str, t.Any]] = []
        for msg in messages:
            # Handle MultiModalMessage — Ollama uses images array, not content parts
            if isinstance(msg, MultiModalMessage):
                api_msg: dict[str, t.Any] = dict(
                    role=msg.role, content=msg.content or ""
                )

                if msg.is_image():
                    images: list[str] = []
                    if msg.data:
                        # Ollama expects base64 strings (no data URL prefix)
                        images.append(msg.to_base64())
                    elif msg.media_url:
                        # Ollama supports URLs too in recent versions
                        images.append(msg.media_url)

                    if images:
                        api_msg["images"] = images

                # Note: Ollama currently doesn't support audio/video in /api/chat
            else:
                # Regular message
                api_msg: dict[str, t.Any] = dict(
                    role=msg.role, content=msg.content or ""
                )

                # Assistant messages with tool calls
                if isinstance(msg, AssistantMessage) and msg.tool_calls:
                    api_msg["tool_calls"] = [
                        {
                            "function": {
                                "name": tc.tool_name,
                                # Ollama wants dict, NOT JSON string (different from OpenAI)
                                "arguments": tc.parameters
                                if isinstance(tc.parameters, dict)  # type: ignore
                                else json.loads(tc.parameters)
                                if isinstance(tc.parameters, str)
                                else {},
                            },
                        }
                        for tc in msg.tool_calls
                    ]

                # Tool response messages
                if isinstance(msg, ToolMessage):
                    # Ollama uses role="tool" and optionally `name` for the tool
                    if hasattr(msg, "tool_name") and msg.tool_name:
                        api_msg["name"] = msg.tool_name

            api_messages.append(api_msg)
        return api_messages

    def format_messages(
        self, ctx: AgentContext, prompts: PromptContext
    ) -> list[BaseMessage]:
        """
        Ollama-specific logic to assemble the final message list.

        Args:
            prompts: PromptContext containing various system prompts
            ctx: Agent context to include recent messages (e.g., from memory)
        Returns:
            List of messages ready to be sent through the pipeline
        """
        system_parts: list[str] = [prompts.behavior_ptx]

        # -------- PROMPTS ORDER -----------------------------------------------------------
        dynamic_layers: list[str | None] = [
            prompts.memory_management_ptx,
            prompts.thread_summary_ptx,
            prompts.persistent_memory_ptx,
            prompts.knowledge_ptx,
            prompts.required_tools_ptx,
        ]

        for layer in dynamic_layers:
            if layer:
                system_parts.append(layer)

        system_message = SystemMessage(
            content="\n\n".join(system_parts), source=self.model
        )

        ll_messages: list[BaseMessage] = [system_message]

        if ctx.messages_ctx.message_history:
            ll_messages.extend(ctx.messages_ctx.message_history)

        if ctx.new_messages:
            ll_messages.extend(ctx.new_messages)

        return ll_messages

    def adapt_schema_for_provider(self, schema: dict[str, t.Any]) -> dict[str, t.Any]:
        """
        Adapt JSON schema for Ollama structured outputs.

        Ollama's `format` parameter accepts a standard JSON schema, but nested
        $refs (common in Pydantic-generated schemas with nested models) can
        cause issues with some Ollama runtimes. We inline all $refs into a
        single flat schema using jsonref (spec-compliant, handles cycles).

        Unlike OpenAI, Ollama does NOT require:
            - additionalProperties: false
            - all fields marked as required
            - strict mode

        Args:
            schema: The JSON schema to adapt (may contain $defs/$refs)

        Returns:
            Flat schema with all $refs inlined
        """
        # jsonref lazily wraps $refs in proxy objects; json roundtrip forces
        # full materialization into a plain dict
        resolved = jsonref.replace_refs(schema, proxies=False, lazy_load=False)
        flat: dict[str, t.Any] = json.loads(json.dumps(resolved))

        # Clean up definitions sections (no longer needed after inlining)
        flat.pop("$defs", None)
        flat.pop("definitions", None)
        return flat

    def build_tool_schema(
        self, tool_definitions: list[dict[str, t.Any]]
    ) -> list[dict[str, t.Any]]:
        """
        Wraps neutral tool definitions into Ollama's tool format.
        Ollama follows the OpenAI tool schema structure for tool calling.

        Args:
            tool_definitions: List of neutral tool descriptors
        Returns:
            List of tools formatted for Ollama API
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": td["name"],
                    "description": td["description"],
                    "parameters": td["parameters"],
                },
            }
            for td in tool_definitions
        ]

    # -------- TOKEN ESTIMATION -----------------------------------------------------------
    def estimate_message_tokens(self, content: str) -> int:
        """
        Estimate token count using tokenizer_base from ModelConfig.

        Note: tiktoken estimates are approximate for non-OpenAI models
        (Qwen, Llama, Mistral use different tokenizers). For accurate
        counts, use eval_count / prompt_eval_count from the response.
        """
        if self.model_config is None:
            log.warning("No Model Config for Token Counting — Using fallback method")
            return max(1, int(len(content) / 4))

        try:
            encoding = tiktoken.get_encoding(self.model_config.tokenizer_base)
            tokens = len(encoding.encode(content)) + 5  # Chat template overhead
            return tokens
        except Exception as e:
            log.warning(
                msg="Failed to Count Tokens — Using fallback",
                error=str(e),
                tokenizer=self.model_config.tokenizer_base,
            )

        return max(1, int(len(content) / 4)) + 5

    def estimate_tool_tokens(self, tool: dict[str, t.Any]) -> int:
        """Estimate tokens for tool definitions based on ModelConfig."""
        fall_back_estimate = 100
        if self.model_config is None:
            log.warning(
                msg="No Model Config for Token Counting — Using fallback for tool calls",
                model=self.model,
            )
            return fall_back_estimate

        try:
            encoding = tiktoken.get_encoding(self.model_config.tokenizer_base)
            total_tokens = 12  # Base tool call overhead
            total_tokens += 7  # Tool call markers

            func = tool.get("function", {})

            # Name and description
            name_desc = f"{func.get('name', '')}:{func.get('description', '')}"
            total_tokens += len(encoding.encode(name_desc))

            # Parameters
            params = func.get("parameters", {})
            props = params.get("properties", {})
            for p_name, p_info in props.items():
                p_info: dict[str, t.Any] = p_info or {}
                total_tokens += 5  # Parameter structure overhead
                p_type = p_info.get("type", "")
                p_desc = p_info.get("description", "")
                total_tokens += len(encoding.encode(f"{p_name}:{p_type}:{p_desc}"))

            required = params.get("required", [])
            if required:
                total_tokens += len(encoding.encode(" ".join(required)))

            return total_tokens

        except Exception as e:
            log.warning(
                msg="Failed to Count Tokens for Tool Call",
                model=self.model,
                tokenizer=self.model_config.tokenizer_base if self.model_config else None,
                error=str(e),
            )
            return fall_back_estimate

    # estimate_usage_cost is inherited from base (concrete implementation)
    # For local Ollama models, pricing is typically None → returns 0.0

    # -------- COMPLETION METHODS -----------------------------------------------------------
    async def complete(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]] | None = None,
        output_format: t.Type[BaseModel] | None = None,
        **kwargs: t.Any,
    ) -> ChatCompletionResult:
        """
        Make a single (non-streaming) Ollama API call.

        Args:
            messages: Pre-formatted list of API messages (already through build_api_messages).
            tools: Optional tool definitions in Ollama format.
            output_format: Optional Pydantic model for structured output.
            **kwargs: Extra parameters (temperature, top_p, top_k, num_ctx, etc.) passed
                      into the Ollama `options` field.

        Returns:
            ChatCompletionResult with message, usage, and optional structured output.

        Raises:
            AuthenticationError: If auth headers are invalid.
            ProviderError: For any other API error.
        """
        try:
            start_time = time.monotonic()
            request = self._build_request(messages, tools, output_format, **kwargs)
            response: ChatResponse = await self.client.chat(**request)
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return self._parse_response(response, output_format, duration_ms)

        except ResponseError as e:
            self._raise_from_response_error(e)
        except Exception as e:
            raise ProviderError(f"Unexpected error during Ollama API call: {str(e)}")

    async def stream(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]] | None = None,
        output_format: t.Type[BaseModel] | None = None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[ChatCompletionChunk, None]:
        """
        Make a streaming Ollama API call.

        Args:
            messages: Pre-formatted list of API messages.
            tools: Optional tool definitions.
            output_format: Optional Pydantic model for structured output.
            **kwargs: Extra parameters for the Ollama `options` field.

        Yields:
            ChatCompletionChunk — content chunks, tool call chunks, and final usage chunk.
        """
        if output_format:
            log.warning(
                "Structured output parsing in streaming yields partial JSON",
                action="Full structured output will be validated in the final chunk",
                output_format=output_format.__name__,
            )

        try:
            start_time = time.monotonic()
            request = self._build_request(messages, tools, output_format, **kwargs)
            request["stream"] = True

            raw_stream = await self.client.chat(**request)
            async for chunk in self._iter_chunks(raw_stream, output_format, start_time):
                yield chunk

        except ResponseError as e:
            self._raise_from_response_error(e)
        except Exception as e:
            raise ProviderError(
                f"Unexpected error during Ollama streaming call: {str(e)}"
            )

    # -------- CONVENIENCE METHODS -----------------------------------------------------------
    def _build_request(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]] | None = None,
        output_format: t.Type[BaseModel] | None = None,
        **kwargs: t.Any,
    ) -> dict[str, t.Any]:
        """
        Build request parameters for Ollama chat endpoint.

        Ollama separates API-level params from model-level options:
            - Top-level: model, messages, tools, format, think, keep_alive, stream
            - options dict: temperature, top_p, top_k, num_ctx, num_predict, seed, etc.

        Args:
            messages: Pre-formatted API messages.
            tools: Optional tool definitions.
            output_format: Optional Pydantic model for structured output.
            **kwargs: Model options routed into the `options` dict.

        Returns:
            Request params dict ready for client.chat().
        """
        # Separate top-level params from model options
        top_level_keys = {"think", "keep_alive", "format"}
        options: dict[str, t.Any] = {}
        overrides: dict[str, t.Any] = {}

        for k, v in kwargs.items():
            if k in top_level_keys:
                overrides[k] = v
            else:
                options[k] = v

        # Route max_tokens → num_predict (Ollama's equivalent)
        if "max_tokens" in options:
            options["num_predict"] = options.pop("max_tokens")

        request: dict[str, t.Any] = {
            "model": self.resolve_model_name(),
            "messages": messages,
        }

        if options:
            request["options"] = options

        if tools:
            request["tools"] = tools

        # Thinking mode (None = model default)
        think_value = overrides.get("think", self.think)
        if think_value is not None:
            request["think"] = think_value

        # Keep alive
        keep_alive_value = overrides.get("keep_alive", self.keep_alive)
        if keep_alive_value is not None:
            request["keep_alive"] = keep_alive_value

        # Structured output — Ollama uses `format` with a JSON schema directly
        if output_format:
            try:
                schema = output_format.model_json_schema()
                request["format"] = self.adapt_schema_for_provider(schema)
            except Exception as e:
                log.error(
                    msg=f"Failed to convert output_format to JSON schema: {e}",
                    action="Building request without structured output",
                )

        return request

    def _parse_tool_calls(self, raw_tool_calls: list[t.Any]) -> list[ToolCallRecord]:
        """
        Parse Ollama tool calls into ToolCallRecord objects.

        Ollama tool calls structure:
            tool_calls: [{
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "Taipei"}  # dict, NOT JSON string
                }
            }]

        Note: Ollama does NOT provide tool_call_id — we generate synthetic ones.
        """
        tool_calls: list[ToolCallRecord] = []
        for idx, tc in enumerate(raw_tool_calls):
            func = getattr(tc, "function", None) or tc.get("function", {})
            name = getattr(func, "name", None) or func.get("name", "")
            raw_args = getattr(func, "arguments", None) or func.get("arguments", {})

            # Ollama usually gives a dict already, but handle string defensively
            if isinstance(raw_args, str):
                try:
                    params: dict[str, t.Any] = json.loads(raw_args)
                except json.JSONDecodeError:
                    log.error(
                        "Failed to parse tool call arguments as JSON",
                        function_name=name,
                        raw_arguments=raw_args,
                    )
                    params = {"raw_error_content": raw_args, "parsing_error": True}
            elif isinstance(raw_args, dict):
                params = raw_args  # type: ignore
            else:
                params = {}

            # Ollama doesn't provide call IDs — synthesize one
            synthetic_id = f"ollama_{self.model}_{idx}_{int(time.time() * 1000)}"

            tool_calls.append(
                ToolCallRecord(
                    tool_name=name,
                    parameters=params,
                    tool_call_id=synthetic_id,
                )
            )
        return tool_calls

    def _parse_response(
        self,
        response: ChatResponse,
        output_format: t.Type[BaseModel] | None = None,
        duration_ms: int = 0,
    ) -> ChatCompletionResult:
        """
        Parse a non-streaming Ollama response into ChatCompletionResult.

        Args:
            response: Raw ChatResponse from Ollama.
            output_format: Optional Pydantic model for structured output.
            duration_ms: Total elapsed time in milliseconds.

        Returns:
            Normalized ChatCompletionResult.
        """
        if not response or not response.message:
            raise ProviderError(
                f"{self.__class__.__name__} received an empty response from Ollama."
            )

        message = response.message
        content: str = message.content or ""

        # Parse tool calls
        raw_tool_calls = getattr(message, "tool_calls", None)
        tool_calls = self._parse_tool_calls(raw_tool_calls) if raw_tool_calls else []

        # Parse structured output
        structured_output: t.Optional[BaseModel] = None
        if output_format and content:
            try:
                structured_output = output_format.model_validate_json(content)
            except Exception as e:
                log.error(
                    msg=f"Failed to parse structured output: {e}",
                    next_action="Returning response without structured output",
                )

        # Normalize usage — Ollama returns usage fields on the ChatResponse itself
        normalized = self.normalize_usage_stats(response)
        usage = Usage(
            duration_ms=duration_ms or normalized.duration_ms,
            llm_calls=1,
            tokens_input=normalized.tokens_input,
            tokens_output=normalized.tokens_output,
            tokens_cached=normalized.tokens_cached,
            tool_calls=len(tool_calls),
            cost_estimate=self.estimate_usage_cost(
                tokens_input=normalized.tokens_input,
                tokens_output=normalized.tokens_output,
                tokens_cached=normalized.tokens_cached,
            ),
        )

        # Capture thinking content for reasoning models (Qwen3-thinking, DeepSeek-R1, etc.)
        thinking_content: str | None = getattr(message, "thinking", None) or None

        return ChatCompletionResult(
            message=AssistantMessage(
                content=content,
                source="llm",
                thinking=thinking_content,
                tool_calls=tool_calls if tool_calls else None,
                structured_content=structured_output if structured_output else None,
            ),
            usage=usage,
            model=response.model or self.model,
            finish_reason=response.done_reason or "stop",
        )

    async def _iter_chunks(
        self,
        raw_stream: t.AsyncIterator[ChatResponse],
        output_format: t.Type[BaseModel] | None = None,
        start_time: float | None = None,
    ) -> t.AsyncGenerator[ChatCompletionChunk, None]:
        """
        Iterate over raw Ollama stream and yield normalized ChatCompletionChunks.

        Ollama streaming specifics:
            - Each chunk has `message.content` with incremental text
            - Tool calls arrive COMPLETE in a single chunk (not fragmented like OpenAI)
            - The final chunk has `done=True` and carries usage fields at the top level
            - `thinking` content may also stream for reasoning models

        Yields:
            ChatCompletionChunk — content chunks, tool call chunks, and final usage chunk.
        """
        accumulated_content: list[str] = []
        accumulated_thinking: list[str] = []
        tool_call_chunks: dict[str, t.Any] = {}
        tool_call_count = 0

        async for chunk in raw_stream:
            message = getattr(chunk, "message", None)

            # Thinking streaming (reasoning models like Qwen3-thinking, DeepSeek-R1)
            # Emitted as a separate chunk so frontends can render it distinctly
            if message:
                thinking_delta = getattr(message, "thinking", None)
                if thinking_delta:
                    accumulated_thinking.append(thinking_delta)
                    yield ChatCompletionChunk(
                        content="",
                        thinking=thinking_delta,
                        is_complete=False,
                        tool_call_chunk=None,
                    )

            # Content streaming
            if message and message.content:
                accumulated_content.append(message.content)
                yield ChatCompletionChunk(
                    content=message.content,
                    thinking=None,
                    is_complete=False,
                    tool_call_chunk=None,
                )

            # Tool calls — in Ollama they usually arrive complete in one chunk
            if message and getattr(message, "tool_calls", None):
                for idx, tc in enumerate(message.tool_calls):
                    func = getattr(tc, "function", None) or tc.get("function", {})
                    name = getattr(func, "name", None) or func.get("name", "")
                    args = getattr(func, "arguments", None) or func.get("arguments", {})

                    tracking_key = f"{tool_call_count}_{idx}"

                    # Validate JSON (args are usually dict already)
                    is_valid = True
                    if isinstance(args, str):
                        try:
                            json.loads(args)
                        except json.JSONDecodeError:
                            is_valid = False

                    tool_call_chunks[tracking_key] = {
                        "id": f"ollama_{self.model}_{tracking_key}",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args) if isinstance(args, dict) else args,
                        },
                        "is_valid_json": is_valid,
                    }
                    tool_call_count += 1

                    yield ChatCompletionChunk(
                        content="",
                        is_complete=False,
                        tool_call_chunk=tool_call_chunks[tracking_key],
                    )

            # Final chunk — Ollama marks it with done=True and includes usage
            if getattr(chunk, "done", False):
                structured_output: BaseModel | None = None
                if output_format and accumulated_content:
                    try:
                        structured_output = output_format.model_validate_json(
                            "".join(accumulated_content)
                        )
                    except Exception as e:
                        log.error(
                            msg=f"Failed to parse final structured output: {e}",
                            action="Yielding final chunk without structured output",
                        )

                normalized = self.normalize_usage_stats(chunk)
                duration_ms = (
                    int((time.monotonic() - start_time) * 1000)
                    if start_time
                    else normalized.duration_ms
                )
                yield ChatCompletionChunk(
                    content="",
                    thinking=None,
                    is_complete=True,
                    tool_call_chunk=None,
                    usage=Usage(
                        duration_ms=duration_ms,
                        llm_calls=1,
                        tokens_input=normalized.tokens_input,
                        tokens_output=normalized.tokens_output,
                        tokens_cached=normalized.tokens_cached,
                        tool_calls=len(tool_call_chunks),
                        cost_estimate=self.estimate_usage_cost(
                            tokens_input=normalized.tokens_input,
                            tokens_output=normalized.tokens_output,
                            tokens_cached=normalized.tokens_cached,
                        ),
                    ),
                    structured_output=structured_output if output_format else None,
                )
                return

    def _raise_from_response_error(self, e: ResponseError) -> t.NoReturn:
        """
        Map Ollama ResponseError to our exception hierarchy based on status_code.
        """
        status = getattr(e, "status_code", None)
        msg = str(e)

        if status == 401 or status == 403:
            raise AuthenticationError(f"Ollama authentication failed: {msg}")
        if status == 429:
            raise RateLimitError(f"Ollama rate limit exceeded: {msg}")
        if status == 400 or status == 404:
            # 404 commonly means "model not found / not pulled"
            raise InvalidRequestError(f"Ollama invalid request: {msg}")

        raise ProviderError(f"Ollama API error (status={status}): {msg}")

    # -------- SERIALIZATION -----------------------------------------------------------
    def _to_config(self) -> OllamaChatCompletionClientConfig:
        """Convert client to configuration for serialization."""
        return OllamaChatCompletionClientConfig(
            model=self.model,
            host=self.host,
            api_key=self.api_key,
            think=self.think,
            keep_alive=self.keep_alive,
            config=self.config,
        )

    @classmethod
    def _from_config(
        cls, config: OllamaChatCompletionClientConfig
    ) -> "OllamaChatCompletionClient":
        """Create client from configuration."""
        return cls(
            model=config.model,
            host=config.host,
            api_key=config.api_key,
            think=config.think,
            keep_alive=config.keep_alive,
            **config.config,
        )
