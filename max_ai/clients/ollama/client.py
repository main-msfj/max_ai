"""
Ollama BaseChatCompletionClient Implementation.

Talks to a local Ollama server via the official ``ollama-python``
``AsyncClient``. Supports tool calling, thinking mode, streaming, and
structured outputs for models like Qwen3, Llama3, Mistral, Phi-4, and
Gemma3.
"""

from __future__ import annotations

import time
import json
import logging
import typing as t
from pydantic import BaseModel, SecretStr
from ollama import AsyncClient, ChatResponse, ResponseError

from ._schema import clean_json_schema

from ...loggers import ScopedLogger
from ...base.component_config import Component
from ...base.clients import CoreChatCompletionClient

from ...types.stacks import PromptCtx
from ...types.run_context import RunContext
from ...types.completions import ChatCompletionChunk, ChatCompletionResult, Usage


from ...core.models import ModelConfig, OllamaChatCompletionClientConfig
from ...core.messages import (
    CoreMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolMessage,
    TextPart,
    ImagePart,
    AudioPart,
    FilePart,
    MessageContent,
)
from ...errors.client import ClientError

if t.TYPE_CHECKING:
    from ...base.tools import CoreTool

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="OllamaChatCompletionClient")


class OllamaChatCompletionClient(
    Component[OllamaChatCompletionClientConfig],
    CoreChatCompletionClient,
):
    """Ollama implementation of ``CoreChatCompletionClient``.

    Wraps the official ``ollama-python`` ``AsyncClient`` for a local
    (or hosted) Ollama server. Supports the OpenAI-style function-call
    schema for tool use, native thinking mode, NDJSON streaming, and
    structured outputs via JSON-schema constrained decoding.
    """

    component_config_schema = OllamaChatCompletionClientConfig
    component_type = "client"
    component_provider_override = "maxai.llm.OllamaChatCompletionClient"

    DEFAULT_HOST = "http://localhost:11434"
    SYSTEM_LAYER_SEPARATOR = "\n\n"
    SYSTEM_SOURCE = "system"

    # -------- CONSTRUCTION -----------------------------------------------------------
    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        api_key: str | SecretStr | None = None,
        config: ModelConfig | None = None,
        think: bool | None = None,
        keep_alive: str | int | None = None,
        **kwargs: t.Any,
    ) -> None:
        """Initialize the Ollama client.

        Args:
            model: Provider-specific model identifier
                (e.g. ``"qwen3:4b-thinking-2507-q4_K_M"``,
                ``"llama3.1:8b"``).
            host: Ollama server URL. Defaults to ``http://localhost:11434``.
            api_key: Optional bearer token. Required only for hosted
                Ollama deployments or auth proxies. ``None`` for local.
            config: Model capabilities (``supports_function_calling``,
                ``supports_thinking``, ``thinking_tag``, etc.).
            think: Override the model's default thinking behavior.
                ``None`` defers to the model, ``True`` forces it on,
                ``False`` forces it off.
            keep_alive: Time the model should stay loaded in memory
                between calls (e.g. ``"5m"``). ``0`` unloads
                immediately after the request.
            **kwargs: Reserved for future provider-specific defaults
                (e.g. ``temperature``, ``top_p``).
        """
        super().__init__(model=model, api_key=api_key, config=config, **kwargs)
        self.host: str = self._require_type(host, str, "host")
        self.think: bool | None = think
        self.keep_alive: str | int | None = keep_alive

        headers: dict[str, str] | None = None
        if self.api_key is not None:
            headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}

        self.client: AsyncClient = AsyncClient(host=self.host, headers=headers)

    # -------- COMPONENT SERIALIZATION -----------------------------------------------------------
    def _to_config(self) -> OllamaChatCompletionClientConfig:
        """Serialize this client into its portable config form."""
        return OllamaChatCompletionClientConfig(
            model=self.model,
            host=self.host,
            api_key=self.api_key,
            think=self.think,
            keep_alive=self.keep_alive,
            config=self.config.model_dump(exclude_none=True),
        )

    @classmethod
    def _from_config(
        cls, config: OllamaChatCompletionClientConfig
    ) -> "OllamaChatCompletionClient":
        """Rebuild a client instance from a config payload."""
        model_config = ModelConfig(**config.config) if config.config else None
        return cls(
            model=config.model,
            host=config.host,
            api_key=config.api_key,
            config=model_config,
            think=config.think,
            keep_alive=config.keep_alive,
        )

    # -------- TOOL SCHEMA -----------------------------------------------------------
    def build_tool_schema(self, tools: list["CoreTool"]) -> list[dict[str, t.Any]]:
        """Convert ``CoreTool`` instances to Ollama's tool schema.

        Ollama accepts the OpenAI-compatible function-call envelope::

            {
              "type": "function",
              "function": {
                "name": "search_docs",
                "description": "Search the docs index.",
                "parameters": { ... JSON Schema ... }
              }
            }

        ``parameters`` comes from ``CoreTool.parameters`` (a
        pydantic-generated JSON Schema) and is run through
        ``clean_json_schema`` to strip metadata that confuses local
        models — titles, ``additionalProperties``, etc.

        Args:
            tools: The ``CoreTool`` list passed to ``run()``. May be
                empty.

        Returns:
            A list of Ollama-native tool dicts. Empty when ``tools``
            is empty.
        """
        if not tools:
            return []
        return [self._tool_to_ollama_schema(tool) for tool in tools]

    def _tool_to_ollama_schema(self, tool: "CoreTool") -> dict[str, t.Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": clean_json_schema(tool.parameters),
            },
        }

    # -------- USAGE -----------------------------------------------------------
    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        """Normalize Ollama's usage payload into framework ``Usage``.

        Ollama exposes token counts directly on the response object:

        - ``prompt_eval_count``: input tokens processed.
        - ``eval_count``: output tokens generated.
        - ``total_duration``: nanoseconds for the whole request.
        - ``load_duration``, ``prompt_eval_duration``, ``eval_duration``:
          finer-grained timings (currently unused).

        Ollama does not expose cached tokens — ``tokens_cached`` is
        always 0.

        Args:
            usage: A ``ChatResponse`` (or any object exposing the
                ``*_count`` / ``*_duration`` attributes).

        Returns:
            A populated ``Usage`` instance.
        """
        prompt_tokens = int(getattr(usage, "prompt_eval_count", 0) or 0)
        completion_tokens = int(getattr(usage, "eval_count", 0) or 0)
        total_ns = int(getattr(usage, "total_duration", 0) or 0)
        duration_ms = total_ns // 1_000_000

        return Usage(
            duration_ms=duration_ms,
            tokens_input=prompt_tokens,
            tokens_output=completion_tokens,
            tokens_cached=0,
        )

    # -------- NOT YET IMPLEMENTED -----------------------------------------------------------
    def format_messages(self, ctx: RunContext, prompts: PromptCtx) -> list[CoreMessage]:
        """Assemble the full message list for the Ollama call.

        Order:
        1. A single ``SystemMessage`` containing every rendered prompt
            layer concatenated with two newlines, in stack order. Empty /
            whitespace-only layers are skipped. If every layer renders
            empty (very unlikely — ``AgentPolicy`` is always present), the
            system message is omitted entirely.
        2. Past conversation messages from ``ctx.message_history``, in
            chronological order.
        3. New messages produced this turn (``ctx.messages``) — typically
            the current user input.

        The system message carries ``source="system"`` so a serialized run
        is portable across providers: the same ``CoreMessage`` list works
        whether the next call goes to Ollama, Anthropic, or OpenAI.

        Args:
            ctx: Per-turn runtime state.
            prompts: Rendered prompt layers from ``agent.prepare()``.

        Returns:
            Provider-agnostic ``list[CoreMessage]`` ready for
            ``build_api_messages``.
        """
        messages: list[CoreMessage] = []

        system_content = self._build_system_content(prompts)
        if system_content:
            messages.append(
                SystemMessage(source=self.SYSTEM_SOURCE, content=system_content)
            )

        messages.extend(ctx.message_history.iter_messages())
        messages.extend(ctx.messages)

        return messages

    def _build_system_content(self, prompts: PromptCtx) -> str:
        """Concatenate non-empty rendered layers with a blank-line separator.

        If the model declares a ``force_no_thinking_marker`` and the
        caller has disabled thinking via ``supports_thinking=False``, the
        marker is appended after the rendered layers. This handles models
        like qwen3-thinking that emit reasoning regardless of the API's
        ``think`` flag — the marker is the in-prompt instruction the
        model was fine-tuned to obey.
        """
        chunks: list[str] = []
        for rendered in prompts.rendered_layers.values():
            if rendered and rendered.strip():
                chunks.append(rendered.strip())

        if (
            not self.config.supports_thinking
            and self.config.force_no_thinking_marker
        ):
            chunks.append(self.config.force_no_thinking_marker.strip())

        return self.SYSTEM_LAYER_SEPARATOR.join(chunks)

    # -------- API MESSAGE BUILDING -----------------------------------------------------------
    def build_api_messages(self, messages: list[CoreMessage]) -> list[dict[str, t.Any]]:
        """Convert each ``CoreMessage`` into Ollama's wire format.

        Ollama's ``/api/chat`` accepts the OpenAI-flavored shape with two
        twists for media:

        - Images are passed as a top-level ``images`` field, as a list of
        raw base64 strings (no ``data:`` URL prefix).
        - Audio is passed as a top-level ``audio`` field, same convention,
        for models that declare ``supports_audio``.

        Capability gating is driven entirely by ``self.config``. If a
        message contains a part the model has not declared support for,
        a ``ClientError.unsupported_feature`` is raised — the framework
        does no fallback, no extraction, no transcoding. That belongs in
        the UI / pre-processing layer.

        Args:
            messages: Output of ``format_messages``.

        Returns:
            A list of dicts ready to ship to ``ollama.AsyncClient.chat``.

        Raises:
            ClientError: If a content part is not supported by the
                current model's declared capabilities, or if an unknown
                message type is encountered.
        """
        return [self._message_to_dict(m) for m in messages]

    def _message_to_dict(self, msg: CoreMessage) -> dict[str, t.Any]:
        """Dispatch by concrete message type."""
        if isinstance(msg, SystemMessage):
            return self._system_to_dict(msg)
        if isinstance(msg, UserMessage):
            return self._user_to_dict(msg)
        if isinstance(msg, AssistantMessage):
            return self._assistant_to_dict(msg)
        if isinstance(msg, ToolMessage):
            return self._tool_to_dict(msg)
        raise ClientError.unsupported_feature(
            feature=f"message type {type(msg).__name__}",
            model=self.model,
        )

    # -------- PER-ROLE BUILDERS -----------------------------------------------------------
    def _system_to_dict(self, msg: SystemMessage) -> dict[str, t.Any]:
        """System messages are always plain text."""
        return {"role": "system", "content": msg.text()}

    def _user_to_dict(self, msg: UserMessage) -> dict[str, t.Any]:
        """User messages may carry text, images, and audio."""
        text, images, audio = self._split_parts(msg.content)
        out: dict[str, t.Any] = {"role": "user", "content": text}
        if images:
            out["images"] = images
        if audio:
            out["audio"] = audio
        return out

    def _assistant_to_dict(self, msg: AssistantMessage) -> dict[str, t.Any]:
        """Assistant messages may carry tool calls. Multimodal output is
        not part of Ollama's response surface, so we serialize text only.
        """
        out: dict[str, t.Any] = {"role": "assistant", "content": msg.text()}
        if msg.tool_calls:
            out["tool_calls"] = [
                {
                    "function": {
                        "name": tc.tool_name,
                        "arguments": tc.parameters,
                    }
                }
                for tc in msg.tool_calls
            ]
        return out

    def _tool_to_dict(self, msg: ToolMessage) -> dict[str, t.Any]:
        """Tool result messages.

        Ollama accepts ``tool_name`` and (optionally) ``tool_call_id``.
        Some models use one, some the other — we send both when we have
        them so any reasonable model can correlate the result.
        """
        out: dict[str, t.Any] = {
            "role": "tool",
            "content": msg.text(),
            "tool_name": msg.tool_name,
        }
        if msg.tool_call_id:
            out["tool_call_id"] = msg.tool_call_id
        return out

    # -------- CONTENT PART SPLITTING -----------------------------------------------------------
    def _split_parts(self, content: MessageContent) -> tuple[str, list[str], list[str]]:
        """Split a message's content into ``(text, images_b64, audio_b64)``.

        The text components are concatenated in order. Images and audio
        are returned as base64 strings without any ``data:`` URL prefix.

        Capability checks:
        - ``ImagePart`` requires ``config.supports_vision``.
        - ``AudioPart`` requires ``config.supports_audio``.
        - ``FilePart`` is never accepted — convert PDFs to text or
            images upstream before reaching the framework.

        Image and audio parts must carry inline ``data`` (bytes). URL
        references are rejected because Ollama does not fetch remote
        media; the caller must download the bytes itself.
        """
        if isinstance(content, str):
            return content, [], []

        text_chunks: list[str] = []
        images: list[str] = []
        audio: list[str] = []

        for part in content:
            if isinstance(part, TextPart):
                text_chunks.append(part.text)

            elif isinstance(part, ImagePart):
                if not self.config.supports_vision:
                    raise ClientError.unsupported_feature(
                        feature="vision",
                        model=self.model,
                    )
                images.append(self._media_to_b64(part, "image"))

            elif isinstance(part, AudioPart):
                if not self.config.supports_audio:
                    raise ClientError.unsupported_feature(
                        feature="audio",
                        model=self.model,
                    )
                audio.append(self._media_to_b64(part, "audio"))

            elif isinstance(part, FilePart):
                raise ClientError.unsupported_feature(
                    feature="file attachments",
                    model=self.model,
                )

            else:
                raise ClientError.unsupported_feature(
                    feature=f"content part {type(part).__name__}",
                    model=self.model,
                )

        return "".join(text_chunks), images, audio

    def _media_to_b64(self, part: ImagePart | AudioPart, kind: str) -> str:
        """Return raw base64 for a media part. Rejects URL-only parts."""
        if part.data is not None:
            b64 = part.to_base64()
            if b64 is None:
                raise ClientError.unsupported_feature(
                    feature=f"{kind} with empty data",
                    model=self.model,
                )
            return b64
        raise ClientError.unsupported_feature(
            feature=f"{kind} by URL",
            model=self.model,
        )

    def _build_request(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]] | None,
        output_format: t.Type[BaseModel] | None,
        stream: bool,
        **kwargs: t.Any,
    ) -> dict[str, t.Any]:
        """Build the kwargs dict for ``AsyncClient.chat``.

        Shared between ``complete`` and ``stream`` so request semantics
        stay identical except for the ``stream`` flag.

        The ``options`` dict carries generation-time knobs (``temperature``,
        ``top_p``, ``top_k``, ``num_ctx``, ``num_predict``, ``seed``,
        ``stop``, ``repeat_penalty``). We pass any unrecognised kwargs
        straight through — Ollama silently ignores unknown options, and
        that's the best we can do without maintaining a hardcoded
        whitelist that goes stale every time Ollama adds a knob.
        """
        request: dict[str, t.Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }

        if tools:
            request["tools"] = tools

        # Per-call think override > client-level default. Ollama accepts
        # ``think=True/False`` only when the model declares thinking; we
        # forward whichever was set most recently.
        think = kwargs.pop("think", self.think)
        if think is not None:
            request["think"] = think

        keep_alive = kwargs.pop("keep_alive", self.keep_alive)
        if keep_alive is not None:
            request["keep_alive"] = keep_alive

        if output_format is not None:
            request["format"] = output_format.model_json_schema()

        if kwargs:
            request["options"] = kwargs
        return request


    def _split_thinking(self, raw: str) -> tuple[str | None, str]:
        """Split a ``<{thinking_tag}>...</{thinking_tag}>`` block from content.

        Two strategies driven by ``ModelConfig.thinking_position``:

        - ``"start"`` (default): only honour an opening tag at the very
        start of the content (after stripping leading whitespace).
        Tags appearing later are part of the assistant's output. This
        matches qwen3, deepseek-r1, claude extended thinking, and
        every thinking model we know of today.
        - ``"anywhere"``: take the first ``<tag>...</tag>`` block found,
        regardless of position. Reserved for hypothetical models that
        interleave reasoning with output.

        Returns ``(thinking, content)``. If thinking is disabled, the
        content does not contain the tag, or the tag is malformed, returns
        ``(None, raw)``.
        """
        if not self.config.supports_thinking:
            return None, raw

        tag = self.config.thinking_tag
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"

        if self.config.thinking_position == "start":
            if not raw.lstrip().startswith(open_tag):
                return None, raw
            open_idx = raw.find(open_tag)
        else:  # "anywhere"
            open_idx = raw.find(open_tag)
            if open_idx == -1:
                return None, raw

        close_idx = raw.find(close_tag, open_idx + len(open_tag))
        if close_idx == -1:
            # Opening without closing — treat the tail as thinking, content empty.
            thinking = raw[open_idx + len(open_tag) :].strip()
            return (thinking or None), ""

        thinking = raw[open_idx + len(open_tag) : close_idx].strip()
        before = raw[:open_idx]
        after = raw[close_idx + len(close_tag) :]
        content = (before + after).strip()
        return (thinking or None), content

    def _parse_tool_calls(self, raw_tool_calls: t.Sequence[t.Any]) -> list[ToolCall]:
        """Convert Ollama tool calls into ``ToolCall`` instances.

        Ollama gives each tool call as ``{"function": {"name": str,
        "arguments": dict | str}}``. Most modern models produce ``arguments``
        as a dict, but a few older / smaller models still emit a JSON
        string — handle both.

        Ollama does not assign an id to tool calls. We synthesize one so
        downstream code (history, approval flow, retry, etc.) has a stable
        handle.
        """
        parsed: list[ToolCall] = []
        for tc in raw_tool_calls:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue

            name = getattr(fn, "name", None)
            if not name:
                continue

            raw_args = getattr(fn, "arguments", {}) or {}
            if isinstance(raw_args, str):
                try:
                    params = json.loads(raw_args)
                except json.JSONDecodeError:
                    log.warning(
                        "Tool call arguments were a non-JSON string; "
                        "passing through as raw_error_content.",
                    )
                    params: dict[str, t.Any] = {
                        "raw_error_content": raw_args,
                        "parsing_error": True,
                    }
            elif isinstance(raw_args, dict):
                params = raw_args  # type: ignore
            else:
                params = {}

            parsed.append(ToolCall(tool_name=name, parameters=params))
        return parsed
    
    def _parse_response(
        self,
        response: "ChatResponse",
        output_format: t.Type[BaseModel] | None,
        duration_ms: int,
    ) -> ChatCompletionResult:
        """Convert an Ollama ``ChatResponse`` into a ``ChatCompletionResult``."""

        raw_content: str = response.message.content or ""

        # Ollama (recent versions) returns reasoning in a dedicated
        # ``thinking`` field on the message when ``think=True`` is set on
        # the request — content is already clean. Older models or builds
        # may still embed ``<tag>...</tag>`` inline, so we keep the
        # in-band parser as a fallback.
        native_thinking = getattr(response.message, "thinking", None)
        if native_thinking:
            thinking = native_thinking
            content = raw_content
        else:
            thinking, content = self._split_thinking(raw_content)

        tool_calls = self._parse_tool_calls(response.message.tool_calls or [])

        structured_output: BaseModel | None = None
        if output_format is not None and content:
            try:
                structured_output = output_format.model_validate_json(content)
            except Exception:
                log.warning(
                    "Failed to parse structured output for %s",
                    output_format_name=output_format.__name__,
                )

        base_usage = self.normalize_usage_stats(response)
        usage = Usage(
            duration_ms=duration_ms,
            llm_calls=1,
            tokens_input=base_usage.tokens_input,
            tokens_output=base_usage.tokens_output,
            tokens_cached=0,
            tool_calls=len(tool_calls),
        )

        return ChatCompletionResult(
            message=AssistantMessage(
                source="llm",
                content=content,
                thinking=thinking,
                tool_calls=tool_calls,
                structured_output=structured_output,
            ),
            usage=usage,
            model=response.model or self.model,
            finish_reason=response.done_reason or "stop",
        )

    # -------- API CALL — COMPLETE -----------------------------------------------------------
    async def complete(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]] | None,
        output_format: t.Type[BaseModel] | None,
        **kwargs: t.Any,
    ) -> ChatCompletionResult:
        """Single (non-streaming) chat completion against Ollama.

        Args:
            messages: Provider-native message list (output of
                ``build_api_messages``).
            tools: Provider-native tool schemas (output of
                ``build_tool_schema``). May be empty or None.
            output_format: Optional Pydantic model. If provided, Ollama
                is asked to constrain output to its JSON Schema via the
                ``format`` field — supported by all recent models.
            **kwargs: Forwarded to ``options`` (temperature, top_p, etc.).

        Returns:
            Normalized ``ChatCompletionResult``.

        Raises:
            ClientError: For Ollama API errors and unexpected failures.
        """
        try:
            start_time = time.monotonic()
            request = self._build_request(
                messages=messages,
                tools=tools,
                output_format=output_format,
                stream=False,
                **kwargs,
            )
            response: ChatResponse = t.cast(
                ChatResponse, await self.client.chat(**request)
            )
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return self._parse_response(response, output_format, duration_ms)

        except ResponseError as e:
            raise ClientError.api_error(
                provider="ollama",
                status=getattr(e, "status_code", None),
                detail=str(e),
            ) from e
        except Exception as e:
            raise ClientError.unexpected(
                provider="ollama",
                detail=str(e),
            ) from e

    # -------- API CALL — STREAM -----------------------------------------------------------
    async def stream(
        self,
        messages: list[dict[str, t.Any]],
        tools: list[dict[str, t.Any]] | None,
        output_format: t.Type[BaseModel] | None,
        **kwargs: t.Any,
    ) -> t.AsyncGenerator[ChatCompletionChunk, None]:
        """Streaming chat completion against Ollama.

        Yields chunks as they arrive. The final chunk carries cumulative
        ``usage`` and ``is_complete=True``.

        Streaming behaviour notes:

        - Ollama does not fragment tool calls. A tool call appears in
        exactly one chunk, fully formed. We yield it in a single
        ``tool_call_chunk``.
        - When ``ModelConfig.supports_thinking`` is True, this method
        buffers content tokens until the closing thinking tag is seen.
        Tokens inside the tag stream out via ``thinking``; tokens after
        the tag stream out via ``content``. This means the consumer
        sees reasoning live, then output live, with no false starts.
        - Structured output is parsed once at the final chunk by
        validating the accumulated content. Streaming + structured
        output works, but the parsed Pydantic instance is only
        available on the last chunk.

        Args:
            messages: Output of ``build_api_messages``.
            tools: Output of ``build_tool_schema``.
            output_format: Optional Pydantic model.
            **kwargs: Forwarded to ``options``.

        Yields:
            ``ChatCompletionChunk`` instances.

        Raises:
            ClientError: For Ollama API errors and unexpected failures.
        """
        try:
            start_time = time.monotonic()
            request = self._build_request(
                messages=messages,
                tools=tools,
                output_format=output_format,
                stream=True,
                **kwargs,
            )

            raw_stream = await self.client.chat(**request)
            async for chunk in self._iter_chunks(raw_stream, output_format, start_time):
                yield chunk

        except ResponseError as e:
            raise ClientError.api_error(
                provider="ollama",
                status=getattr(e, "status_code", None),
                detail=str(e),
            ) from e
        except Exception as e:
            raise ClientError.unexpected(
                provider="ollama",
                detail=str(e),
            ) from e

    async def _iter_chunks(
        self,
        raw_stream: t.AsyncIterator["ChatResponse"],
        output_format: t.Type[BaseModel] | None,
        start_time: float,
    ) -> t.AsyncGenerator[ChatCompletionChunk, None]:
        accumulated_content: list[str] = []
        thinking_buf = _StreamThinkingBuffer(self.config)
        tool_call_count = 0
        final_response: "ChatResponse | None" = None

        async for response in raw_stream:
            if getattr(response, "done", False):
                final_response = response
                for thinking_delta, content_delta in thinking_buf.flush():
                    if thinking_delta:
                        yield ChatCompletionChunk(
                            content="",
                            thinking=thinking_delta,
                            is_complete=False,
                        )
                    if content_delta:
                        accumulated_content.append(content_delta)
                        yield ChatCompletionChunk(
                            content=content_delta,
                            is_complete=False,
                        )
                break

            message = getattr(response, "message", None)
            if message is None:
                continue

            # Tool calls — emitted whole, not fragmented.
            raw_tool_calls = getattr(message, "tool_calls", None) or []
            for tc in self._parse_tool_calls(raw_tool_calls):
                tool_call_count += 1
                yield ChatCompletionChunk(
                    content="",
                    is_complete=False,
                    tool_call_chunk={
                        "id": tc.id,
                        "name": tc.tool_name,
                        "arguments": tc.parameters,
                    },
                )

            # Native thinking field — Ollama already split it for us. Emit
            # straight, no buffer needed.
            native_thinking = getattr(message, "thinking", None)
            if native_thinking:
                yield ChatCompletionChunk(
                    content="",
                    thinking=native_thinking,
                    is_complete=False,
                )

            # Content — only run through the thinking buffer if the model
            # is embedding tags inline. If we already saw native thinking
            # this turn (or any prior turn), trust the provider and emit
            # content straight.
            delta_text = getattr(message, "content", "") or ""
            if not delta_text:
                continue

            if native_thinking is not None or thinking_buf.bypassed:
                # Provider gave us pre-split content — bypass the buffer
                # for the rest of the stream.
                thinking_buf.bypass()
                accumulated_content.append(delta_text)
                yield ChatCompletionChunk(
                    content=delta_text,
                    is_complete=False,
                )
            else:
                for thinking_delta, content_delta in thinking_buf.feed(delta_text):
                    if thinking_delta:
                        yield ChatCompletionChunk(
                            content="",
                            thinking=thinking_delta,
                            is_complete=False,
                        )
                    if content_delta:
                        accumulated_content.append(content_delta)
                        yield ChatCompletionChunk(
                            content=content_delta,
                            is_complete=False,
                        )

        # Final chunk
        structured_output: BaseModel | None = None
        if output_format is not None and accumulated_content:
            try:
                structured_output = output_format.model_validate_json(
                    "".join(accumulated_content)
                )
            except Exception:
                log.warning(
                    "Failed to parse structured output for %s",
                    output_format.__name__,
                )

        duration_ms = int((time.monotonic() - start_time) * 1000)
        base_usage = (
            self.normalize_usage_stats(final_response)
            if final_response is not None
            else Usage()
        )
        usage = Usage(
            duration_ms=duration_ms,
            llm_calls=1,
            tokens_input=base_usage.tokens_input,
            tokens_output=base_usage.tokens_output,
            tokens_cached=0,
            tool_calls=tool_call_count,
        )

        yield ChatCompletionChunk(
            content="",
            is_complete=True,
            usage=usage,
            structured_output=structured_output,
        )


# -------- STREAMING THINKING BUFFER -----------------------------------------------------------
class _StreamThinkingBuffer:
    """Routes streaming tokens into ``thinking`` vs ``content`` deltas.

    The thinking tag arrives one token at a time. We can't classify a
    token as thinking-vs-content until we know whether we're inside a
    tag, so we buffer just enough characters to detect the boundary.

    State machine:
      - ``BEFORE``: have not seen the opening tag yet. Hold tokens
        until we either see the open tag (transition to INSIDE) or
        accumulate enough to know an open tag is impossible (flush
        as content, transition to AFTER).
      - ``INSIDE``: between open and close tag. Tokens stream out as
        ``thinking``. Hold a small tail of characters in case the
        close tag is split across feed calls.
      - ``AFTER``: closing tag was seen. Tokens stream straight as
        ``content``.

    For ``thinking_position == "start"`` only: any non-whitespace
    character before the open tag means thinking is not happening,
    so we shortcut to AFTER and emit as content.

    For ``thinking_position == "anywhere"``: BEFORE keeps buffering
    until we either see the tag or hit EOF.

    If thinking is not enabled (``supports_thinking == False``), the
    buffer is a passthrough: ``feed`` returns ``[(None, delta)]``.
    """

    _BEFORE = "before"
    _INSIDE = "inside"
    _AFTER = "after"

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.open_tag = f"<{config.thinking_tag}>"
        self.close_tag = f"</{config.thinking_tag}>"
        self.state = self._AFTER if not config.supports_thinking else self._BEFORE
        self.buffer = ""
        self.bypassed = False

    def bypass(self) -> None:
        """Disable in-band tag parsing. Use when the provider emits
        thinking and content as separate fields (Ollama native mode).
        """
        self.bypassed = True
        self.state = self._AFTER
        self.buffer = ""

    def feed(self, delta: str) -> list[tuple[str | None, str | None]]:
        """Feed a streamed token. Yields ``(thinking_delta, content_delta)`` pairs.

        Either field can be None if there is nothing to emit for it
        in this token. Both can be None on a token that is fully
        absorbed into the buffer (waiting for more context).
        """
        if not delta:
            return []
        if self.state == self._AFTER:
            return [(None, delta)]

        self.buffer += delta
        return self._drain()

    def flush(self) -> list[tuple[str | None, str | None]]:
        """Drain remaining buffered content at end of stream."""
        out: list[tuple[str | None, str | None]] = []
        if not self.buffer:
            return out

        if self.state == self._INSIDE:
            # Stream ended inside the tag — emit as thinking.
            out.append((self.buffer, None))
        else:
            # BEFORE without ever seeing the tag — emit as content.
            out.append((None, self.buffer))
        self.buffer = ""
        return out

    def _drain(self) -> list[tuple[str | None, str | None]]:
        out: list[tuple[str | None, str | None]] = []

        while True:
            if self.state == self._BEFORE:
                emitted = self._drain_before(out)
            elif self.state == self._INSIDE:
                emitted = self._drain_inside(out)
            else:
                # AFTER: flush remaining buffer as content.
                if self.buffer:
                    out.append((None, self.buffer))
                    self.buffer = ""
                emitted = False

            if not emitted:
                break

        return out

    def _drain_before(self, out: list[tuple[str | None, str | None]]) -> bool:
        """Look for the opening tag. Return True if state advanced."""
        idx = self.buffer.find(self.open_tag)
        if idx >= 0:
            # Anything before the tag is content (only possible in "anywhere"
            # mode, since "start" rejects pre-tag text below).
            preamble = self.buffer[:idx]
            if preamble:
                out.append((None, preamble))
            self.buffer = self.buffer[idx + len(self.open_tag) :]
            self.state = self._INSIDE
            return True

        # Tag not found yet. Decide whether to keep buffering.
        if self.config.thinking_position == "start":
            # If the buffer holds any non-whitespace that can't be a tag prefix,
            # we know the tag will never come — bail out.
            stripped = self.buffer.lstrip()
            if stripped and not self.open_tag.startswith(stripped):
                out.append((None, self.buffer))
                self.buffer = ""
                self.state = self._AFTER
                return True

        # Could still become a tag — hold all but the safe prefix.
        # The safe prefix is everything except the last (len(open_tag)-1) chars.
        keep = max(0, len(self.buffer) - (len(self.open_tag) - 1))
        if keep > 0 and self.config.thinking_position == "anywhere":
            # In "anywhere" mode, content before a not-yet-seen tag is still
            # valid content — emit the safe prefix.
            out.append((None, self.buffer[:keep]))
            self.buffer = self.buffer[keep:]
            return bool(keep)

        return False

    def _drain_inside(self, out: list[tuple[str | None, str | None]]) -> bool:
        """Look for the closing tag. Return True if state advanced."""
        idx = self.buffer.find(self.close_tag)
        if idx >= 0:
            thinking_part = self.buffer[:idx]
            if thinking_part:
                out.append((thinking_part, None))
            self.buffer = self.buffer[idx + len(self.close_tag) :]
            self.state = self._AFTER
            return True

        # Closing tag not found — emit the safe prefix as thinking,
        # hold the tail in case the tag straddles future tokens.
        keep = max(0, len(self.buffer) - (len(self.close_tag) - 1))
        if keep > 0:
            out.append((self.buffer[:keep], None))
            self.buffer = self.buffer[keep:]
            return True

        return False
