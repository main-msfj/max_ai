"""
Core message types for agent communication using Pydantic models.

Design principles:
- Messages follow the standard role taxonomy: system / user / assistant / tool.
  There is NO "multimodal" role — multi-modality lives inside `content`.
- `content` can be a plain string (simple case) or a list of ContentPart
  (multi-modal case). This mirrors how OpenAI, Anthropic and Gemini all work.
- ContentPart is a discriminated union (text/image/audio/file), so Pydantic
  validates and serializes each kind correctly.
- Each concrete message class declares its `role` as a `Literal[...]`,
  letting Pydantic deserialize the `Message` union into the correct
  subtype natively.
- This file is the INTERNAL representation only. Provider-specific formats
  (OpenAI, Anthropic, etc.) live in separate adapter functions.
"""

from __future__ import annotations

import uuid
import base64
import typing as t
from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    TypeAdapter,
    model_validator,
)


# -------- CONTENT PARTS -----------------------------------------------------------
# A message's `content` is either a plain string (text only) or a list of
# these parts. Each part is a discriminated union on the `type` field.


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """An image. Provide EITHER a url OR raw bytes in `data`."""

    type: Literal["image"] = "image"
    url: str | None = None
    data: bytes | None = None
    mime_type: str = Field(
        default="image/png", description="e.g. image/png, image/jpeg"
    )

    @model_validator(mode="after")
    def _one_source(self):
        if (self.url is None) == (self.data is None):
            raise ValueError("ImagePart requires exactly one of: url or data")
        return self

    def to_base64(self) -> str | None:
        if self.data is None:
            return None
        return base64.b64encode(self.data).decode("utf-8")


class AudioPart(BaseModel):
    """Audio clip (e.g. mp3, wav). NOTE: not every provider supports audio."""

    type: Literal["audio"] = "audio"
    url: str | None = None
    data: bytes | None = None
    mime_type: str = Field(
        default="audio/mpeg", description="e.g. audio/mpeg, audio/wav"
    )

    @model_validator(mode="after")
    def _one_source(self):
        if (self.url is None) == (self.data is None):
            raise ValueError("AudioPart requires exactly one of: url or data")
        return self

    def to_base64(self) -> str | None:
        if self.data is None:
            return None
        return base64.b64encode(self.data).decode("utf-8")


class FilePart(BaseModel):
    """Document file (PDF, etc). Can reference by file_id, url, or inline bytes."""

    type: Literal["file"] = "file"
    file_id: str | None = None  # provider-uploaded file reference
    url: str | None = None
    data: bytes | None = None
    mime_type: str = "application/pdf"
    filename: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self):
        sources: list[t.Any] = [self.file_id, self.url, self.data]
        if sum(s is not None for s in sources) != 1:
            raise ValueError("FilePart requires exactly one of: file_id, url, or data")
        return self

    def to_base64(self) -> str | None:
        if self.data is None:
            return None
        return base64.b64encode(self.data).decode("utf-8")


ContentPart = Annotated[
    Union[TextPart, ImagePart, AudioPart, FilePart],
    Discriminator("type"),
]

MessageContent = Union[str, list[ContentPart]]


# -------- TOOL CALL -----------------------------------------------------------
class ToolCall(BaseModel):
    """A tool invocation requested by the assistant."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: dict[str, t.Any] = Field(default_factory=dict)


# -------- CORE MESSAGE -----------------------------------------------------------
class CoreMessage(BaseModel):
    """Base class for all message types.

    The `role` field is overridden in each concrete subclass with a
    Literal so the `Message` discriminated union can pick the right
    subtype during deserialization.
    """

    model_config = ConfigDict(frozen=True)

    role: str = Field(default="", description="The message role")
    source: str = Field(description="Source of the message (agent name, user id, etc.)")
    content: MessageContent = Field(default="", description="Message content")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = Field(default=0, description="Number of tokens")

    # -------- HELPERS TO INTROSPECT CONTENT -----------------------------------------------------------
    def is_multimodal(self) -> bool:
        return isinstance(self.content, list)

    def text(self) -> str:
        """Return the textual portion of the content (concatenated)."""
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text for p in self.content if isinstance(p, TextPart))

    def parts(self) -> list[ContentPart]:
        """Always return a list of parts (wrapping a string in a TextPart)."""
        if isinstance(self.content, str):
            return [TextPart(text=self.content)] if self.content else []
        return list(self.content)

    def __str__(self) -> str:
        time_str = self.created_at.strftime("%H:%M:%S")
        body = self.text() if self.is_multimodal() else self.content
        extras = ""
        if self.is_multimodal():
            kinds = [p.type for p in self.content if not isinstance(p, TextPart)]
            if kinds:
                extras = f" [+{', '.join(kinds)}]"
        return f"[{self.source}] {time_str} | {body}{extras}"

    @classmethod
    def parse_msg(cls, data: dict[str, t.Any]) -> "CoreMessage":
        """Deserialize a dict into the correct concrete subtype.

        Delegates to the discriminated `Message` union — `role` selects
        the subtype.
        """
        return _MESSAGE_ADAPTER.validate_python(data)

    def with_token_count(self, func: t.Callable[[str], int] | None = None) -> t.Self:
        """Return a copy with token_count set. Only counts textual content."""
        if self.token_count > 0:
            return self
        text = self.text()
        if func:
            count = func(text) if text else 0
        else:
            buffer = 5  # Message structure overhead: <|start|>role<|message|>
            count = buffer + max(1, int(len(text) / 4)) if text else 0
        return self.model_copy(update={"token_count": count})


# -------- CONCRETE MESSAGE TYPES -----------------------------------------------------------
class SystemMessage(CoreMessage):
    """System message containing instructions / role definition for the agent."""

    role: Literal["system"] = "system"


class UserMessage(CoreMessage):
    """User message containing input from a human or external system.

    Multi-modal input (images, audio, files) lives in `content` as a list
    of ContentPart objects.
    """

    role: Literal["user"] = "user"
    name: str | None = Field(default=None, description="Optional name of the user")

    @classmethod
    def with_image(
        cls,
        *,
        source: str,
        text: str | None,
        image_url: str | None = None,
        image_data: bytes | None = None,
        mime_type: str = "image/png",
        name: str | None = None,
    ) -> "UserMessage":
        parts: list[ContentPart] = []
        if text:
            parts.append(TextPart(text=text))
        parts.append(ImagePart(url=image_url, data=image_data, mime_type=mime_type))
        return cls(source=source, content=parts, name=name)

    @classmethod
    def with_audio(
        cls,
        *,
        source: str,
        text: str | None,
        audio_url: str | None = None,
        audio_data: bytes | None = None,
        mime_type: str = "audio/mpeg",
        name: str | None = None,
    ) -> "UserMessage":
        parts: list[ContentPart] = []
        if text:
            parts.append(TextPart(text=text))
        parts.append(AudioPart(url=audio_url, data=audio_data, mime_type=mime_type))
        return cls(source=source, content=parts, name=name)

    @classmethod
    def with_file(
        cls,
        *,
        source: str,
        text: str | None,
        file_id: str | None = None,
        file_url: str | None = None,
        file_data: bytes | None = None,
        mime_type: str = "application/pdf",
        filename: str | None = None,
        name: str | None = None,
    ) -> "UserMessage":
        parts: list[ContentPart] = []
        if text:
            parts.append(TextPart(text=text))
        parts.append(
            FilePart(
                file_id=file_id,
                url=file_url,
                data=file_data,
                mime_type=mime_type,
                filename=filename,
            )
        )
        return cls(source=source, content=parts, name=name)


class AssistantMessage(CoreMessage):
    """Assistant message containing output from the agent/LLM."""

    role: Literal["assistant"] = "assistant"

    tool_calls: list[ToolCall] = Field(default_factory=list)
    thinking: str | None = Field(default=None, description="Reasoning")
    structured_output: BaseModel | None = Field(default=None)

    def _format_tool_calls(self) -> str:
        if not self.tool_calls:
            return ""
        formatted: list[str] = []
        for tc in self.tool_calls:
            params = ", ".join(f"{k}={v}" for k, v in tc.parameters.items())
            formatted.append(f"{tc.tool_name}({params})")
        return ", ".join(formatted)

    def __str__(self) -> str:
        base = super().__str__()
        if not self.tool_calls:
            return base
        tools = self._format_tool_calls()
        if self.text().strip():
            return f"{base} [tools: {tools}]"
        return f"{base} [calling tools: {tools}]"


class ToolMessage(CoreMessage):
    """Tool message containing output from a tool execution."""

    role: Literal["tool"] = "tool"

    tool_call_id: str = Field(description="ID of the tool call this responds to")
    tool_name: str = Field(description="Name of the tool that was called")
    success: bool = Field(description="Whether the tool execution was successful")
    error: str | None = Field(default=None, description="Error message, if any")

    @classmethod
    def _create(
        cls,
        *,
        tool_call_id: str,
        tool_name: str,
        content: MessageContent,
        source: str,
        success: bool,
        error: str | None = None,
    ) -> t.Self:
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            success=success,
            content=content,
            source=source,
            error=error,
        )

    @classmethod
    def success_message(
        cls, tool_call_id: str, tool_name: str, content: MessageContent, source: str
    ) -> t.Self:
        return cls._create(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=content,
            source=source,
            success=True,
        )

    @classmethod
    def error_message(
        cls, tool_call_id: str, tool_name: str, error: str, source: str
    ) -> t.Self:
        return cls._create(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=error,
            source=source,
            success=False,
            error=error,
        )

    @classmethod
    def rejected_by_user(
        cls, tool_call_id: str, tool_name: str, reason: str, source: str
    ) -> t.Self:
        content = (
            f"Tool '{tool_name}' was rejected by the user. "
            f"Reason: {reason}. "
            "Do not attempt to fulfill this request by other means. "
            "Acknowledge and ask how to proceed."
        )
        return cls._create(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=content,
            source=source,
            success=True,
        )


class StopMessage(CoreMessage):
    """Signals that orchestration has stopped."""

    role: Literal["stop"] = "stop"
    metadata: dict[str, t.Any] = Field(
        default_factory=dict, description="Additional stop details"
    )


# -------- UNION TYPE -----------------------------------------------------------
Message = Annotated[
    Union[
        SystemMessage,
        UserMessage,
        AssistantMessage,
        ToolMessage,
        StopMessage,
    ],
    Discriminator("role"),
]


# Module-level adapter so `parse_msg` doesn't rebuild it on every call.
_MESSAGE_ADAPTER: TypeAdapter[Message] = TypeAdapter(Message)