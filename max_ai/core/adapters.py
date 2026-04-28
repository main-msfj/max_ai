"""Global Adapter for API"""

import typing as t
from .messages import TextPart, ImagePart, AudioPart, FilePart, ContentPart, CoreMessage


def _part_to_openai(part: ContentPart) -> dict[str, t.Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        if part.url:
            return {"type": "image_url", "image_url": {"url": part.url}}
        b64 = part.to_base64()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{part.mime_type};base64,{b64}"},
        }
    if isinstance(part, AudioPart):
        # OpenAI expects format without the "audio/" prefix
        fmt = part.mime_type.split("/")[-1]
        return {
            "type": "input_audio",
            "input_audio": {"data": part.to_base64(), "format": fmt},
        }
    if isinstance(part, FilePart):
        if part.file_id:
            return {"type": "file", "file": {"file_id": part.file_id}}
        if part.url:
            return {"type": "file", "file": {"file_url": part.url}}
        return {
            "type": "file",
            "file": {
                "filename": part.filename or "file",
                "file_data": f"data:{part.mime_type};base64,{part.to_base64()}",
            },
        }
    raise ValueError(f"Unsupported part type: {type(part)}")


def to_openai(msg: CoreMessage) -> dict[str, t.Any]:
    """Convert a CoreMessage to the OpenAI chat completions format."""
    if isinstance(msg.content, str):
        return {"role": msg.role, "content": msg.content}
    return {
        "role": msg.role,
        "content": [_part_to_openai(p) for p in msg.content],
    }


def _part_to_anthropic(part: ContentPart) -> dict[str, t.Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        if part.url:
            return {"type": "image", "source": {"type": "url", "url": part.url}}
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": part.mime_type,
                "data": part.to_base64(),
            },
        }
    if isinstance(part, FilePart):
        # Anthropic supports PDFs as documents
        if part.url:
            return {"type": "document", "source": {"type": "url", "url": part.url}}
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": part.mime_type,
                "data": part.to_base64(),
            },
        }
    if isinstance(part, AudioPart):
        raise NotImplementedError(
            "Anthropic Messages API does not accept audio input. "
            "Transcribe the audio first and send as text, or route to a "
            "provider that supports audio (OpenAI / Gemini)."
        )
    raise ValueError(f"Unsupported part type: {type(part)}")


def to_anthropic(msg: CoreMessage) -> dict[str, t.Any]:
    """Convert a CoreMessage to the Anthropic Messages format."""
    if isinstance(msg.content, str):
        return {"role": msg.role, "content": msg.content}
    return {
        "role": msg.role,
        "content": [_part_to_anthropic(p) for p in msg.content],
    }
