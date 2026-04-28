"""Errors raised by chat completion clients.

A single ``ClientError`` exception covers all client-side failure
modes. The ``kind`` attribute identifies the specific category for
callers that need to react differently (e.g. retry on rate limit).

Use the factory classmethods rather than ``ClientError(...)`` directly
to keep error messages consistent across providers.
"""

from __future__ import annotations


class ClientError(Exception):
    """Generic chat completion client error.

    Use the factory classmethods to construct instances with
    standard messages and a ``kind`` tag identifying the category.
    """

    def __init__(self, message: str, kind: str = "generic") -> None:
        super().__init__(message)
        self.kind = kind

    # -------- VALIDATION ERRORS -----------------------------------------------------------
    @classmethod
    def tools_not_supported(cls, model: str, tool_count: int) -> "ClientError":
        return cls(
            f"Model {model!r} does not support function calling, "
            f"but {tool_count} tool(s) were provided. Either set "
            "ModelConfig.supports_function_calling=True if you know "
            "the model supports it, or use a tool-capable model.",
            kind="tools_not_supported",
        )

    @classmethod
    def invalid_model_config(cls, detail: str) -> "ClientError":
        return cls(f"Invalid model configuration: {detail}", kind="invalid_config")

    @classmethod
    def invalid_request(cls, detail: str) -> "ClientError":
        return cls(f"Invalid request: {detail}", kind="invalid_request")

    # -------- AUTH & QUOTA ERRORS -----------------------------------------------------------
    @classmethod
    def authentication_failed(cls, detail: str | None = None) -> "ClientError":
        msg = "Authentication failed"
        if detail:
            msg = f"{msg}: {detail}"
        return cls(msg, kind="authentication")

    @classmethod
    def rate_limit_exceeded(
        cls, retry_after: float | None = None
    ) -> "ClientError":
        msg = "Rate limit exceeded"
        if retry_after is not None:
            msg = f"{msg}, retry after {retry_after:.1f}s"
        return cls(msg, kind="rate_limit")

    @classmethod
    def token_limit_exceeded(
        cls, requested: int | None = None, limit: int | None = None
    ) -> "ClientError":
        if requested is not None and limit is not None:
            msg = (
                f"Token limit exceeded: requested {requested}, "
                f"limit {limit}"
            )
        else:
            msg = "Token limit exceeded"
        return cls(msg, kind="token_limit")

    # -------- PROVIDER & TRANSPORT ERRORS -----------------------------------------------------------
    @classmethod
    def provider_error(
        cls, provider: str, status_code: int | None = None, detail: str = ""
    ) -> "ClientError":
        prefix = f"{provider} error"
        if status_code is not None:
            prefix = f"{prefix} (HTTP {status_code})"
        msg = f"{prefix}: {detail}" if detail else prefix
        return cls(msg, kind="provider")

    @classmethod
    def request_timeout(cls, seconds: float) -> "ClientError":
        return cls(
            f"Request timed out after {seconds:.1f}s",
            kind="timeout",
        )

    @classmethod
    def stream_interrupted(cls, detail: str | None = None) -> "ClientError":
        msg = "Stream interrupted"
        if detail:
            msg = f"{msg}: {detail}"
        return cls(msg, kind="stream_interrupted")

    # -------- RESPONSE PARSING ERRORS -----------------------------------------------------------
    @classmethod
    def invalid_response(cls, detail: str) -> "ClientError":
        return cls(
            f"Provider returned an invalid response: {detail}",
            kind="invalid_response",
        )
    
    @classmethod
    def unsupported_feature(cls, feature: str, model: str) -> "ClientError":
        return cls(
            f"Model '{model}' does not support {feature}. "
            f"Either declare support in ModelConfig or remove the unsupported "
            f"content from the message before sending."
        )

    @classmethod
    def api_error(
        cls, provider: str, status: int | None = None, detail: str | None = None
    ) -> "ClientError":
        """Provider API error (HTTP or generic).

        Used when a provider client returns an HTTP/API error response.
        """
        if status is not None:
            prefix = f"{provider} API error (HTTP {status})"
        else:
            prefix = f"{provider} API error"
        msg = f"{prefix}: {detail}" if detail else prefix
        return cls(msg, kind="api_error")

    @classmethod
    def unexpected(cls, provider: str, detail: str | None = None) -> "ClientError":
        """Unexpected provider/client exception not matching known categories."""
        msg = f"Unexpected {provider} error"
        if detail:
            msg = f"{msg}: {detail}"
        return cls(msg, kind="unexpected")