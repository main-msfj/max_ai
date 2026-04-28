import logging
import typing as t

from ..base.middleware import CoreMiddleware


class MiddlewareLogger:
    """Structured logger for middleware execution pipeline."""

    def __init__(self, logger: logging.Logger, operation: str, agent_id: str):
        self._logger = logger
        self._base = {"operation": operation, "agent_id": agent_id}

    def _extra(
        self,
        phase: str,
        middleware: CoreMiddleware | None = None,
        **kwargs: t.Any,
    ) -> dict[str, t.Any]:
        payload = {**self._base, "phase": phase, **kwargs}

        if middleware is not None:
            payload["middleware"] = middleware.__class__.__name__

        return payload

    def _log(
        self,
        level: str,
        msg: str,
        exc: Exception | None = None,
        phase: str = "unknown",
        middleware: CoreMiddleware | None = None,
        **extra: t.Any,
    ) -> None:
        log_fn = getattr(self._logger, level)
        log_fn(
            msg,
            exc_info=exc,
            extra=self._extra(phase, middleware, **extra),
        )

    def request_error(self, middleware: CoreMiddleware, exc: Exception) -> None:
        self._log(
            "error",
            "Middleware failed during request phase",
            exc=exc,
            phase="request",
            middleware=middleware,
            next_action="attempt_recovery",
        )

    def execution_error(self, exc: Exception) -> None:
        self._log(
            "error",
            "Core execution failed",
            exc=exc,
            phase="execution",
            next_action="attempt_recovery",
        )

    def response_error(self, middleware: CoreMiddleware, exc: Exception) -> None:
        self._log(
            "error",
            "Middleware failed during response phase",
            exc=exc,
            phase="response",
            middleware=middleware,
            next_action="attempt_recovery",
        )

    def stream_error(self, middleware: CoreMiddleware, exc: Exception) -> None:
        self._log(
            "warning",
            "Middleware failed to process stream chunk",
            exc=exc,
            phase="stream",
            middleware=middleware,
        )

    def recovery_success(self, middleware: CoreMiddleware) -> None:
        self._logger.info(
            "Middleware recovered successfully",
            extra=self._extra(
                "recovery",
                middleware,
                next_action="continue_execution",
            ),
        )

    def recovery_error(self, middleware: CoreMiddleware, exc: Exception) -> None:
        self._log(
            "error",
            "Middleware failed during recovery",
            exc=exc,
            phase="recovery",
            middleware=middleware,
        )

    def unrecoverable(self, exc: Exception) -> None:
        self._log(
            "error",
            "Middleware chain failed with no recovery path",
            exc=exc,
            phase="fatal",
            next_action="abort",
        )

    def unexpected_yield(
        self,
        middleware: CoreMiddleware,
        item: t.Any,
        method: str,
    ) -> None:
        self._logger.warning(
            "Unexpected yield from middleware",
            extra=self._extra(
                "validation",
                middleware,
                type=type(item).__name__,
                method=method,
            ),
        )
