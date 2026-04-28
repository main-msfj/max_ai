"""
Structured scoped logger for contextual logging.

Adds immutable context fields (scope) to every log call, making logs
traceable across agents, tools, and sessions.
"""

import logging
import typing as t


class ScopedLogger:
    """Logger that attaches structured context to every log entry."""

    __slots__ = ("_logger", "_scope")

    def __init__(self, logger: logging.Logger, **scope: t.Any) -> None:
        self._logger = logger
        self._scope: dict[str, t.Any] = dict(scope)

    def child(self, **extra_scope: t.Any) -> "ScopedLogger":
        """Return a new logger with extended context."""
        return ScopedLogger(self._logger, **self._scope, **extra_scope)

    def _log(
        self,
        level: int,
        msg: str,
        *,
        exc: Exception | None = None,
        **extra: t.Any,
    ) -> None:
        self._logger.log(
            level,
            msg,
            extra={**self._scope, **extra},
            exc_info=exc,
        )

    def debug(self, msg: str, **extra: t.Any) -> None:
        self._log(logging.DEBUG, msg, **extra)

    def info(self, msg: str, **extra: t.Any) -> None:
        self._log(logging.INFO, msg, **extra)

    def warning(self, msg: str, **extra: t.Any) -> None:
        self._log(logging.WARNING, msg, **extra)

    def error(
        self,
        msg: str,
        *,
        exc: Exception | None = None,
        **extra: t.Any,
    ) -> None:
        self._log(logging.ERROR, msg, exc=exc, **extra)
