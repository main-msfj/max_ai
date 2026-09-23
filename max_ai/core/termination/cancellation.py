"""
Cancellation token for async task control.

Provides a lightweight, thread-safe mechanism to cancel ongoing or pending
asynchronous operations. Designed to coordinate cancellation across asyncio
and threaded execution contexts.
"""

import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Callable, List, Union


class CancellationToken:
    """Thread-safe token used to cancel async or pending operations."""

    def __init__(self) -> None:
        self._cancelled: bool = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        """Trigger cancellation and notify all registered callbacks."""
        with self._lock:
            if self._cancelled:
                return

            self._cancelled = True

            for callback in self._callbacks:
                try:
                    callback()
                except Exception:
                    pass

    def is_cancelled(self) -> bool:
        """Return True if cancellation has been requested."""
        with self._lock:
            return self._cancelled

    def add_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a callback to be executed when cancellation occurs.

        If already cancelled, the callback is executed immediately.
        """
        with self._lock:
            if self._cancelled:
                try:
                    callback()
                except Exception:
                    pass
            else:
                self._callbacks.append(callback)

    def link_future(
        self,
        future: Union[Future[Any], asyncio.Future[Any], asyncio.Task[Any]],
    ) -> Union[Future[Any], asyncio.Future[Any], asyncio.Task[Any]]:
        """
        Link a future or task to this token.

        The future will be cancelled automatically if the token is triggered.
        """
        with self._lock:
            if self._cancelled:
                future.cancel()
                return future

            def _cancel() -> None:
                future.cancel()

            self._callbacks.append(_cancel)

        return future
