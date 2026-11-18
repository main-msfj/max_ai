"""Signals a middleware raises."""


class StopRun(Exception):
    """Raise from any middleware to end the turn cleanly: ``message`` is
    what the user reads, ``finish_reason`` why the turn ended, and ``label``
    the short reason the model reads on each call that did not run."""

    def __init__(self, message: str, finish_reason: str = "stopped", label: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.finish_reason = finish_reason
        self.label = label or message
