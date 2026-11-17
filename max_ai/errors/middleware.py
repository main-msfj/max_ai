"""Signals a middleware raises."""


class StopRun(Exception):
    """Raise from any middleware to end the turn cleanly: ``message`` is
    what the user reads, ``finish_reason`` why the turn ended."""

    def __init__(self, message: str, finish_reason: str = "stopped") -> None:
        """
        Initialize a run-stop signal with its user-facing reason.

        Parameters
        ----------
        message : str
            Value used to initialize the run-stop signal.
        finish_reason : str, default='stopped'
            Value used to initialize the run-stop signal.
        """
        super().__init__(message)
        self.message = message
        self.finish_reason = finish_reason
