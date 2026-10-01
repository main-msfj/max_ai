"""Errors raised by executors (Local, Docker, Modal)."""

SANDBOX_LOST = (
    "The sandbox stopped ({detail}) and a new one starts with the next command. "
    "Workspace files are kept as of the last command that finished; /tmp, running "
    "processes and packages installed in this session are gone. This command did not "
    "finish: check its effects before running it again."
)


class SandboxLost(RuntimeError):
    """The runtime behind a session is gone (killed, timed out, out of memory).

    The EnvironmentManager drops the session and connects a new one on the
    next acquire; the command that hit it is not run again.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(SANDBOX_LOST.format(detail=detail.strip() or "unknown reason"))
