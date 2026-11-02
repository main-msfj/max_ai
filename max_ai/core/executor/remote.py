"""What Docker and Modal share: network settings and the command wrapper."""

import math

from ...base.executor import ExecutorBase

# Where pip, uv and npm download packages from.
PACKAGE_REGISTRIES: tuple[str, ...] = (
    "pypi.org", "*.pypi.org", "files.pythonhosted.org", "*.pythonhosted.org",
    "registry.npmjs.org", "*.npmjs.org",
)
NETWORK_MODES = ("packages", "internet", "none")

def supervised(script: str, timeout: float, run_id: str) -> list[str]:
    """argv that runs ``script`` in a sandbox, bounded by ``timeout``.

    GNU timeout gives the command its own process group and kills the whole
    group; its PID goes to /tmp so ``kill_argv`` can stop it on cancel.
    """
    return [
        "bash", "--noprofile", "--norc", "-c",
        'echo $$ > "/tmp/maxai-$1.pid"; exec timeout -k 5 "$2" bash --noprofile --norc -c "$3"',
        "maxai", run_id, str(math.ceil(timeout)), script,
    ]


def kill_argv(run_id: str) -> list[str]:
    """argv that stops a ``supervised`` command; timeout forwards it to the group."""
    return ["bash", "-c", 'kill -TERM "$(cat "/tmp/maxai-$1.pid")" 2>/dev/null; true', "maxai", run_id]


class RemoteExecutor(ExecutorBase):
    """Base for executors that run commands away from the host.

    Network access is shared by every provider:

    - ``"packages"``: only the package registries (pip, uv, npm) plus ``allow_list``.
    - ``"internet"``: every site, or only ``allow_list`` when it is given.
    - ``"none"``: no network; ``allow_list`` is rejected.

    A provider that cannot filter by domain sets ``supports_allow_list = False``
    and only accepts ``"internet"`` and ``"none"``.
    """

    supports_allow_list = True
    isolated = True
    network: str = "none"
    allow_list: list[str] = []

    def _init_network(self, network: str, allow_list: list[str] | None) -> None:
        """Validate and store the network settings."""
        allowed = NETWORK_MODES if self.supports_allow_list else ("internet", "none")
        if network not in allowed:
            raise ValueError(f"{type(self).__name__} network must be one of {', '.join(allowed)}")
        if allow_list and not self.supports_allow_list:
            raise ValueError(f"{type(self).__name__} cannot filter by domain: allow_list is not supported")
        if allow_list and network == "none":
            raise ValueError("allow_list needs network='packages' or 'internet', not 'none'")
        self.network, self.allow_list = network, list(allow_list or [])

    def _allowed_domains(self) -> list[str] | None:
        """The domains the sandbox may reach; ``None`` means no domain filter."""
        if self.network == "packages":
            return [*PACKAGE_REGISTRIES, *self.allow_list]
        if self.network == "internet":
            return self.allow_list or None
        return None

    def _network_description(self, installers: str = "pip, uv or npm") -> str:
        """One sentence for the prompt about what the model can reach."""
        extra = ", ".join(self.allow_list)
        if self.network == "none":
            return "There is no network access: installs and downloads fail."
        if self.network == "internet":
            return f"It can only reach: {extra}." if extra else "It has internet access."
        if extra:
            return (f"Install what a script needs with {installers} before running it; besides the "
                    f"package registries it can only reach: {extra}.")
        return f"Install what a script needs with {installers} before running it; other sites are blocked."
