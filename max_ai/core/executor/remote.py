"""Common native-tool invocation for Docker and Modal."""

import json
from datetime import datetime

from ...base.executor import ExecutorBase
from ...types.tool_call import ToolResult
from .reference import reference_for

# Where pip, uv and npm download packages from.
PACKAGE_REGISTRIES: tuple[str, ...] = (
    "pypi.org", "*.pypi.org", "files.pythonhosted.org", "*.pythonhosted.org",
    "registry.npmjs.org", "*.npmjs.org",
)
NETWORK_MODES = ("packages", "internet", "none")
# max_ai runs inside the sandbox from its own venv, apart from the agent's python.
FRAMEWORK_PYTHON = "/opt/maxai/bin/python"
# Until the PyPI release; then "maxai==<version>".
DEFAULT_FRAMEWORK = "maxai @ git+https://github.com/main-msfj/max_ai@main"


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

    @staticmethod
    def _emit_events(context, events):
        if context.emit_event is None:
            return
        from ..event_type import (
            BashCancelledEvent,
            BashFailedEvent,
            BashFinishedEvent,
            BashStartedEvent,
            DirectoryCreatedEvent,
            DirectoryListedEvent,
            FileDeletedEvent,
            FileInfoEvent,
            FileReadEvent,
            FilesSearchedEvent,
            FileWrittenEvent,
        )
        event_types = {event.EVENT_TYPE: event for event in (
            BashStartedEvent, BashFinishedEvent, BashFailedEvent, BashCancelledEvent,
            DirectoryCreatedEvent, DirectoryListedEvent, FileDeletedEvent,
            FileInfoEvent, FileReadEvent, FileWrittenEvent, FilesSearchedEvent,
        )}
        for raw in events if isinstance(events, list) else ():
            if not isinstance(raw, dict) or raw.get("event_type") not in event_types:
                continue
            data = dict(raw)
            event_type = data.pop("event_type")
            timestamp = data.get("timestamp")
            if isinstance(timestamp, str):
                data["timestamp"] = datetime.fromisoformat(timestamp)
            context.emit_event(event_types[event_type](**data))

    async def run_tool(self, session, tool, record, context, cancellation_token=None):
        reference = context.deps.get("tool_reference") or reference_for(tool)
        payload = {
            "reference": reference.model_dump(mode="json"),
            "record": record.model_dump(mode="json"),
            "context": {
                "run_id": context.run_id, "user_id": context.user_id,
                "session_id": context.session_id,
                "retry_count": context.retry_count,
            },
            "workspace_path": session.workspace_path,
            "tool_parameters": tool.parameters,
        }
        # Only explicit JSON context crosses this boundary; no host clients,
        # filesystem handles, tokens, callbacks or credentials are serialized.
        result = await self.execute_argv(
            session, [FRAMEWORK_PYTHON, "-m", "max_ai.core.executor.worker"],
            stdin=json.dumps(payload), timeout=tool.timeout_seconds,
            cancellation_token=cancellation_token,
        )
        if result.timed_out:
            return ToolResult.timeout(record.id, tool.timeout_seconds)
        if result.exit_code != 0 or result.truncated:
            return ToolResult.execution_error(
                record.id, result.stderr or "Remote worker failed or output exceeded limit",
            )
        try:
            response = ToolResult.model_validate_json(result.stdout)
        except ValueError:
            return ToolResult.execution_error(record.id, "Invalid remote worker response")
        if response.tool_call_id != record.id:
            return ToolResult.execution_error(record.id, "Remote result identity mismatch")
        self._emit_events(context, response.metadata.get("events"))
        return response
