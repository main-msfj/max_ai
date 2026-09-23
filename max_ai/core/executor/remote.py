"""Common native-tool invocation for Docker and Modal."""

import json
from datetime import datetime

from ...base.executor import ExecutorBase
from ...types.tool_call import ToolResult
from .reference import reference_for


class RemoteExecutor(ExecutorBase):
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
            session, ["python", "-m", "max_ai.core.executor.worker"],
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
