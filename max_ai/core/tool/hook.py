"""ToolHook: the four moments of a tool call where the harness has a say."""

from __future__ import annotations

import json
import mimetypes
import re
import typing as t

from ...base.sandbox import SandboxError
from ...types.tool_call import ToolCallRecord, ToolResult
from ...types.tools import ToolApprovalMode

if t.TYPE_CHECKING:
    from ...base.tools import CoreTool, ToolContext

MAX_RESULT_CHARS = 40_000
PREVIEW_CHARS = 2_000  # shown from each end when the rest went to a file

# What the model reads after a result that was too long.
SAVED = (
    "The result has {size:,} characters, too many to show: only its start and "
    "end are above. All of it is in {path}"
)
MAX_FILE_BYTES = 20 * 1024 * 1024
FILES = "The tool returned {count} file(s):"
NOT_KEPT = "could not be kept, so it cannot be shown"
CUT = "The result has {size:,} characters, too many to show: {cut:,} were cut from the middle."


class ToolHook:
    """Inherit and write only the moments you need. As it is, every call
    goes on, a tool runs alone only if it says so, and a result too long
    is shortened."""

    async def before(
        self, tool: CoreTool, record: ToolCallRecord, tool_ctx: ToolContext
    ) -> ToolResult | None:
        """The call is valid and about to go on. Return None to let it, or
        the result the model gets in its place."""
        return None

    async def permission(
        self, tool: CoreTool, record: ToolCallRecord, tool_ctx: ToolContext
    ) -> bool | None:
        """True runs it, False denies it, None asks the user. As it is: what
        the tool's own ``approval_mode`` says."""
        auto = ToolApprovalMode(tool.approval_mode) == ToolApprovalMode.AUTO_APPROVED
        return True if auto else None

    async def after(
        self, tool: CoreTool, record: ToolCallRecord, tool_ctx: ToolContext, result: ToolResult
    ) -> ToolResult:
        """The tool worked. Return the result the model gets."""
        result = await self.keep_files(record, tool_ctx, result)
        return await self.shorten(record, tool_ctx, result)

    async def failed(
        self, tool: CoreTool, record: ToolCallRecord, tool_ctx: ToolContext, result: ToolResult
    ) -> ToolResult:
        """The tool failed. Return the error the model gets."""
        result = await self.keep_files(record, tool_ctx, result)
        return await self.shorten(record, tool_ctx, result)

    async def keep_files(
        self, record: ToolCallRecord, tool_ctx: ToolContext, result: ToolResult
    ) -> ToolResult:
        """Each file the tool returned goes to the sandbox, and the model gets
        where it is. Without a sandbox it gets what the file was."""
        if not result.files:
            return result
        lines = []
        for number, file in enumerate(result.files, 1):
            what = f"{file.mime_type}, {len(file.data) / 1024:,.0f} KB"
            path = None
            if len(file.data) <= MAX_FILE_BYTES:
                extension = mimetypes.guess_extension(file.mime_type) or ".bin"
                path = await _save(file.data, f"{_safe(record.id)}-{number}{extension}", tool_ctx)
            lines.append(f"- {what}: {path or NOT_KEPT}")
        note = "\n".join([FILES.format(count=len(lines)), *lines])
        text = _as_text(result)
        shown = f"{text}\n\n{note}" if text else note
        return result.model_copy(
            update={"result" if result.success else "error": shown, "files": []}
        )

    async def shorten(
        self, record: ToolCallRecord, tool_ctx: ToolContext, result: ToolResult
    ) -> ToolResult:
        """A result too long to show: saved whole in the sandbox when there
        is one, cut in the middle when there is not."""
        text = _as_text(result)
        if len(text) <= MAX_RESULT_CHARS:
            return result
        path = await _save(text.encode(), f"{_safe(record.id)}.txt", tool_ctx)
        if path is not None:
            keep, note = PREVIEW_CHARS, SAVED.format(size=len(text), path=path)
        else:
            keep = MAX_RESULT_CHARS // 2
            note = CUT.format(size=len(text), cut=len(text) - 2 * keep)
        shown = f"{text[:keep]}\n\n[...]\n\n{text[-keep:]}\n\n{note}"
        return result.model_copy(update={"result" if result.success else "error": shown})


def _as_text(result: ToolResult) -> str:
    """What the model would read of ``result``."""
    if not result.success:
        return result.error or ""
    if isinstance(result.result, str):
        return result.result
    return json.dumps(result.result, ensure_ascii=False, default=str)


def _safe(call_id: str) -> str:
    """A call id as a file name: it comes from the model's provider."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", call_id)


async def _save(data: bytes, name: str, tool_ctx: ToolContext) -> str | None:
    """Where ``data`` was saved in the session's tmp; None when it could not be."""
    if tool_ctx.sandbox is None or not tool_ctx.session_id:
        return None
    try:
        return await tool_ctx.sandbox.upload_file(
            tool_ctx.session_id, f"tool_results/{name}", data, temporary=True
        )
    except (SandboxError, ValueError):
        return None  # saving failed: the caller says so instead
