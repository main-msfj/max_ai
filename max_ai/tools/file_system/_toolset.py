"""User-scoped filesystem tools exposed to agents."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from ...base.tools import ToolContext
from ...core.event_type import (
    DirectoryCreatedEvent,
    DirectoryListedEvent,
    FileDeletedEvent,
    FileInfoEvent,
    FileReadEvent,
    FileWrittenEvent,
    FilesSearchedEvent,
)
from ...types.tools import ToolApprovalMode
from ...workspace.filesystem import UserFileSystem
from ...base.workspace import Workspace
from ..decorator import tool


_MAX_TOOL_RESULTS = 200
_MAX_SEARCH_FILES = 1000
_MAX_SEARCH_FILE_BYTES = 128 * 1024
_MAX_SEARCH_TOTAL_BYTES = 8 * 1024 * 1024
_MAX_QUERY_CHARS = 2000
_MAX_MATCH_CHARS = 500

READ_ONLY_TOOL_NAMES = frozenset(
    {
        "list_directory",
        "find_files",
        "search_text",
        "read_file",
        "file_info",
    }
)


class FileSystemTools:
    """Conversation-relative file tools backed by one user workspace."""

    def __init__(self, workspace: str | Path | UserFileSystem | None = None):
        self._workspace = (
            workspace
            if isinstance(workspace, UserFileSystem)
            else UserFileSystem(workspace)
            if workspace is not None
            else None
        )

        @tool(
            name="list_directory",
            description="List a folder in the current conversation. Use paths relative to the conversation.",
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
        def list_directory(
            context: ToolContext, path: str = "", limit: int = 200
        ) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            resolved = self._conversation_path(context, path, allow_empty=True)
            result = filesystem.list_files(context.user_id, path=resolved, limit=limit)
            self._emit(
                context,
                DirectoryListedEvent(
                    source="list_directory",
                    tool_call_id=self._tool_call_id(context),
                    path=result["path"],
                    root_dir=str(filesystem.root),
                    entry_count=len(result["items"]),
                ),
            )
            return result

        @tool(
            name="find_files",
            description="Find a file name or glob across all conversations belonging to the current user.",
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
        def find_files(
            context: ToolContext,
            file_name: str,
            limit: int = 100,
        ) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            result = self._find_files(context, file_name, limit)
            self._emit(
                context,
                FilesSearchedEvent(
                    source="find_files",
                    tool_call_id=self._tool_call_id(context),
                    operation="find_files",
                    path=".",
                    root_dir=str(filesystem.root),
                    match_count=len(result["items"]),
                    truncated=result["truncated"],
                ),
            )
            return result

        @tool(
            name="search_text",
            description="Search text across all conversations belonging to the current user.",
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
        def search_text(
            context: ToolContext,
            query: str,
            limit: int = 100,
        ) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            result = self._search_text(context, query, limit)
            self._emit(
                context,
                FilesSearchedEvent(
                    source="search_text",
                    tool_call_id=self._tool_call_id(context),
                    operation="search_text",
                    path=".",
                    root_dir=str(filesystem.root),
                    match_count=len(result["matches"]),
                    truncated=result["truncated"],
                ),
            )
            return result

        @tool(
            name="read_file",
            description=(
                "Read a file from the current conversation. Pass conversation_id only "
                "when reading a result found in another conversation."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
        def read_file(
            context: ToolContext,
            file_name: str,
            conversation_id: str | None = None,
            max_bytes: int = 65536,
        ) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            resolved = self._conversation_path(context, file_name, conversation_id)
            result = filesystem.read_file_details(
                context.user_id, path=resolved, max_bytes=max_bytes
            )
            self._emit(
                context,
                FileReadEvent(
                    source="read_file",
                    tool_call_id=self._tool_call_id(context),
                    path=result["path"],
                    root_dir=str(filesystem.root),
                    content_hash=result["sha256"],
                ),
            )
            return result

        @tool(
            name="write_file",
            description="Create a UTF-8 text file in the current conversation.",
            approval_mode=ToolApprovalMode.ASK_APPROVED,
        )
        def write_file(
            context: ToolContext, file_name: str, content: str
        ) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            result = filesystem.create_text_file(
                context.user_id,
                context.session_id,
                path=file_name,
                content=content,
            )
            self._emit(
                context,
                FileWrittenEvent(
                    source="write_file",
                    tool_call_id=self._tool_call_id(context),
                    operation="write_file",
                    path=result["path"],
                    root_dir=str(filesystem.root),
                    content_hash=result["sha256"],
                ),
            )
            return result

        @tool(
            name="edit_file",
            description=(
                "Replace one exact text occurrence in a conversation file after verifying "
                "the SHA-256 returned by read_file."
            ),
            approval_mode=ToolApprovalMode.ASK_APPROVED,
        )
        def edit_file(
            context: ToolContext,
            file_name: str,
            old_text: str,
            new_text: str,
            expected_sha256: str,
            conversation_id: str | None = None,
        ) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            resolved = self._conversation_path(context, file_name, conversation_id)
            result = filesystem.edit_text_file(
                context.user_id,
                path=resolved,
                old_text=old_text,
                new_text=new_text,
                expected_sha256=expected_sha256,
            )
            self._emit(
                context,
                FileWrittenEvent(
                    source="edit_file",
                    tool_call_id=self._tool_call_id(context),
                    operation="edit_file",
                    path=result["path"],
                    root_dir=str(filesystem.root),
                    content_hash=result["sha256"],
                ),
            )
            return result

        @tool(
            name="create_directory",
            description="Create a directory in the current conversation.",
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
        def create_directory(context: ToolContext, path: str) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            result = filesystem.create_directory(
                context.user_id, context.session_id, path
            )
            self._emit(
                context,
                DirectoryCreatedEvent(
                    source="create_directory",
                    tool_call_id=self._tool_call_id(context),
                    path=result["path"],
                    root_dir=str(filesystem.root),
                ),
            )
            return result

        @tool(
            name="file_info",
            description=(
                "Inspect a file or directory in the current conversation. Pass "
                "conversation_id only for a result from another conversation."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
        def file_info(
            context: ToolContext, file_name: str, conversation_id: str | None = None
        ) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            resolved = self._conversation_path(context, file_name, conversation_id)
            result = filesystem.file_info(context.user_id, resolved)
            self._emit(
                context,
                FileInfoEvent(
                    source="file_info",
                    tool_call_id=self._tool_call_id(context),
                    path=result["path"],
                    root_dir=str(filesystem.root),
                    file_type=result["type"],
                ),
            )
            return result

        @tool(
            name="delete_file",
            description=(
                "Delete a file from the current conversation after verifying the SHA-256 "
                "returned by read_file. Pass conversation_id only for an older conversation."
            ),
            approval_mode=ToolApprovalMode.ASK_APPROVED,
        )
        def delete_file(
            context: ToolContext,
            file_name: str,
            expected_sha256: str,
            conversation_id: str | None = None,
        ) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            resolved = self._conversation_path(context, file_name, conversation_id)
            result = filesystem.delete_file(context.user_id, resolved, expected_sha256)
            self._emit(
                context,
                FileDeletedEvent(
                    source="delete_file",
                    tool_call_id=self._tool_call_id(context),
                    path=result["path"],
                    root_dir=str(filesystem.root),
                    content_hash=result["sha256"],
                ),
            )
            return result

        self.tools = [
            list_directory,
            find_files,
            search_text,
            read_file,
            write_file,
            edit_file,
            create_directory,
            file_info,
            delete_file,
        ]

    @staticmethod
    def _conversation_path(
        context: ToolContext,
        path: str,
        conversation_id: str | None = None,
        *,
        allow_empty: bool = False,
    ) -> str:
        """Keep user and workspace roots out of model-visible tool arguments."""
        session = conversation_id or context.session_id
        UserFileSystem._safe_session_id(session)
        if path == "" and allow_empty:
            return session
        parts = UserFileSystem._visible_parts(path)
        if parts[0] == "skills":
            if conversation_id is not None:
                raise ValueError("conversation_id cannot be used with skills")
            return "/".join(parts)
        return "/".join((session, *parts))

    @staticmethod
    def _tool_call_id(context: ToolContext) -> str:
        """Return the current call id when a tool adapter provides it."""
        return str(context.deps.get("tool_call_id", "unknown"))

    @staticmethod
    def _emit(context: ToolContext, event: Any) -> None:
        if context.emit_event is not None:
            context.emit_event(event)

    def _filesystem(self, context: ToolContext) -> UserFileSystem:
        if self._workspace is not None:
            filesystem = self._workspace
        else:
            filesystem = context.deps.get("workspace_filesystem")
        if filesystem is None:
            root = context.deps.get("filesystem_root")
            if root is None:
                root = Workspace().base_root
            if isinstance(root, UserFileSystem):
                filesystem = root
            else:
                filesystem = UserFileSystem(root)
        return filesystem

    def _find_files(
        self, context: ToolContext, pattern: str, limit: int
    ) -> dict[str, Any]:
        if (
            not isinstance(pattern, str)
            or not pattern
            or len(pattern) > 255
            or "/" in pattern
            or "\\" in pattern
            or "\0" in pattern
        ):
            raise ValueError(
                "file_name must be a basename or glob of 1 to 255 characters"
            )
        effective_limit = UserFileSystem._bounded_limit(limit, _MAX_TOOL_RESULTS)
        glob = (
            pattern
            if any(character in pattern for character in "*?[")
            else f"*{pattern}*"
        )
        filesystem = self._filesystem(context)
        files, scan_truncated = filesystem.scan_files(
            context.user_id, limit=_MAX_SEARCH_FILES
        )
        matches = [
            item for item in files if fnmatch.fnmatch(Path(item["path"]).name, glob)
        ]
        truncated = scan_truncated or len(matches) > effective_limit
        return {
            "pattern": pattern,
            "items": matches[:effective_limit],
            "truncated": truncated,
        }

    def _search_text(
        self, context: ToolContext, query: str, limit: int
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query or len(query) > _MAX_QUERY_CHARS:
            raise ValueError(f"query must contain 1 to {_MAX_QUERY_CHARS} characters")
        effective_limit = UserFileSystem._bounded_limit(limit, _MAX_TOOL_RESULTS)
        filesystem = self._filesystem(context)
        files, scan_truncated = filesystem.scan_files(
            context.user_id, limit=_MAX_SEARCH_FILES
        )
        matches: list[dict[str, Any]] = []
        truncated = scan_truncated
        bytes_read = 0
        for item in files:
            file_bytes = min(int(item["bytes"]), _MAX_SEARCH_FILE_BYTES)
            if bytes_read + file_bytes > _MAX_SEARCH_TOTAL_BYTES:
                truncated = True
                break
            data = filesystem.read_bytes(context.user_id, item["path"], file_bytes)
            bytes_read += len(data)
            if b"\0" in data:
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if query not in line:
                    continue
                matches.append(
                    {
                        "path": item["path"],
                        "line": line_number,
                        "text": line[:_MAX_MATCH_CHARS],
                    }
                )
                if len(matches) >= effective_limit:
                    truncated = True
                    break
            if len(matches) >= effective_limit:
                break
        return {"query": query, "matches": matches, "truncated": truncated}
