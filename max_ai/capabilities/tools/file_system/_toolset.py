"""User-scoped file tools, in the style of Claude Code: ReadFile, WriteFile,
EditFile, DeleteFile, ListDirectory, FindFiles and SearchFile.

The harness remembers what the model read (``RunContext.file_reads``): writing
over, editing or deleting a file needs a read of its current version.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

from pydantic import Field

from ....base.tools import ToolContext
from ....core.event_type import (
    DirectoryListedEvent,
    FileDeletedEvent,
    FileReadEvent,
    FilesSearchedEvent,
    FileWrittenEvent,
)
from ....types.tools import ToolApprovalMode
from ...workspace.local import LocalWorkspace as Workspace
from ...workspace.local._filesystem import Stamp, UserFileSystem
from ..decorator import tool
from . import constant as c

READ_ONLY_TOOL_NAMES = frozenset(
    {c.READ_FILE, c.LIST_DIRECTORY, c.FIND_FILES, c.SEARCH_FILE}
)


class FileSystemTools:
    """Workspace-relative file tools, one shared workspace per user."""

    def __init__(self, workspace: str | Path | UserFileSystem | None = None):
        """Tools over ``workspace``; without one, the agent's workspace is used."""
        self._workspace = (
            workspace
            if isinstance(workspace, UserFileSystem)
            else UserFileSystem(workspace)
            if workspace is not None
            else None
        )
        # Used only when no RunContext comes with the call (tools run alone).
        self._own_reads: dict[str, dict[str, Stamp]] = {}

        @tool(
            name=c.READ_FILE,
            description=c.READ_FILE_DESCRIPTION,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            policy_subject="file_path",
            read_only=True,
        )
        def read_file(
            context: ToolContext,
            file_path: Annotated[str, Field(description=c.FILE_PATH)],
            offset: Annotated[int, Field(description=c.OFFSET, ge=1)] = 1,
            limit: Annotated[int, Field(description=c.LIMIT_LINES, ge=1)] = c.DEFAULT_READ_LINES,
        ) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            data = filesystem.read_text(context.user_id, self._workspace_path(context, file_path))
            self._reads(context)[data["path"]] = data["stamp"]
            self._emit(context, FileReadEvent(
                source=c.READ_FILE, tool_call_id=self._tool_call_id(context),
                path=data["path"], root_dir=str(filesystem.root), content_hash=data["sha256"],
            ))
            result: dict[str, Any] = {"path": data["path"]}
            if data["text"] is None:
                return {**result, "bytes": data["bytes"], "binary": True, "note": c.BINARY}
            lines = data["text"].splitlines()
            if not lines:
                return {**result, "content": "", "note": c.EMPTY}
            shown = lines[offset - 1:offset - 1 + limit]
            result["content"] = "\n".join(
                f"{number:>6}\t{_cut(line)}" for number, line in enumerate(shown, start=offset)
            )
            last = offset - 1 + len(shown)
            result["lines"] = f"{offset}-{last} of {len(lines)}" if shown else f"0 of {len(lines)}"
            if last < len(lines):
                result["note"] = c.MORE_LINES.format(total=len(lines), next=last + 1)
            return result

        @tool(
            name=c.WRITE_FILE,
            description=c.WRITE_FILE_DESCRIPTION,
            approval_mode=ToolApprovalMode.ASK_APPROVED,
            policy_subject="file_path",
        )
        def write_file(
            context: ToolContext,
            file_path: Annotated[str, Field(description=c.WRITE_PATH)],
            content: Annotated[str, Field(description=c.CONTENT)],
        ) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            target = self._write_target(context, file_path)
            reads = self._reads(context)
            try:
                result = filesystem.write_text_file(
                    context.user_id, target, content, reads.get(target)
                )
            except FileNotFoundError:
                if target not in reads:
                    raise
                # It was read, then deleted (by bash, say): create it again.
                result = filesystem.write_text_file(context.user_id, target, content)
            reads[result["path"]] = result.pop("stamp")
            self._emit(context, FileWrittenEvent(
                source=c.WRITE_FILE, tool_call_id=self._tool_call_id(context),
                operation=c.WRITE_FILE, path=result["path"], root_dir=str(filesystem.root),
                content_hash=result.pop("sha256"),
            ))
            return result

        @tool(
            name=c.EDIT_FILE,
            description=c.EDIT_FILE_DESCRIPTION,
            approval_mode=ToolApprovalMode.ASK_APPROVED,
            policy_subject="file_path",
        )
        def edit_file(
            context: ToolContext,
            file_path: Annotated[str, Field(description=c.FILE_PATH)],
            old_string: Annotated[str, Field(description=c.OLD_STRING)],
            new_string: Annotated[str, Field(description=c.NEW_STRING)],
            replace_all: Annotated[bool, Field(description=c.REPLACE_ALL)] = False,
        ) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            path = self._workspace_path(context, file_path)
            reads = self._reads(context)
            result = filesystem.edit_text_file(
                context.user_id, path=path, old_text=old_string, new_text=new_string,
                expected=self._read_before(reads, path), replace_all=replace_all,
            )
            reads[result["path"]] = result.pop("stamp")
            self._emit(context, FileWrittenEvent(
                source=c.EDIT_FILE, tool_call_id=self._tool_call_id(context),
                operation=c.EDIT_FILE, path=result["path"], root_dir=str(filesystem.root),
                content_hash=result.pop("sha256"),
            ))
            return result

        @tool(
            name=c.DELETE_FILE,
            description=c.DELETE_FILE_DESCRIPTION,
            approval_mode=ToolApprovalMode.ASK_APPROVED,
            policy_subject="file_path",
        )
        def delete_file(
            context: ToolContext,
            file_path: Annotated[str, Field(description=c.FILE_PATH)],
        ) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            path = self._workspace_path(context, file_path)
            reads = self._reads(context)
            result = filesystem.delete_file(context.user_id, path, self._read_before(reads, path))
            reads.pop(result["path"], None)
            self._emit(context, FileDeletedEvent(
                source=c.DELETE_FILE, tool_call_id=self._tool_call_id(context),
                path=result["path"], root_dir=str(filesystem.root),
                content_hash=result.pop("sha256"),
            ))
            return result

        @tool(
            name=c.LIST_DIRECTORY,
            description=c.LIST_DIRECTORY_DESCRIPTION,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            read_only=True,
        )
        def list_directory(
            context: ToolContext,
            path: Annotated[str, Field(description=c.DIRECTORY)] = "",
            limit: Annotated[int, Field(description=c.LIMIT_RESULTS, ge=1)] = c.MAX_RESULTS,
        ) -> dict[str, Any]:
            filesystem = self._filesystem(context)
            resolved = self._workspace_path(context, path, allow_empty=True)
            result = filesystem.list_files(context.user_id, path=resolved, limit=limit)
            self._emit(context, DirectoryListedEvent(
                source=c.LIST_DIRECTORY, tool_call_id=self._tool_call_id(context),
                path=result["path"], root_dir=str(filesystem.root),
                entry_count=len(result["items"]),
            ))
            return result

        @tool(
            name=c.FIND_FILES,
            description=c.FIND_FILES_DESCRIPTION,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            read_only=True,
        )
        def find_files(
            context: ToolContext,
            pattern: Annotated[str, Field(description=c.GLOB_PATTERN)],
            path: Annotated[str, Field(description=c.SEARCH_ROOT)] = "",
            limit: Annotated[int, Field(description=c.LIMIT_RESULTS, ge=1)] = 100,
        ) -> dict[str, Any]:
            if not pattern or len(pattern) > 255 or "\0" in pattern or "\\" in pattern:
                raise ValueError("pattern must be a glob of 1 to 255 characters")
            filesystem = self._filesystem(context)
            root = self._search_root(context, path)
            files, truncated = filesystem.scan_files(
                context.user_id, path=root, limit=c.MAX_SCANNED_FILES
            )
            matches = [
                item["path"] for item in files
                if _glob_match(_relative(item["path"], root), pattern)
            ]
            limit = min(limit, c.MAX_RESULTS)
            truncated = truncated or len(matches) > limit
            self._emit(context, FilesSearchedEvent(
                source=c.FIND_FILES, tool_call_id=self._tool_call_id(context),
                operation=c.FIND_FILES, path=root or ".", root_dir=str(filesystem.root),
                match_count=len(matches), truncated=truncated,
            ))
            return {"pattern": pattern, "files": matches[:limit], "truncated": truncated}

        @tool(
            name=c.SEARCH_FILE,
            description=c.SEARCH_FILE_DESCRIPTION,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            read_only=True,
        )
        def search_file(
            context: ToolContext,
            pattern: Annotated[str, Field(description=c.REGEX)],
            path: Annotated[str, Field(description=c.SEARCH_ROOT)] = "",
            glob: Annotated[str, Field(description=c.GLOB_FILTER)] = "",
            ignore_case: Annotated[bool, Field(description=c.IGNORE_CASE)] = False,
            limit: Annotated[int, Field(description=c.LIMIT_RESULTS, ge=1)] = 100,
        ) -> dict[str, Any]:
            if not pattern or len(pattern) > c.MAX_PATTERN_CHARS:
                raise ValueError(f"pattern must have 1 to {c.MAX_PATTERN_CHARS} characters")
            try:
                regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
            except re.error as error:
                raise ValueError(f"pattern is not a valid regular expression: {error}") from None
            filesystem = self._filesystem(context)
            root = self._search_root(context, path)
            result = self._search(context, filesystem, regex, root, glob, min(limit, c.MAX_RESULTS))
            self._emit(context, FilesSearchedEvent(
                source=c.SEARCH_FILE, tool_call_id=self._tool_call_id(context),
                operation=c.SEARCH_FILE, path=root or ".", root_dir=str(filesystem.root),
                match_count=len(result["matches"]), truncated=result["truncated"],
            ))
            return {"pattern": pattern, **result}

        self.tools = [
            read_file,
            write_file,
            edit_file,
            delete_file,
            list_directory,
            find_files,
            search_file,
        ]

    # -------- PATHS -----------------------------------------------------------
    @staticmethod
    def _workspace_path(
        context: ToolContext,
        path: str,
        *,
        allow_empty: bool = False,
    ) -> str:
        """Keep the user root out of model-visible tool arguments."""
        if path == "" and allow_empty:
            return "workspace"
        parts = FileSystemTools._as_given(context, UserFileSystem._visible_parts(path))
        if not parts:
            return "workspace"
        if parts[0] == "skills":
            return "/".join(parts)
        return "/".join(("workspace", *parts))

    @staticmethod
    def _search_root(context: ToolContext, path: str) -> str:
        """Where FindFiles/SearchFile look: empty is the workspace and the skills."""
        return "" if path in ("", ".") else FileSystemTools._workspace_path(context, path)

    @staticmethod
    def _write_target(context: ToolContext, path: str) -> str:
        """Resolve a WriteFile path to its full location (always in the workspace)."""
        parts = FileSystemTools._as_given(context, UserFileSystem._visible_parts(path))
        return "/".join(("workspace", *parts))

    @staticmethod
    def _as_given(context: ToolContext, parts: tuple[str, ...]) -> tuple[str, ...]:
        """Accept paths exactly as the tools return them: ``workspace/x`` is ``x``."""
        if parts and parts[0] == "workspace":
            return parts[1:]
        return parts

    # -------- READS -----------------------------------------------------------
    def _reads(self, context: ToolContext) -> dict[str, Stamp]:
        """What the model has read: ``RunContext.file_reads``, saved with the session."""
        run_context = context.deps.get("run_context")
        if run_context is not None:
            return run_context.file_reads
        return self._own_reads.setdefault(f"{context.user_id}/{context.session_id}", {})

    @staticmethod
    def _read_before(reads: dict[str, Stamp], path: str) -> Stamp:
        """The stamp from the model's last read of ``path``; changing needs one."""
        if path not in reads:
            raise ValueError(c.NOT_READ.format(path=path))
        return reads[path]

    # -------- HELPERS -----------------------------------------------------------
    @staticmethod
    def _tool_call_id(context: ToolContext) -> str:
        """Return the current call id when a tool adapter provides it."""
        return str(context.deps.get("tool_call_id", "unknown"))

    @staticmethod
    def _emit(context: ToolContext, event: Any) -> None:
        """Send ``event`` to the run's stream, when there is one."""
        if context.emit_event is not None:
            context.emit_event(event)

    def _filesystem(self, context: ToolContext) -> UserFileSystem:
        """The explicit workspace, else the agent's, else the default root."""
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

    @staticmethod
    def _search(
        context: ToolContext,
        filesystem: UserFileSystem,
        regex: re.Pattern[str],
        root: str,
        glob: str,
        limit: int,
    ) -> dict[str, Any]:
        """Matching lines below ``root``, within the scan and byte budgets."""
        files, truncated = filesystem.scan_files(
            context.user_id, path=root, limit=c.MAX_SCANNED_FILES
        )
        matches: list[dict[str, Any]] = []
        bytes_read = 0
        for item in files:
            if glob and not _glob_match(_relative(item["path"], root), glob):
                continue
            size = min(int(item["bytes"]), c.MAX_SEARCH_FILE_BYTES)
            if bytes_read + size > c.MAX_SEARCH_TOTAL_BYTES:
                truncated = True
                break
            data = filesystem.read_bytes(context.user_id, item["path"], size)
            bytes_read += len(data)
            if b"\0" in data:
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(
                        {"path": item["path"], "line": number, "text": line[:c.MAX_MATCH_CHARS]}
                    )
                    if len(matches) >= limit:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": truncated}


def _cut(line: str) -> str:
    """A line as ReadFile shows it: long ones end in ``…``."""
    return line if len(line) <= c.MAX_LINE_CHARS else line[:c.MAX_LINE_CHARS] + "…"


def _relative(path: str, root: str) -> str:
    """``path`` as seen from the search root, so ``reports/*.md`` works the same
    from anywhere; with no root, workspace files drop their ``workspace/``."""
    if not root:
        return path.removeprefix("workspace/")
    if path == root:
        return PurePosixPath(path).name
    return PurePosixPath(path).relative_to(root).as_posix()


def _glob_match(path: str, pattern: str) -> bool:
    """Glob as people write it: a bare name matches at any depth, ``**/`` is any
    folder, and a word without wildcards is part of the file name."""
    pattern = pattern.removeprefix("./")
    while pattern.startswith("**/"):
        pattern = pattern[3:]
    name = PurePosixPath(path).name
    if not any(char in pattern for char in "*?["):
        return path == pattern if "/" in pattern else pattern.lower() in name.lower()
    if "/" not in pattern:
        return fnmatch.fnmatchcase(name, pattern)
    # fnmatch's * also crosses "/", so data/**/*.csv matches any depth below data/.
    return fnmatch.fnmatchcase(path, pattern.replace("**/", "*"))
