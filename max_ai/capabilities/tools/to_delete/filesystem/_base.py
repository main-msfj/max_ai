"""FileSystem: the common base of read_file, write_file and edit_file."""

from __future__ import annotations

from .....base.tools import CoreTool

SESSION = "$SESSION/"
WORKSPACE = "$WORKSPACE/"
TMP = "$TMP/"


class FileSystem(CoreTool):
    """Common base of the file tools."""

    @classmethod
    def all(cls) -> list[FileSystem]:
        """One of each file tool."""
        return [tool() for tool in cls.__subclasses__()]

    @staticmethod
    def locate(path: str, session_id: str) -> tuple[str, str]:
        """The folder the script works in, relative to the workspace, and the
        path inside it: ``$SESSION/`` is this session, ``$WORKSPACE/`` all of
        them, ``$TMP/`` this session's scratch files."""
        if path.startswith(WORKSPACE):
            return ".", path.removeprefix(WORKSPACE)
        if not path.startswith((SESSION, TMP)):
            raise ValueError(f"{path} must start with {SESSION}, {WORKSPACE} or {TMP}")
        if not session_id:
            raise ValueError("there is no current session")
        if path.startswith(SESSION):
            return session_id, path.removeprefix(SESSION)
        return f"../tmp/{session_id}", path.removeprefix(TMP)
