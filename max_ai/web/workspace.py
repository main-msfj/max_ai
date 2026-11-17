"""A conversation's folder in the agent's sandbox, as the web UI uses it:
the Files pane, ``@path`` mentions and ``!command``. Everything runs with
python3 in the sandbox, so a Docker sandbox works the same as this computer."""

from __future__ import annotations

import asyncio
import json
import posixpath
import shlex
import typing as t
from pathlib import Path

from ..base.sandbox import CommandResult

if t.TYPE_CHECKING:
    from ..agents.agent import Agent

# Every file under the folder, folders included (size -1), without caches and .git.
LIST = r"""
import json, os, sys
root = sys.argv[1]
os.makedirs(root, exist_ok=True)
skip = {".git", "node_modules", "__pycache__", ".venv", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
out = []
for here, dirs, files in os.walk(root):
    dirs[:] = sorted(d for d in dirs if d not in skip)
    rel = os.path.relpath(here, root)
    for name in dirs:
        out.append([os.path.normpath(os.path.join(rel, name)), -1])
    for name in sorted(files):
        try:
            size = os.path.getsize(os.path.join(here, name))
        except OSError:
            size = 0
        out.append([os.path.normpath(os.path.join(rel, name)), size])
    if len(out) > int(sys.argv[2]):
        break
print(json.dumps(out))
"""

# One file of the folder: its text, or why it has none.
READ = r"""
import json, os, sys
root = os.path.realpath(sys.argv[1])
path = os.path.realpath(os.path.join(root, sys.argv[2]))
if not path.startswith(root + os.sep) or not os.path.isfile(path):
    print(json.dumps({"error": "not a file of the workspace"}))
    sys.exit()
size = os.path.getsize(path)
with open(path, "rb") as f:
    data = f.read(int(sys.argv[3]))
if b"\0" in data[:8192]:
    print(json.dumps({"size": size, "binary": True}))
else:
    print(json.dumps({"size": size, "text": data.decode("utf-8", "replace"), "cut": size > len(data)}))
"""


class Workspace:
    """The files of one user's sessions, in the agent's sandbox or, without
    one, in the folder the program runs in."""

    MAX_FILES = 3000  # Entries the Files pane lists
    MAX_READ = 256_000  # Bytes a file view or a mention reads
    TIMEOUT = 120  # Seconds a ``!command`` may run

    def __init__(self, agent: Agent, user_id: str) -> None:
        self.agent = agent
        self.user_id = user_id

    def folder(self, session_id: str) -> str:
        """Where the session's files are, as the agent sees it."""
        sandbox = self.agent.sandbox
        if sandbox is None:
            return str(Path.cwd())
        return sandbox.bind(self.user_id).session_paths(session_id)["workspace"]

    async def files(self, session_id: str) -> list[tuple[str, int]]:
        """Relative paths and sizes; a folder has size -1."""
        folder = self.folder(session_id)
        ran = await self._python(LIST, folder, str(self.MAX_FILES))
        return [(str(path), int(size)) for path, size in json.loads(ran.stdout or "[]")]

    async def read(self, session_id: str, path: str) -> dict[str, t.Any]:
        """``text`` (and ``cut`` when longer than shown), ``binary``, or ``error``."""
        if not self.relative(path):
            return {"error": "not a file of the workspace"}
        ran = await self._python(READ, self.folder(session_id), path, str(self.MAX_READ))
        try:
            return t.cast("dict[str, t.Any]", json.loads(ran.stdout))
        except json.JSONDecodeError:
            return {"error": ran.stderr.strip() or "could not read it"}

    async def shell(self, session_id: str, command: str) -> CommandResult:
        """``command`` run from the session's folder."""
        folder = shlex.quote(self.folder(session_id))
        return await self._run(f"mkdir -p {folder} && cd {folder} && {command}", timeout=self.TIMEOUT)

    async def mentions(self, session_id: str, text: str, found: list[str]) -> str:
        """``text`` with the files in ``found`` (``@path`` of the message) attached."""
        attached: list[str] = []
        for path in dict.fromkeys(found):
            content = await self.read(session_id, path)
            if "text" in content:
                attached.append(f"\n\n--- {path} ---\n\n{content['text']}")
        return text + "".join(attached)

    @staticmethod
    def relative(path: str) -> bool:
        """A path inside the folder: not absolute, no ``..``."""
        clean = posixpath.normpath(path)
        return bool(path) and not posixpath.isabs(path) and clean != ".." and not clean.startswith("../")

    async def _python(self, script: str, *args: str) -> CommandResult:
        quoted = " ".join(shlex.quote(arg) for arg in args)
        return await self._run(f"python3 - {quoted}", stdin=script.encode())

    async def _run(self, command: str, *, timeout: float = 30, stdin: bytes | None = None) -> CommandResult:
        """In the sandbox; without one, with bash here."""
        sandbox = self.agent.sandbox
        if sandbox is not None:
            box = sandbox.bind(self.user_id)
            await box.prepare()
            return await box.execute_command(command, timeout=timeout, stdin=stdin)
        process = await asyncio.create_subprocess_exec(
            "bash", "-c", command, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(process.communicate(stdin), timeout)
        except TimeoutError:
            process.kill()
            return CommandResult("", "timed out", None, timed_out=True)
        return CommandResult(out.decode(errors="replace"), err.decode(errors="replace"), process.returncode)
