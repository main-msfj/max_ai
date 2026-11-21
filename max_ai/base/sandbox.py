"""The machine where the agent works: one per user.

``.agent/`` lives inside it (see Layout): skills and the user's files go
in, and commands run there.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import posixpath
import shlex
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel

from ..config import setting
from ..errors.sandbox import SandboxError
from .component import ComponentBase
from .layout import SCRIPT, Layout
from .session_store import validate_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandResult:
    """What a command printed and how it ended."""

    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False


class SandboxCore(ComponentBase[BaseModel], ABC):
    """Implement once per provider: Local, Docker, ACA, Firecracker.

    A child only says where ``.agent/`` goes (``base``) and how a command
    runs; it also defines its config model and ``_to_config`` /
    ``_from_config``. Everything else is written here, once.
    """

    # What only the harness does: these write setting.json or the skills.
    HARNESS_OPS: t.ClassVar[frozenset[str]] = frozenset({"create", "install", "permit", "unpermit"})

    component_type = "sandbox"

    def __init__(self, base: str, extra_env: list[str] | None = None) -> None:
        self.layout = Layout(base)
        # Names of variables of this machine that the commands also see.
        self.extra_env = list(extra_env or [])
        # Set by bind(): whose machine this is and what its commands see.
        self.identifier: str | None = None
        self.variables: dict[str, str] = {}

    # -------- EACH PROVIDER ------------------------------------------------------
    @abstractmethod
    async def execute_command(
        self,
        command: str,
        *,
        timeout: float = 60,
        stdin: bytes | None = None,
        harness: bool = False,
    ) -> CommandResult:
        """Run ``command`` with bash in the sandbox, with ``self.environment()``
        in its environment. A failing command is an exit code; only
        infrastructure problems raise. ``harness``: it is the harness's own
        work (it writes the settings or a skill), never a command of the model."""

    async def close(self) -> None:
        """Release what the provider started (a container). Nothing by default."""

    # -------- WRITTEN ONCE -------------------------------------------------------
    def bind(self, identifier: str, variables: dict[str, str] | None = None) -> t.Self:
        """A copy for one user: ``identifier`` names whose machine it is,
        ``variables`` is what their commands see. ``self`` is never changed
        and the variables are never serialized."""
        bound = copy.copy(self)
        bound.identifier = identifier
        bound.variables = dict(variables or {})
        return bound

    def environment(self) -> dict[str, str]:
        """What the commands see besides the provider's own: the variables
        named in ``extra_env`` that this machine has, read now and never
        stored, and the user's ``variables``."""
        extra = {name: os.environ[name] for name in self.extra_env if name in os.environ}
        return {**extra, **self.variables}

    async def prepare(self) -> None:
        """Create ``.agent/`` if it is missing; never overwrite what is there."""
        await self._run("create", **self.layout.skeleton())

    def session_paths(self, session_id: str) -> dict[str, str]:
        """Where a session works in the sandbox: what the prompt shows."""
        validate_id("session_id", session_id)
        return {
            "workspace": str(self.layout.workspace / session_id),
            "tmp": str(self.layout.tmp / session_id),
        }

    async def upload_file(
        self, session_id: str, path: str, content: bytes, *, temporary: bool = False
    ) -> str:
        """Put a file the user attached (an image, a PDF) at ``path`` inside
        the session's folder; with ``temporary``, inside the session's tmp
        folder instead. Returns where it is in the sandbox."""
        validate_id("session_id", session_id)
        if posixpath.isabs(path):
            raise ValueError(f"{path} must be relative to the session's folder")
        root = self.layout.tmp if temporary else self.layout.workspace
        answer = await self._run(
            "upload", content, folder=str(root / session_id), path=path,
        )
        return answer["path"]

    async def outdated_skills(self, wanted: dict[str, str]) -> list[str]:
        """The skills of ``wanted`` (name → hash or commit) that are not
        installed, whose key in ``setting.json`` is another one, or whose
        files are no longer the ones that were installed."""
        if not wanted:
            return []
        # 1. What the settings say, and what is really on disk.
        installed = await self.installed_skills()
        answer = await self._run("hashes", folder=str(self.layout.skills), names=list(wanted))
        on_disk: dict[str, str | None] = answer["hashes"]
        outdated: list[str] = []
        for name, key in wanted.items():
            known = installed.get(name, {})
            # 2. Another version, or never installed.
            if known.get("key") != key:
                outdated.append(name)
            # 3. The same version, but someone changed its files in the sandbox.
            elif known.get("hash") != on_disk.get(name):
                logger.warning(
                    "The skill %s was changed inside the sandbox of %s; it is installed again.",
                    name, self.identifier,
                )
                outdated.append(name)
        return outdated

    async def permissions(self) -> dict[str, list[str]]:
        """The user's rules in ``setting.json``: ``allow``, ``ask`` and ``deny``."""
        answer = await self._run("settings", settings=str(self.layout.settings))
        return answer["settings"].get("permissions", {})

    async def add_permission(self, kind: str, rule: str) -> None:
        """Keep ``rule`` in the ``kind`` list (allow, ask or deny) of ``setting.json``."""
        await self._run(
            "permit", settings=str(self.layout.settings), lock=str(self.layout.lock),
            kind=kind, rule=rule,
        )

    async def remove_permission(self, kind: str, rule: str) -> None:
        """Take ``rule`` out of the ``kind`` list (allow, ask or deny) of ``setting.json``."""
        await self._run(
            "unpermit", settings=str(self.layout.settings), lock=str(self.layout.lock),
            kind=kind, rule=rule,
        )

    async def installed_skills(self) -> dict[str, dict[str, str]]:
        """What ``setting.json`` says is installed: name → its ``key`` and
        ``description``."""
        answer = await self._run("settings", settings=str(self.layout.settings))
        return answer["settings"].get("skills", {})

    async def install_skill(self, name: str, key: str, archive: bytes, description: str = "") -> None:
        """Unpack ``archive`` (a tar, compressed or not) into ``skills/<name>``,
        replacing what was there, and record ``key`` and ``description`` in
        ``setting.json``."""
        validate_id("skill name", name)
        await self._run(
            "install",
            archive,
            name=name,
            key=key,
            description=description,
            folder=str(self.layout.skills / name),
            settings=str(self.layout.settings),
            lock=str(self.layout.lock),
        )

    async def _run(self, op: str, data: bytes | None = None, **request: t.Any) -> dict[str, t.Any]:
        """Run one operation of ``layout/script.py`` inside the sandbox. The
        request goes as an argument; ``data`` goes as it is on stdin, with
        its sha256 in the request so the script knows all of it arrived."""
        request["op"] = op
        if data is not None:
            request["sha256"] = hashlib.sha256(data).hexdigest()
        result = await self.execute_command(
            f"python3 -c {shlex.quote(SCRIPT)} {shlex.quote(json.dumps(request))}",
            timeout=setting.tool_timeout_seconds,
            stdin=data,
            harness=op in self.HARNESS_OPS,
        )
        if result.timed_out or result.exit_code != 0:
            raise SandboxError(f"{op} failed in the sandbox: {result.stderr.strip() or 'timed out'}")
        answer = json.loads(result.stdout)
        if not answer.pop("ok"):
            raise SandboxError(f"{op} failed in the sandbox: {answer['error']}")
        return answer
