"""
Filesystem-backed routine registry.

Reads routine packages from a local source directory. To stay
behaviorally identical to remote-backed registries (S3, Git, HTTP),
``connect()`` materializes the authorized routines into a dedicated
temporary directory and all subsequent operations work against that
tmp copy. The loader never touches the original source files.

The agent only sees the routines explicitly named in ``routines`` at
construction. A repository may contain hundreds of routine files —
the registry surfaces only the ones authorized for this agent. If
the user references an unauthorized routine, ``fetch`` fails with a
message listing what's actually available.

The tmp directory is created on ``connect()`` and is NOT cleaned up
by the registry itself. Cleanup is the caller's responsibility (or a
separate pipeline). The agent doesn't need to know or care.

Not concurrency-safe: simultaneous external writers can clobber each
other. Intended for local development and tests.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path
from pydantic import BaseModel

from ...base.routines import CoreRoutineRegistry, RoutineToolMode
from ...core import RoutineBlocks
from ...types.routines import RoutineSummary


class LocalRoutineRegistryConfig(BaseModel):
    source_path: str
    routines: list[str]
    tool_mode: RoutineToolMode = RoutineToolMode.FULL

class LocalRoutineRegistry(CoreRoutineRegistry):
    component_schema = LocalRoutineRegistryConfig
    component_type = "routines"

    """Filesystem-backed implementation of ``CoreRoutineRegistry``.

    Source layout (read-only)::

        source_path/
            routines/
                client_followup.json
                email_reply.json
                code_review.json
                ... (may contain many more)

    Authorized layout (after ``connect()``, in the registry's tmp dir)::

        {tmp_root}/
            client_followup.json   ← only the names passed in `routines`
            email_reply.json

    Each ``.json`` file contains a single ``RoutineBlocks`` payload.
    The filename stem MUST match the ``name`` field inside the JSON.
    """

    def __init__(
        self,
        source_path: str | Path,
        routines: list[str],
        tool_mode: RoutineToolMode = RoutineToolMode.FULL,
    ) -> None:
        super().__init__(tool_mode=tool_mode)
        self.source_path: Path = Path(source_path).expanduser().resolve()
        if not self.source_path.is_dir():
            raise FileNotFoundError(
                f"Routine source directory does not exist: {self.source_path}"
            )

        self.routines: list[str] = self._validate_routine_names(routines)
        self._tmp_root: Path | None = None

    def _to_config(self) -> LocalRoutineRegistryConfig:
        return LocalRoutineRegistryConfig(
            source_path=str(self.source_path),
            routines=list(self.routines),
            tool_mode=self.tool_mode,
        )

    @classmethod
    def _from_config(cls, config: LocalRoutineRegistryConfig) -> "LocalRoutineRegistry":
        return cls(
            source_path=config.source_path,
            routines=config.routines,
            tool_mode=config.tool_mode,
        )

    # -------- VALIDATION -----------------------------------------------------------
    @staticmethod
    def _validate_routine_names(routines: list[str]) -> list[str]:
        if not isinstance(routines, list):
            raise TypeError(
                f"routines must be a list, got {type(routines).__name__}"
            )
        if not routines:
            raise ValueError("routines list cannot be empty")
        if not all(isinstance(r, str) and r.strip() for r in routines):
            raise ValueError(
                "routines must be a list of non-empty, non-whitespace strings"
            )
        cleaned = [r.strip() for r in routines]
        dupes = {r for r in cleaned if cleaned.count(r) > 1}
        if dupes:
            raise ValueError(f"Duplicate routine names: {sorted(dupes)}")
        return cleaned

    # -------- PATH HELPERS -----------------------------------------------------------
    @property
    def _source_routines_dir(self) -> Path:
        return self.source_path / "routines"

    def _tmp_routine_file(self, name: str) -> Path:
        assert self._tmp_root is not None  # set by connect()
        return self._tmp_root / f"{name}.json"

    # -------- LIFECYCLE -----------------------------------------------------------
    async def connect(self) -> None:
        """Create the tmp directory and stage every authorized routine.

        Each name in ``self.routines`` must correspond to an existing
        ``{source_path}/routines/{name}.json`` file — missing files
        fail loud here so the problem surfaces at agent setup, not at
        first tool call.
        """
        if self._tmp_root is not None:
            return  # already connected

        self._tmp_root = Path(tempfile.mkdtemp(prefix="maxai_routines_"))

        if not self._source_routines_dir.is_dir():
            raise FileNotFoundError(
                f"Routine source must contain a 'routines/' subdirectory: "
                f"{self._source_routines_dir}"
            )

        missing: list[str] = []
        for name in self.routines:
            source_file = self._source_routines_dir / f"{name}.json"
            if not source_file.is_file():
                missing.append(name)
                continue
            shutil.copy2(source_file, self._tmp_routine_file(name))

        if missing:
            raise FileNotFoundError(
                f"Authorized routines not found in source: {sorted(missing)}. "
                f"Source dir: {self._source_routines_dir}"
            )

    async def disconnect(self) -> None:
        # Cleanup is intentionally NOT performed here. A separate
        # pipeline owns lifecycle of the tmp space.
        return None

    # -------- AGENT BOUNDARY -----------------------------------------------------------
    async def get_catalog(self) -> list[RoutineSummary]:
        """List every authorized routine as a lightweight summary."""
        await self._ensure_connected()
        return [block.to_summary() for block in self._load_all()]

    # -------- READ OPERATIONS -----------------------------------------------------------
    async def search(
        self, query: str, limit: int = 5
    ) -> list[RoutineSummary]:
        """Search authorized routines by ``name`` and ``description`` only.

        Routines whose combined ``name + description`` has zero token
        overlap with the query are dropped; the rest are sorted by
        score descending and the top ``limit`` are returned.
        """
        await self._ensure_connected()
        if not isinstance(query, str) or not query.strip():
            return []

        scored: list[tuple[float, RoutineSummary]] = []
        for block in self._load_all():
            haystack = f"{block.name} {block.description}"
            score = self._fake_score(query, haystack)
            if score <= 0:
                continue
            scored.append(
                (score, RoutineSummary(name=block.name, description=block.description))
            )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [summary for _, summary in scored[:limit]]

    async def fetch(self, name: str) -> RoutineBlocks:
        """Load an authorized routine's full content by name.

        Raises ``ValueError`` when ``name`` is not in the authorized
        list. The message includes the authorized names so the LLM
        can recover by picking a real one on the next turn.
        """
        await self._ensure_connected()
        clean_name = self._validate_name(name)

        if clean_name not in self.routines:
            raise ValueError(
                f"Unknown routine {clean_name!r}. "
                f"Available routines: {sorted(self.routines)}"
            )

        path = self._tmp_routine_file(clean_name)
        if not path.is_file():
            # Authorized but missing in tmp → corruption (caller deleted
            # the staging file?). Fail loud rather than masking it.
            raise FileNotFoundError(
                f"Authorized routine {clean_name!r} is missing from "
                f"the staging directory: {path}. The tmp space may "
                f"have been tampered with externally."
            )
        return self._load_one(path)

    # -------- INTERNALS -----------------------------------------------------------
    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got {type(name).__name__}")
        clean = name.strip()
        if not clean:
            raise ValueError("name must be a non-empty, non-whitespace string")
        return clean

    @staticmethod
    def _fake_score(query: str, content: str) -> float:
        """Token-overlap score with a tiny random tiebreaker."""
        query_tokens = {t for t in query.lower().split() if t}
        if not query_tokens:
            return 0.0
        content_tokens = {t for t in content.lower().split() if t}
        overlap = len(query_tokens & content_tokens)
        if overlap == 0:
            return 0.0
        base = overlap / len(query_tokens)
        return base + random.uniform(0, 0.001)

    def _load_all(self) -> list[RoutineBlocks]:
        """Load every authorized routine from the staging directory."""
        return [self._load_one(self._tmp_routine_file(name))
                for name in self.routines]

    def _load_one(self, path: Path) -> RoutineBlocks:
        """Load and validate one routine file from the staging dir.

        Cross-checks the ``name`` field inside the JSON against the
        filename stem so a misnamed file can't quietly shadow another
        routine.
        """
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Routine file {path} is not valid JSON: {e}") from e

        if not isinstance(raw, dict):
            raise ValueError(
                f"Routine file {path} must contain a JSON object at the "
                f"top level, got {type(raw).__name__}"
            )

        try:
            block = RoutineBlocks.model_validate(raw)
        except Exception as e:
            raise ValueError(
                f"Routine file {path} is not a valid RoutineBlocks: {e}"
            ) from e

        if block.name != path.stem:
            raise ValueError(
                f"Routine file {path}: 'name' field is {block.name!r} but "
                f"filename stem is {path.stem!r}. They must match."
            )
        return block
