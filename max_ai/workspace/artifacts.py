"""Revisioned, user-scoped artifact storage backed by SQLite."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .filesystem import UserFileSystem


_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
_DEFAULT_LIST_LIMIT = 1000


@dataclass(frozen=True)
class Artifact:
    path: str
    revision: str
    sha256: str
    size: int


class ArtifactConflict(ValueError):
    """Raised when an artifact create or update loses its revision CAS."""


class ArtifactStore(ABC):
    """Backend-neutral interface for versioned artifact storage."""

    @property
    @abstractmethod
    def identity(self) -> str:
        """Return a stable identifier for this backing store."""

    @abstractmethod
    def list(
        self, user_id: str, limit: int = _DEFAULT_LIST_LIMIT
    ) -> tuple[list[Artifact], bool]:
        """List the latest artifact revision for each path and indicate truncation."""

    @abstractmethod
    def get(self, user_id: str, path: str) -> tuple[Artifact, bytes]:
        """Return the latest artifact revision, or raise FileNotFoundError."""

    @abstractmethod
    def put(
        self,
        user_id: str,
        path: str,
        data: bytes,
        expected_revision: str | None = None,
    ) -> Artifact:
        """Create an artifact or replace its latest revision with an exact CAS."""


class LocalArtifactStore(ArtifactStore):
    """SQLite implementation retaining every committed artifact revision."""

    def __init__(self, root: str | os.PathLike[str]):
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._database = self._root / "artifacts.sqlite3"
        self._identity = f"sqlite:{self._database}"
        self._initialize()

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def root(self) -> Path:
        """Return the canonical root directory used for local persistence."""
        return self._root

    def list(
        self, user_id: str, limit: int = _DEFAULT_LIST_LIMIT
    ) -> tuple[list[Artifact], bool]:
        user = UserFileSystem._safe_id(user_id, "user_id")
        effective_limit = self._bounded_limit(limit)
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT v.path, v.revision, v.sha256, v.size
                   FROM artifact_latest AS latest
                   JOIN artifact_versions AS v
                     ON v.user_id = latest.user_id
                    AND v.path = latest.path
                    AND v.revision = latest.revision
                   WHERE latest.user_id = ?
                   ORDER BY v.path
                   LIMIT ?""",
                (user, effective_limit + 1),
            ).fetchall()
        truncated = len(rows) > effective_limit
        return [self._artifact(row) for row in rows[:effective_limit]], truncated

    def get(self, user_id: str, path: str) -> tuple[Artifact, bytes]:
        user = UserFileSystem._safe_id(user_id, "user_id")
        clean_path = self._validated_path(path)
        with self._connection() as connection:
            row = connection.execute(
                """SELECT v.path, v.revision, v.sha256, v.size, v.data
                   FROM artifact_latest AS latest
                   JOIN artifact_versions AS v
                     ON v.user_id = latest.user_id
                    AND v.path = latest.path
                    AND v.revision = latest.revision
                   WHERE latest.user_id = ? AND latest.path = ?""",
                (user, clean_path),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(clean_path)
        artifact = self._artifact(row)
        data = bytes(row[4])
        self._verify_data(artifact, data)
        return artifact, data

    def put(
        self,
        user_id: str,
        path: str,
        data: bytes,
        expected_revision: str | None = None,
    ) -> Artifact:
        user = UserFileSystem._safe_id(user_id, "user_id")
        clean_path = self._validated_path(path)
        if not isinstance(data, bytes):
            raise ValueError("data must be bytes")
        if len(data) > _MAX_ARTIFACT_BYTES:
            raise ValueError(f"data exceeds {_MAX_ARTIFACT_BYTES} bytes")
        if expected_revision is not None and (
            not isinstance(expected_revision, str) or not expected_revision
        ):
            raise ValueError("expected_revision must be a non-empty string")

        revision = uuid.uuid4().hex
        digest = hashlib.sha256(data).hexdigest()
        artifact = Artifact(clean_path, revision, digest, len(data))
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision FROM artifact_latest WHERE user_id = ? AND path = ?",
                (user, clean_path),
            ).fetchone()
            current_revision = row[0] if row is not None else None
            if current_revision != expected_revision:
                raise ArtifactConflict("artifact revision does not match")
            connection.execute(
                """INSERT INTO artifact_versions
                   (user_id, path, revision, sha256, size, data)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user, clean_path, revision, digest, len(data), sqlite3.Binary(data)),
            )
            connection.execute(
                """INSERT INTO artifact_latest (user_id, path, revision)
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id, path) DO UPDATE SET revision = excluded.revision""",
                (user, clean_path, revision),
            )
            connection.commit()
        return artifact

    def get_revision(
        self, user_id: str, path: str, revision: str
    ) -> tuple[Artifact, bytes]:
        """Return a retained historical revision, or raise FileNotFoundError."""
        user = UserFileSystem._safe_id(user_id, "user_id")
        clean_path = self._validated_path(path)
        if not isinstance(revision, str) or not revision:
            raise ValueError("revision must be a non-empty string")
        with self._connection() as connection:
            row = connection.execute(
                """SELECT path, revision, sha256, size, data
                   FROM artifact_versions
                   WHERE user_id = ? AND path = ? AND revision = ?""",
                (user, clean_path, revision),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(clean_path)
        artifact = self._artifact(row)
        data = bytes(row[4])
        self._verify_data(artifact, data)
        return artifact, data

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """CREATE TABLE IF NOT EXISTS artifact_versions (
                       user_id TEXT NOT NULL,
                       path TEXT NOT NULL,
                       revision TEXT NOT NULL,
                       sha256 TEXT NOT NULL,
                       size INTEGER NOT NULL,
                       data BLOB NOT NULL,
                       PRIMARY KEY (user_id, path, revision)
                   );
                   CREATE TABLE IF NOT EXISTS artifact_latest (
                       user_id TEXT NOT NULL,
                       path TEXT NOT NULL,
                       revision TEXT NOT NULL,
                       PRIMARY KEY (user_id, path),
                       FOREIGN KEY (user_id, path, revision)
                         REFERENCES artifact_versions (user_id, path, revision)
                   );"""
            )

    @staticmethod
    def _bounded_limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        return min(limit, _DEFAULT_LIST_LIMIT)

    @staticmethod
    def _validated_path(path: str) -> str:
        parts = UserFileSystem._conversation_file_parts(path)
        return "/".join(parts)

    @staticmethod
    def _artifact(row: tuple[object, ...]) -> Artifact:
        return Artifact(str(row[0]), str(row[1]), str(row[2]), int(row[3]))

    @staticmethod
    def _verify_data(artifact: Artifact, data: bytes) -> None:
        if len(data) != artifact.size or hashlib.sha256(data).hexdigest() != artifact.sha256:
            raise ValueError("stored artifact failed integrity verification")
