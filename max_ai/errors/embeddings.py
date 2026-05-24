"""Errors raised by lightweight embedding helpers."""

from __future__ import annotations


class EmbeddingError(Exception):
    """Raised when local embedding setup or input validation fails."""

    def __init__(self, message: str, kind: str = "generic") -> None:
        super().__init__(message)
        self.kind = kind

    @classmethod
    def dependency_missing(cls) -> "EmbeddingError":
        return cls(
            "fastembed is required for lightweight embeddings. Install "
            "qdrant-client[fastembed] or fastembed.",
            kind="dependency_missing",
        )

    @classmethod
    def invalid_text(cls) -> "EmbeddingError":
        return cls("Embedding text must be a string.", kind="invalid_text")

    @classmethod
    def invalid_texts(cls) -> "EmbeddingError":
        return cls(
            "Embedding texts must contain only strings.",
            kind="invalid_texts",
        )
