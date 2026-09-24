"""Store vectors as compact bytes (float32), e.g. in a database column."""

from __future__ import annotations

from array import array


def pack_vector(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def unpack_vector(data: bytes) -> list[float]:
    values = array("f")
    values.frombytes(data)
    return values.tolist()
