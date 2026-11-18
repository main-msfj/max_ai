"""Search by words: split a text into words, rank by keywords (BM25) and
fuse several rankings (RRF). Any piece that searches may use it."""

from .ranking import bm25, fuse
from .words import words

__all__ = ["bm25", "fuse", "words"]
