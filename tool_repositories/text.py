"""Text and markdown analysis tools."""

from __future__ import annotations

import difflib
import re


def extract_markdown_outline(markdown: str, max_headings: int = 100) -> list[dict[str, str | int]]:
    """Extract headings from markdown text.

    Args:
        markdown: Markdown content.
        max_headings: Maximum headings returned.
    """
    headings: list[dict[str, str | int]] = []
    for line_no, line in enumerate(markdown.splitlines(), start=1):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue
        headings.append(
            {
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
                "line": line_no,
            }
        )
        if len(headings) >= max_headings:
            break
    return headings


def summarize_text_stats(text: str) -> dict[str, int]:
    """Return basic text statistics."""
    words = re.findall(r"\b[\w'-]+\b", text)
    sentences = re.findall(r"[^.!?]+[.!?]", text)
    lines = text.splitlines()
    return {
        "characters": len(text),
        "lines": len(lines),
        "words": len(words),
        "sentences": len(sentences),
        "approx_tokens": max(1, len(text) // 4) if text else 0,
    }


def extract_urls(text: str, max_urls: int = 100) -> list[str]:
    """Extract URLs from text."""
    pattern = re.compile(r"https?://[^\s<>)\"']+")
    urls: list[str] = []
    for match in pattern.finditer(text):
        urls.append(match.group(0).rstrip(".,;:!?"))
        if len(urls) >= max_urls:
            break
    return urls


def unified_diff(old_text: str, new_text: str, fromfile: str = "old", tofile: str = "new") -> str:
    """Return a unified diff between two text values."""
    return "\n".join(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    )

