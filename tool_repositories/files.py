"""Read-only filesystem inspection tools."""

from __future__ import annotations

from pathlib import Path


def list_directory(path: str = ".", max_entries: int = 200) -> list[dict[str, str | int]]:
    """List files and directories directly under a path.

    Args:
        path: Directory to inspect.
        max_entries: Maximum number of entries to return.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    entries: list[dict[str, str | int]] = []
    for child in sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if len(entries) >= max_entries:
            break
        stat = child.stat()
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "kind": "directory" if child.is_dir() else "file",
                "size_bytes": stat.st_size,
            }
        )
    return entries


def directory_tree(path: str = ".", max_depth: int = 2, max_entries: int = 300) -> list[str]:
    """Return a compact directory tree.

    Args:
        path: Directory to inspect.
        max_depth: Maximum folder depth from the root.
        max_entries: Maximum tree lines to return.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    lines = [root.name + "/"]

    def walk(current: Path, depth: int, prefix: str) -> None:
        if depth >= max_depth or len(lines) >= max_entries:
            return
        children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        for index, child in enumerate(children):
            if len(lines) >= max_entries:
                return
            branch = "`-- " if index == len(children) - 1 else "|-- "
            suffix = "/" if child.is_dir() else ""
            lines.append(prefix + branch + child.name + suffix)
            if child.is_dir():
                next_prefix = prefix + ("    " if index == len(children) - 1 else "|   ")
                walk(child, depth + 1, next_prefix)

    walk(root, 0, "")
    return lines


def read_text_file(path: str, max_chars: int = 12000) -> dict[str, str | int | bool]:
    """Read a text file with a size cap.

    Args:
        path: File to read.
        max_chars: Maximum characters returned.
    """
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = file_path.read_text(encoding="utf-8", errors="replace")
    truncated = len(text) > max_chars
    return {
        "path": str(file_path),
        "content": text[:max_chars],
        "chars": min(len(text), max_chars),
        "truncated": truncated,
    }


def search_text(
    path: str,
    query: str,
    glob: str = "**/*",
    max_matches: int = 50,
    max_file_bytes: int = 1_000_000,
) -> list[dict[str, str | int]]:
    """Search text files for a literal query.

    Args:
        path: Root directory to search.
        query: Literal text to find.
        glob: File glob under path.
        max_matches: Maximum matches returned.
        max_file_bytes: Skip files larger than this many bytes.
    """
    if not query:
        raise ValueError("query cannot be empty")

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    matches: list[dict[str, str | int]] = []
    for file_path in root.glob(glob):
        if len(matches) >= max_matches:
            break
        if not file_path.is_file():
            continue
        if file_path.stat().st_size > max_file_bytes:
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if query in line:
                matches.append(
                    {
                        "path": str(file_path),
                        "line": line_no,
                        "text": line.strip(),
                    }
                )
                if len(matches) >= max_matches:
                    break
    return matches

