"""Runs inside the sandbox: reads one file for read_file. Standard library only."""

import hashlib
import importlib
import json
import os
import sys
from collections.abc import Callable
from typing import Any


def inside(file: str, folders: list[str]) -> bool:
    """``file`` (links resolved) is in one of ``folders``."""
    return any(file.startswith(os.path.realpath(folder) + os.sep) for folder in folders)


def text(file: str) -> list[str]:
    """Code or any text file, line by line. A file with null bytes is not text."""
    with open(file, "rb") as handle:
        content = handle.read()
    if b"\0" in content[:8192]:
        raise ValueError("it is not a text file")
    return content.decode("utf-8", errors="replace").splitlines()


def notebook(file: str) -> list[str]:
    """A Jupyter notebook as text: each cell with its number, kind and outputs."""
    with open(file, encoding="utf-8") as handle:
        cells: list[dict[str, Any]] = json.load(handle).get("cells", [])
    lines: list[str] = []
    for number, cell in enumerate(cells, 1):
        lines.append(f"--- cell {number} ({cell.get('cell_type', '?')}) ---")
        lines += "".join(cell.get("source", [])).splitlines()
        for output in cell.get("outputs", []):
            shown = output.get("text") or output.get("data", {}).get("text/plain", [])
            if shown:
                lines.append("[output]")
                lines += "".join(shown).splitlines()
    return lines


def pdf(file: str) -> list[str]:
    """The text of a PDF, page by page. Needs pypdf in the sandbox."""
    try:
        pypdf: Any = importlib.import_module("pypdf")
    except ImportError:
        raise ValueError("reading a PDF needs the pypdf package, which is not installed")
    lines: list[str] = []
    reader: Any = pypdf.PdfReader(file)
    for number, page in enumerate(reader.pages, 1):
        lines.append(f"--- page {number} ---")
        lines += (page.extract_text() or "").splitlines()
    return lines


# By extension; anything else is read as text.
READERS: dict[str, Callable[[str], list[str]]] = {".ipynb": notebook, ".pdf": pdf}


def read(request: dict[str, Any]) -> dict[str, Any]:
    # 1. Resolve the path from the working directory, links included.
    cwd: str = request["cwd"]
    given: str = request["path"]
    file = os.path.realpath(os.path.join(cwd, given))
    if not inside(file, request["allowed"]):
        folders = ", ".join(request["allowed"])
        raise ValueError(
            f"it resolves to {file}, which is outside the folders you can read "
            f"({folders}); use a full path inside one of them"
        )
    if not os.path.exists(file):
        raise ValueError("it does not exist")
    if os.path.isdir(file):
        raise ValueError("it is a folder, not a file")

    # 2. Turn the file into lines: by its extension, or as plain text.
    lines = READERS.get(os.path.splitext(file)[1].lower(), text)(file)

    # 3. Give back only the lines asked for, numbered and cut to the width.
    first: int = request["first"]
    last: int = first + request["limit"] - 1
    wanted = lines[first - 1:last]
    numbered = [f"{first + i:>6}\t{line[:request['width']]}" for i, line in enumerate(wanted)]

    # 4. The hash of the whole file: what the model has now seen.
    with open(file, "rb") as handle:
        seen = hashlib.sha256(handle.read()).hexdigest()
    return {"total": len(lines), "lines": "\n".join(numbered), "sha256": seen}


def main() -> None:
    try:
        answer: dict[str, Any] = {"ok": True, **read(json.loads(sys.argv[1]))}
    except (ValueError, OSError) as error:
        answer = {"ok": False, "error": str(error)}
    print(json.dumps(answer))


if __name__ == "__main__":
    main()
