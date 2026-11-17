"""Runs inside the sandbox: reads one file for read_file. Standard library only."""

import hashlib
import json
import os
import sys


def inside(file, folders):
    """``file`` (links resolved) is in one of ``folders``."""
    return any(file.startswith(os.path.realpath(folder) + os.sep) for folder in folders)


def text(file):
    """Code or any text file, line by line. A file with null bytes is not text."""
    with open(file, "rb") as handle:
        content = handle.read()
    if b"\0" in content[:8192]:
        raise ValueError("it is not a text file")
    return content.decode("utf-8", errors="replace").splitlines()


def notebook(file):
    """A Jupyter notebook as text: each cell with its number, kind and outputs."""
    with open(file, encoding="utf-8") as handle:
        cells = json.load(handle).get("cells", [])
    lines = []
    for number, cell in enumerate(cells, 1):
        lines.append(f"--- cell {number} ({cell.get('cell_type', '?')}) ---")
        lines += "".join(cell.get("source", [])).splitlines()
        for output in cell.get("outputs", []):
            shown = output.get("text") or output.get("data", {}).get("text/plain", [])
            if shown:
                lines.append("[output]")
                lines += "".join(shown).splitlines()
    return lines


def pdf(file):
    """The text of a PDF, page by page. Needs pypdf in the sandbox."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ValueError("reading a PDF needs the pypdf package, which is not installed")
    lines = []
    for number, page in enumerate(PdfReader(file).pages, 1):
        lines.append(f"--- page {number} ---")
        lines += (page.extract_text() or "").splitlines()
    return lines


# By extension; anything else is read as text.
READERS = {".ipynb": notebook, ".pdf": pdf}


def read(request):
    # 1. Resolve the path from the working directory, links included.
    file = os.path.realpath(os.path.join(request["cwd"], request["path"]))
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
    first, last = request["first"], request["first"] + request["limit"] - 1
    wanted = lines[first - 1:last]
    numbered = [f"{first + i:>6}\t{line[:request['width']]}" for i, line in enumerate(wanted)]

    # 4. The hash of the whole file: what the model has now seen.
    with open(file, "rb") as handle:
        seen = hashlib.sha256(handle.read()).hexdigest()
    return {"total": len(lines), "lines": "\n".join(numbered), "sha256": seen}


def main():
    try:
        answer = {"ok": True, **read(json.loads(sys.argv[1]))}
    except (ValueError, OSError) as error:
        answer = {"ok": False, "error": str(error)}
    print(json.dumps(answer))


if __name__ == "__main__":
    main()
