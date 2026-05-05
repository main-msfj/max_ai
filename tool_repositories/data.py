"""JSON and CSV inspection tools."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any


def validate_json_text(text: str) -> dict[str, Any]:
    """Validate JSON text and return a compact structural summary."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "error": e.msg,
            "line": e.lineno,
            "column": e.colno,
        }

    if isinstance(value, dict):
        shape: Any = {"type": "object", "keys": sorted(value.keys())}
    elif isinstance(value, list):
        shape = {"type": "array", "length": len(value)}
    else:
        shape = {"type": type(value).__name__}
    return {"valid": True, "shape": shape}


def profile_csv(path: str, sample_rows: int = 20) -> dict[str, Any]:
    """Profile a CSV file without modifying it."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        row_count = 0
        non_empty: Counter[str] = Counter()
        for row in reader:
            row_count += 1
            if len(rows) < sample_rows:
                rows.append(dict(row))
            for key, value in row.items():
                if value not in (None, ""):
                    non_empty[key] += 1

    fields = list(reader.fieldnames or [])
    return {
        "path": str(file_path),
        "columns": fields,
        "row_count": row_count,
        "non_empty_counts": dict(non_empty),
        "sample_rows": rows,
    }


def csv_text_preview(text: str, max_rows: int = 10) -> dict[str, Any]:
    """Parse CSV text and return columns plus the first rows."""
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        rows.append(dict(row))
        if len(rows) >= max_rows:
            break
    return {
        "columns": list(reader.fieldnames or []),
        "rows": rows,
    }
