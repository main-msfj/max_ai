"""Diff analysis utilities."""

import typing as t


def analyze_diff(diff_text: str) -> dict[str, t.Any]:
    """Analyze a unified diff and return basic statistics.

    Counts added/removed lines, files touched, and detects simple
    complexity signals like deeply nested changes or large blocks.

    Args:
        diff_text: The full unified diff text from `git diff` or a
            PR description.

    Returns:
        A dict with keys: lines_added, lines_removed, files_touched,
        complexity_signals (list of strings).
    """
    lines = diff_text.splitlines()
    added = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
    files = sum(1 for ln in lines if ln.startswith("+++ b/"))

    signals: list[str] = []
    if added > 500:
        signals.append("large_addition")
    if removed > 500:
        signals.append("large_deletion")
    if files > 20:
        signals.append("many_files_touched")
    if any("TODO" in ln or "FIXME" in ln for ln in lines if ln.startswith("+")):
        signals.append("introduces_todo_or_fixme")

    return {
        "lines_added": added,
        "lines_removed": removed,
        "files_touched": files,
        "complexity_signals": signals,
    }