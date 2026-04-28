"""Bug triage helpers."""

import re
import typing as t


_SEVERITY_KEYWORDS: dict[str, list[str]] = {
    "critical": [
        "data loss", "data corruption", "security", "breach", "leak",
        "all users", "production down", "cannot login", "payment fail",
    ],
    "high": [
        "crash", "error 500", "broken", "blocking", "cannot",
        "many users", "majority", "regression",
    ],
    "medium": [
        "slow", "intermittent", "occasional", "sometimes", "edge case",
        "minor", "workaround",
    ],
    "low": [
        "typo", "ui glitch", "cosmetic", "wording", "alignment",
        "color", "spacing",
    ],
}


def extract_severity_signals(bug_text: str) -> list[str]:
    """Extract severity-indicating phrases from a bug report.

    Returns the list of keywords found, with their severity bucket
    prefixed (e.g. "critical:data loss"). Use the result to inform
    your overall severity judgment — multiple signals across buckets
    means you should reason about which dominate.

    Args:
        bug_text: The full text of the bug report or user complaint.

    Returns:
        A list of strings formatted as "severity:keyword". Empty if
        no known signals were found.
    """
    text = bug_text.lower()
    found: list[str] = []
    for severity, keywords in _SEVERITY_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                found.append(f"{severity}:{kw}")
    return found


def suggest_priority(severity: str, frequency: str) -> str:
    """Suggest a priority bucket from severity and frequency.

    Combines severity (critical/high/medium/low) with frequency
    (always/often/sometimes/rare) into a P0–P3 recommendation.

    Args:
        severity: One of "critical", "high", "medium", "low".
        frequency: One of "always", "often", "sometimes", "rare".

    Returns:
        One of "P0", "P1", "P2", "P3", with a one-line rationale.
    """
    sev = severity.lower().strip()
    freq = frequency.lower().strip()

    if sev not in {"critical", "high", "medium", "low"}:
        return f"unknown severity {severity!r}; expected critical/high/medium/low"
    if freq not in {"always", "often", "sometimes", "rare"}:
        return f"unknown frequency {frequency!r}; expected always/often/sometimes/rare"

    if sev == "critical":
        return "P0 — critical severity: address immediately regardless of frequency"
    if sev == "high":
        if freq in ("always", "often"):
            return "P1 — high severity hitting users frequently"
        return "P2 — high severity but limited frequency"
    if sev == "medium":
        if freq == "always":
            return "P2 — medium severity, consistent occurrence"
        return "P3 — medium severity, intermittent"
    return "P3 — low severity, schedule for routine cleanup"