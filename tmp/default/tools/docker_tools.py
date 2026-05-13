from max_ai.tools import tool


@tool(name="extract_action", description="Extract partical action item from shote meeting notes")
def extract_action_items(notes: str) -> list[dict[str, str]]:
    """Extract practical action items from short meeting notes."""
    items: list[dict[str, str]] = []

    for raw_line in notes.splitlines():
        line = raw_line.strip(" -\t")
        if not line:
            continue

        lower = line.lower()
        if not any(
            marker in lower
            for marker in ("todo", "action", "next", "follow up", "owner:")
        ):
            continue

        owner = "unassigned"
        if "owner:" in lower:
            before, after = line.split("owner:", 1)
            owner = after.split(",", 1)[0].strip() or "unassigned"
            line = before.strip(" -,:") or line

        items.append(
            {
                "task": line,
                "owner": owner,
            }
        )

    return items


@tool
def readability_snapshot(text: str) -> dict[str, float | int]:
    """Return a small readability snapshot for draft text."""
    sentences = [
        part.strip()
        for part in text.replace("!", ".").replace("?", ".").split(".")
        if part.strip()
    ]
    return {
        "sentences": len(sentences),
        "words": len([word for word in text.split() if word]),
        "characters": len(text),
        "average_words_per_sentence": round(
            len([word for word in text.split() if word]) / max(len(sentences), 1),
            2,
        ),
    }


@tool
def project_health_score(
    blocked_tasks: int,
    open_tasks: int,
    completed_tasks: int,
    days_until_deadline: int,
) -> dict[str, int | str]:
    """Estimate a simple project health label from task status."""
    total_tasks = max(open_tasks + completed_tasks, 1)
    completion = completed_tasks / total_tasks
    risk = blocked_tasks * 18 + max(0, 7 - days_until_deadline) * 8
    score = max(0, min(100, round(completion * 100 - risk)))

    if score >= 75:
        label = "healthy"
    elif score >= 45:
        label = "watch"
    else:
        label = "at_risk"

    return {
        "score": score,
        "label": label,
        "blocked_tasks": blocked_tasks,
        "days_until_deadline": days_until_deadline,
    }


@tool
def make_decision_matrix(
    options: list[str],
    criteria: list[str],
    scores: dict[str, dict[str, int]],
) -> dict[str, object]:
    """Rank options from a small criteria score matrix."""
    ranked: list[dict[str, object]] = []
    for option in options:
        total = 0
        details: dict[str, int] = {}
        for criterion in criteria:
            value = int(scores.get(option, {}).get(criterion, 0))
            details[criterion] = value
            total += value
        ranked.append({"option": option, "score": total, "scores": details})

    ranked.sort(key=lambda item: int(item["score"]), reverse=True)
    return {
        "winner": ranked[0]["option"] if ranked else None,
        "ranked": ranked,
    }
