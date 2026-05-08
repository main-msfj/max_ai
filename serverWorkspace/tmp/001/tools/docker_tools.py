from max_ai.base.tools import ToolContext
from max_ai.tools import tool


@tool
def add(a: int, b: int) -> int:
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    return a * b


@tool
def slugify(text: str) -> str:
    cleaned = text.strip().lower()
    pieces = []
    last_was_dash = False

    for char in cleaned:
        if char.isalnum():
            pieces.append(char)
            last_was_dash = False
        elif not last_was_dash:
            pieces.append("-")
            last_was_dash = True

    return "".join(pieces).strip("-")


@tool
def word_count(text: str) -> dict[str, int]:
    words = [word for word in text.split() if word]
    return {
        "words": len(words),
        "characters": len(text),
    }


@tool
def summarize_numbers(numbers: list[float]) -> dict[str, float | int | None]:
    if not numbers:
        return {
            "count": 0,
            "total": 0,
            "min": None,
            "max": None,
            "average": None,
        }

    total = sum(numbers)
    return {
        "count": len(numbers),
        "total": total,
        "min": min(numbers),
        "max": max(numbers),
        "average": total / len(numbers),
    }


@tool
def session_echo(context: ToolContext, message: str) -> dict[str, str | int]:
    return {
        "message": message,
        "run_id": context.run_id,
        "session_id": context.session_id,
        "retry_count": context.retry_count,
    }
