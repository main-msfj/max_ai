"""Fill template.html with a summary: render.py SUMMARY.json REPORT.html"""

import json
import sys
from html import escape
from pathlib import Path

TEMPLATE = Path(__file__).parent.parent / "template.html"


def money(value: float) -> str:
    return f"{value:,.2f}"


def rows(entries: list[dict]) -> str:
    """One line per entry, with a bar as long as its share of the largest."""
    most = max((entry["amount"] for entry in entries), default=0) or 1
    return "".join(
        f'<div class="row"><span>{escape(entry["name"])}</span>'
        f'<div class="bar" style="width:{max(2, 100 * entry["amount"] / most):.0f}%"></div>'
        f'<b>{money(entry["amount"])}</b></div>'
        for entry in sorted(entries, key=lambda entry: -entry["amount"])
    )


def main(source: str, target: str) -> None:
    summary = json.loads(Path(source).read_text())
    cards = "".join(
        f'<div class="card"><span>{escape(card["label"])}</span>'
        f'<strong>{money(card["value"])}</strong></div>'
        for card in summary["cards"]
    )
    notes = "".join(f"<li>{escape(note)}</li>" for note in summary.get("notes", []))
    parts = {
        "title": escape(summary["title"]), "business": escape(summary["business"]),
        "currency": escape(summary["currency"]), "cards": cards, "notes": notes,
        "sales": rows(summary["sales"]), "expenses": rows(summary["expenses"]),
    }
    page = TEMPLATE.read_text()
    for name, value in parts.items():
        page = page.replace("{{" + name + "}}", value)
    Path(target).write_text(page)
    print(f"Report written to {target}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Use: render.py SUMMARY.json REPORT.html")
    main(sys.argv[1], sys.argv[2])
