"""Tools that can run in the Docker execution environment."""

from pathlib import Path

from max_ai.tools import tool


@tool(approval_mode="auto_approval")
def calculate_compound_interest(principal: float, annual_rate: float, years: int) -> float:
    """Calculate a balance with annual compound interest."""
    return round(principal * (1 + annual_rate / 100) ** years, 2)


@tool(approval_mode="ask_for_approval")
def save_report(report: str) -> str:
    """Save a report in the execution workspace after approval."""
    path = Path("./reports/report.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return f"Report saved to {path}"
