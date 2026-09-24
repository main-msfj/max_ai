"""Example tools for the CLI and self-directed agent demos."""

from pathlib import Path

from max_ai.tools import tool


@tool(approval_mode="auto_approval")
def calculate_compound_interest(principal: float, annual_rate: float, years: int) -> float:
    """Calculate a balance with annual compound interest."""
    return round(principal * (1 + annual_rate / 100) ** years, 2)


@tool(approval_mode="ask_for_approval")
def save_report(report: str) -> str:
    """Save a report to a local file after approval."""
    path = Path("./local/reports/demo_report.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return f"Report saved to {path}"


DEMO_TOOLS = [calculate_compound_interest, save_report]
