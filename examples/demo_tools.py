"""Four mock tools to test whether the LLM picks the right one.

Deterministic, side-effect-free stand-ins for real integrations — enough to
see the model choose tools, fill arguments, and chain them in a plan, without
any external services. Read-only tools (weather, stock price) auto-approve;
state-changing tools (book hotel, send email) ask for approval, so they also
exercise the CLI/UI approval flow.

Used by ``agent_cli.py`` and ``agent_self_directed_loop.py``.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from max_ai.tools import tool


@tool(approval_mode="auto_approval")
def get_weather(city: str, unit: str = "celsius") -> dict[str, str | int]:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. "Tokyo".
        unit: "celsius" or "fahrenheit".
    """
    city_name = city.strip() or "unknown"
    unit_name = unit.strip().lower()
    if unit_name not in {"celsius", "fahrenheit"}:
        unit_name = "celsius"

    seed = sum(ord(c) for c in city_name.lower())
    condition = ["sunny", "cloudy", "rainy", "windy", "clear"][seed % 5]
    celsius = 12 + seed % 18
    temperature = celsius if unit_name == "celsius" else round(celsius * 9 / 5 + 32)

    return {
        "city": city_name,
        "condition": condition,
        "temperature": temperature,
        "unit": unit_name,
    }


@tool(approval_mode="auto_approval")
def get_stock_price(ticker: str) -> dict[str, str | float]:
    """Get the latest stock price for a ticker symbol.

    Args:
        ticker: Stock ticker, e.g. "NVDA" or "GOOGL".
    """
    symbol = ticker.strip().upper() or "UNKNOWN"
    seed = sum(ord(c) for c in symbol)
    price = round(50 + (seed % 950) + random.random(), 2)
    change_pct = round(((seed % 21) - 10) / 2, 2)  # -5.0%..+5.0%
    return {
        "ticker": symbol,
        "price_usd": price,
        "change_pct": change_pct,
        "currency": "USD",
    }


@tool(approval_mode="ask_for_approval")
def book_hotel(city: str, check_in: str, nights: int = 1, guests: int = 1) -> dict[str, str | int]:
    """Book a hotel room. This reserves a room and is a state-changing action.

    Args:
        city: City to book in.
        check_in: Check-in date, YYYY-MM-DD.
        nights: Number of nights.
        guests: Number of guests.
    """
    confirmation = f"BK-{random.randint(100000, 999999)}"
    try:
        start = date.fromisoformat(check_in.strip())
        check_out = (start + timedelta(days=max(1, nights))).isoformat()
    except ValueError:
        check_out = "unknown"
    return {
        "status": "confirmed",
        "confirmation": confirmation,
        "city": city.strip(),
        "check_in": check_in.strip(),
        "check_out": check_out,
        "nights": max(1, nights),
        "guests": max(1, guests),
    }


@tool(approval_mode="ask_for_approval")
def send_email(to: str, subject: str, body: str) -> dict[str, str]:
    """Send an email. This sends a real message and is a state-changing action.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text.
    """
    message_id = f"<{random.randint(10**9, 10**10)}@max-ai.demo>"
    return {
        "status": "sent",
        "message_id": message_id,
        "to": to.strip(),
        "subject": subject.strip(),
        "chars": str(len(body)),
    }


# Convenience: the full toolset to hand to an agent.
DEMO_TOOLS = [get_weather, get_stock_price, book_hotel, send_email]
