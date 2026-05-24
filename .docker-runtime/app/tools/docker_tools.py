
import random
from max_ai.tools import tool


@tool(approval_mode="ask_for_approval")
def get_weather(city: str, unit: str = "celsius") -> dict[str, str | int]:
    """Return a deterministic mock weather report for a city."""
    city_name = city.strip() or "unknown"
    unit_name = unit.strip().lower()
    if unit_name not in {"celsius", "fahrenheit"}:
        unit_name = "celsius"

    seed = sum(ord(char) for char in city_name.lower())
    condition = ["sunny", "cloudy", "rainy", "windy", "clear"][seed % 5]
    celsius = 12 + seed % 18
    temperature = celsius if unit_name == "celsius" else round(celsius * 9 / 5 + 32)

    return {
        "city": city_name,
        "condition": condition,
        "temperature": temperature,
        "unit": unit_name,
    }


@tool(approval_mode="ask_for_approval")
def check_link(link:str):
    """Return the link status"""
    status = "Available" if bool(random.randint(0,1)) else "Offline"
    return f"The provided link is {status}"
