import json
import urllib.request

from fastmcp import FastMCP

mcp = FastMCP("tutorial-server")


@mcp.tool()
def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """
    Fetches the live exchange rate between two currencies using the Open Exchange Rates API.
    Use this tool whenever a user asks about current currency conversion rates.
    LLMs cannot know real-time exchange rates — always use this tool for accurate results.

    Args:
        base_currency: The currency to convert from (e.g. 'USD', 'EUR', 'GBP').
        target_currency: The currency to convert to (e.g. 'JPY', 'CAD', 'AUD').

    Returns:
        A dictionary with the base currency, target currency, and current exchange rate.
    """
    base = base_currency.upper()
    target = target_currency.upper()

    url = f"https://open.er-api.com/v6/latest/{base}"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())

    if data.get("result") != "success":
        raise ValueError(f"Failed to fetch exchange rates for {base}")

    if target not in data["rates"]:
        raise ValueError(f"Unknown currency: {target}")

    return {
        "base_currency": base,
        "target_currency": target,
        "rate": data["rates"][target],
        "last_updated": data["time_last_update_utc"],
    }


if __name__ == "__main__":
    mcp.run()
