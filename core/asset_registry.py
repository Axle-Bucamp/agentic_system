"""Centralised asset registry shared across agents.

Provides a lightweight, in-memory catalogue of assets returned by the
Forecasting API so that all agents can work from the same dynamic list.
"""
from __future__ import annotations

import asyncio
from typing import Dict, Iterable, List, Optional

from core.config import settings
from core.logging import log

_asset_lock = asyncio.Lock()
_assets: List[str] = []
_symbol_map: Dict[str, str] = {}
_intervals_map: Dict[str, List[str]] = {}

# Reasonable defaults based on currently enabled assets exposed by the forecasting API
_FALLBACK_ASSETS: List[str] = [
    "AAVE",
    "ADA",
    "AXS",
    "BTC",
    "DAI",
    "ETH",
    "GALA",
    "IMX",
    "MANA",
    "SAND",
    "SOL",
    "SUI",
    "USDC",
    "XRP",
]


async def update_assets(
    available_tickers: Iterable[dict],
    enabled_assets: Optional[Iterable[str]] = None,
) -> None:
    """Update the shared asset registry.

    Args:
        available_tickers: Iterable of ticker payloads from the forecasting API.
        enabled_assets: Optional iterable of assets currently enabled for trading.
    """

    enabled_set = {asset.upper() for asset in (enabled_assets or [])}

    new_assets: List[str] = []
    new_symbol_map: Dict[str, str] = {}
    new_intervals_map: Dict[str, List[str]] = {}

    for ticker in available_tickers:
        symbol = ticker.get("symbol")
        if not symbol:
            continue

        if not ticker.get("has_dqn", True):
            continue

        base_symbol = symbol.replace("-USD", "").upper()
        if enabled_set and base_symbol not in enabled_set:
            continue

        intervals = ticker.get("intervals") or []

        new_assets.append(base_symbol)
        new_symbol_map[base_symbol] = symbol
        new_intervals_map[base_symbol] = intervals

    if not new_assets:
        log.debug("Asset registry update skipped – no assets returned from API")
        return

    async with _asset_lock:
        _assets.clear()
        _assets.extend(sorted(set(new_assets)))

        _symbol_map.clear()
        _symbol_map.update(new_symbol_map)

        _intervals_map.clear()
        _intervals_map.update(new_intervals_map)

    log.info("Asset registry refreshed with %d assets", len(_assets))


def get_assets() -> List[str]:
    """Return the current list of supported assets."""

    if _assets:
        return list(_assets)

    # Fall back to the static configuration until the registry is populated
    fallback = [asset for asset in settings.supported_assets if asset.upper() in _FALLBACK_ASSETS]
    return fallback or list(settings.supported_assets)


def get_symbol(ticker: str) -> str:
    """Return the API symbol (e.g. BTC-USD) for a given base ticker."""

    return _symbol_map.get(ticker.upper(), f"{ticker.upper()}-USD")


def get_intervals(ticker: str) -> List[str]:
    """Return the supported intervals for the specified ticker."""

    return _intervals_map.get(ticker.upper(), ["hours", "days"])


async def use_fallback_assets() -> None:
    """Populate the registry with a sensible static fallback set."""

    async with _asset_lock:
        _assets.clear()
        _assets.extend(_FALLBACK_ASSETS)

        _symbol_map.clear()
        _symbol_map.update({asset: f"{asset}-USD" for asset in _FALLBACK_ASSETS})

        _intervals_map.clear()
        _intervals_map.update({asset: ["hours", "days"] for asset in _FALLBACK_ASSETS})


