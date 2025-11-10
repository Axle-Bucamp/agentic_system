"""Centralised asset registry shared across agents.

Provides a lightweight, in-memory catalogue of assets returned by the
Forecasting API so that all agents can work from the same dynamic list.
"""
from __future__ import annotations

import asyncio
from typing import Dict, Iterable, List, Optional, Set

from core.config import settings
from core.logging import log

_asset_lock = asyncio.Lock()
_assets: List[str] = []
_symbol_map: Dict[str, str] = {}
_intervals_map: Dict[str, List[str]] = {}
_disabled_assets: Set[str] = set()

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

_REFUGE_ASSETS: Set[str] = {"DAI"}


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
    reenabled_assets: Set[str] = set()

    for ticker in available_tickers:
        symbol = ticker.get("symbol")
        if not symbol:
            continue

        if not ticker.get("has_dqn", True):
            continue

        base_symbol = symbol.replace("-USD", "").upper()
        if base_symbol in _disabled_assets:
            if enabled_set and base_symbol in enabled_set:
                reenabled_assets.add(base_symbol)
            else:
                continue
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
        for asset in reenabled_assets:
            _disabled_assets.discard(asset)

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
        return [asset for asset in _assets if asset not in _disabled_assets]

    # Fall back to the static configuration until the registry is populated
    fallback = [asset for asset in settings.supported_assets if asset.upper() in _FALLBACK_ASSETS and asset.upper() not in _disabled_assets]

    for refuge in _REFUGE_ASSETS:
        if refuge not in fallback and refuge not in _disabled_assets:
            fallback.append(refuge)

    # Ensure we always return unique, upper-case sorted assets for consistency
    unique_fallback = sorted({asset.upper() for asset in fallback})
    return unique_fallback or [asset.upper() for asset in settings.supported_assets if asset.upper() not in _disabled_assets]


def get_symbol(ticker: str) -> str:
    """Return the API symbol (e.g. BTC-USD) for a given base ticker."""

    ticker_upper = ticker.upper()
    # If the incoming ticker already looks like a fully-qualified symbol, trust it.
    if "-" in ticker_upper:
        return ticker_upper

    return _symbol_map.get(ticker_upper, f"{ticker_upper}-USD")


def get_intervals(ticker: str) -> List[str]:
    """Return the supported intervals for the specified ticker."""

    return _intervals_map.get(ticker.upper(), ["hours", "days"])


async def disable_asset(ticker: str) -> None:
    """Temporarily disable an asset that is unavailable from external APIs."""

    ticker_upper = ticker.upper()
    if "-" in ticker_upper:
        ticker_upper = ticker_upper.split("-")[0]

    async with _asset_lock:
        if ticker_upper in _disabled_assets:
            return

        _disabled_assets.add(ticker_upper)
        if ticker_upper in _assets:
            _assets.remove(ticker_upper)
        _symbol_map.pop(ticker_upper, None)
        _intervals_map.pop(ticker_upper, None)


async def use_fallback_assets() -> None:
    """Populate the registry with a sensible static fallback set."""

    async with _asset_lock:
        _assets.clear()
        _assets.extend(asset for asset in _FALLBACK_ASSETS if asset not in _disabled_assets)

        _symbol_map.clear()
        _symbol_map.update(
            {asset: f"{asset}-USD" for asset in _FALLBACK_ASSETS if asset not in _disabled_assets}
        )

        _intervals_map.clear()
        _intervals_map.update(
            {asset: ["hours", "days"] for asset in _FALLBACK_ASSETS if asset not in _disabled_assets}
        )


def reset_disabled_assets() -> None:
    """Clear the disabled asset cache (primarily for tests)."""

    _disabled_assets.clear()


