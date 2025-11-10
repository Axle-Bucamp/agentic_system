"""Utility client for CoinMarketCap market sentiment indicators."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from core.config import settings
from core.logging import log


class CMCIndicatorClient:
    """Fetches sentiment indicators from the CoinMarketCap public API."""

    _BASE_URL = "https://pro-api.coinmarketcap.com"

    def __init__(self, api_key: Optional[str]) -> None:
        self._api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> None:
        if not self._api_key:
            raise RuntimeError("CMC API key not configured")
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._BASE_URL,
                headers={
                    "X-CMC_PRO_API_KEY": self._api_key,
                    "Accept": "application/json",
                },
                timeout=20.0,
            )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_snapshot(self) -> Optional[Dict[str, Any]]:
        """Return aggregated sentiment snapshot if the API key is present."""

        if not self._api_key:
            return None

        await self.connect()
        assert self._client is not None

        try:
            fear_greed = await self._get_latest_value("/v3/fear-and-greed/latest")
            altcoin_season = await self._get_latest_value("/v3/altcoin-season/latest")
            market_cycle = await self._get_latest_value("/v3/market-cycle/latest")
            dominance = await self._get_latest_value("/v3/bitcoin-dominance/latest")
            cmc20 = await self._get_latest_value("/v3/cmc20/latest")
            cmc100 = await self._get_latest_value("/v3/cmc100/latest")
        except Exception as exc:  # pragma: no cover - network
            log.debug("Failed to fetch CMC indicators: %s", exc)
            return None

        snapshot = {
            "fear_greed_index": fear_greed,
            "altcoin_season_index": altcoin_season,
            "market_cycle_indicator": market_cycle,
            "bitcoin_dominance": dominance,
            "cmc20_index": cmc20,
            "cmc100_index": cmc100,
        }
        return snapshot

    async def _get_latest_value(self, endpoint: str) -> Optional[Dict[str, Any]]:
        if not self._client:
            return None
        response = await self._client.get(endpoint)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict):
            return data
        return None


async def get_indicator_snapshot() -> Optional[Dict[str, Any]]:
    """Convenience wrapper used by pipelines."""

    if not settings.cmc_api_key:
        return None

    client = CMCIndicatorClient(settings.cmc_api_key)
    try:
        return await client.get_snapshot()
    finally:
        await client.close()
