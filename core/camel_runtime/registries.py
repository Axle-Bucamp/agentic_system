"""
Toolkit registry for CAMEL runtime.

This module exposes a registry that builds the FunctionTool collections
used by CAMEL ChatAgents and Workforces.  The registry de-duplicates tool
construction and keeps the mapping between logical capabilities and the
underlying service clients (forecasting MCP, DEX simulator, research
sources, etc.).
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Awaitable, Callable, Dict, List, Optional

from camel.toolkits import FunctionTool

from core.config import settings
from core.forecasting_client import ForecastingClient
from core.dex_simulator_client import DEXSimulatorClient
from core.logging import log
from core import asset_registry
from core.telemetry.guidry_stats import guidry_cloud_stats
from core.camel_tools.market_data_toolkit import MarketDataToolkit
from core.camel_tools.mcp_forecasting_toolkit import MCPForecastingToolkit
from core.camel_tools.playwright_toolkit import PlaywrightToolkit


AsyncFn = Callable[..., Awaitable[Any]]


class ToolkitRegistry:
    """
    Build and cache FunctionTool instances for CAMEL agents.

    The registry lazily initialises shared service clients (Forecasting MCP,
    DEX simulator, research APIs) and wraps the high level async functions as
    CAMEL `FunctionTool` objects as required by the official tooling
    interface.  This matches the guidance from the CAMEL docs that tools
    should be provided as `FunctionTool` instances rather than raw callables
    so that argument schemas can be inferred automatically.
    """

    def __init__(self) -> None:
        self._forecasting_client: Optional[ForecastingClient] = None
        self._dex_client: Optional[DEXSimulatorClient] = None
        self._tool_cache: Dict[str, FunctionTool] = {}
        self._lock = asyncio.Lock()
        self._playwright_toolkit: Optional[PlaywrightToolkit] = None

    async def ensure_clients(self) -> None:
        """Initialise shared service clients once."""
        async with self._lock:
            if self._forecasting_client is None:
                self._forecasting_client = ForecastingClient(
                    {
                        "base_url": settings.mcp_api_url,
                        "api_key": settings.mcp_api_key,
                        "mock_mode": settings.use_mock_services,
                    }
                )
                await self._forecasting_client.connect()
                log.info("Toolkit registry connected Forecasting MCP client")

            if self._dex_client is None:
                self._dex_client = DEXSimulatorClient()
                await self._dex_client.connect()
                log.info("Toolkit registry connected DEX simulator client")

            if self._playwright_toolkit is None:
                self._playwright_toolkit = PlaywrightToolkit()
                log.info("Toolkit registry prepared Playwright toolkit")

    async def get_default_toolset(self) -> List[FunctionTool]:
        """
        Return the default toolkit collection for the trading workforce.

        The returned list always contains fresh FunctionTool wrappers so that
        CAMEL can introspect parameter metadata, but the underlying async
        functions reuse shared clients.
        """
        await self.ensure_clients()

        tools = [
            self._tool("get_stock_forecast", self._tool_get_stock_forecast),
            self._tool("get_action_recommendation", self._tool_get_action_recommendation),
            self._tool("list_supported_assets", self._tool_list_supported_assets),
            self._tool("get_ohlc", self._tool_get_ohlc),
            self._tool("get_model_metrics", self._tool_get_model_metrics),
            self._tool("dex_buy_asset", self._tool_dex_buy_asset),
            self._tool("dex_sell_asset", self._tool_dex_sell_asset),
            self._tool("dex_get_portfolio", self._tool_dex_get_portfolio),
            self._tool("get_guidry_cloud_api_stats", self._tool_get_guidry_stats),
            self._tool("browse_url", self._tool_browse_url),
        ]
        return tools

    # ------------------------------------------------------------------
    # Direct helpers for non-CAMEL callers (legacy agents/pipelines)
    # ------------------------------------------------------------------

    async def get_stock_forecast(self, ticker: str, interval: str) -> Dict[str, Any]:
        await self.ensure_clients()
        return await self._tool_get_stock_forecast(ticker, interval)

    async def get_action_recommendation(self, ticker: str, interval: str) -> Dict[str, Any]:
        await self.ensure_clients()
        return await self._tool_get_action_recommendation(ticker, interval)

    async def list_supported_assets(self) -> Dict[str, Any]:
        await self.ensure_clients()
        return await self._tool_list_supported_assets()

    async def get_ohlc(self, ticker: str, interval: str, limit: int = 200) -> Dict[str, Any]:
        await self.ensure_clients()
        return await self._tool_get_ohlc(ticker, interval, limit)

    async def get_model_metrics(self, ticker: str, interval: str) -> Dict[str, Any]:
        await self.ensure_clients()
        return await self._tool_get_model_metrics(ticker, interval)

    def _tool(self, name: str, fn: AsyncFn) -> FunctionTool:
        """
        Wrap an async function into a FunctionTool, caching the schema.

        CAMEL's FunctionTool handles the automatic argument inspection and
        JSON schema generation.  The registry memoises wrappers so repeated
        requests do not rebuild the schema.
        """
        if name in self._tool_cache:
            return self._tool_cache[name]

        try:
            tool = FunctionTool(fn)
        except TypeError as exc:
            log.error("Failed to wrap tool %s: %s", name, exc)
            raise

        # Normalise schema/name so downstream agents see stable identifiers.
        try:
            schema = tool.get_openai_tool_schema()
        except Exception as exc:  # pragma: no cover - defensive logging
            log.warning("Unable to fetch schema for tool %s: %s", name, exc)
            schema = None

        if schema:
            schema = dict(schema)
            function_schema = dict(schema.get("function", {}))
            function_schema["name"] = name
            schema["function"] = function_schema
            tool.openai_tool_schema = schema

        if hasattr(tool, "name"):
            try:
                setattr(tool, "name", name)
            except Exception:
                pass

        self._tool_cache[name] = tool
        return tool

    # ---------------------------------------------------------------------
    # Forecasting helpers
    # ---------------------------------------------------------------------

    async def _tool_get_stock_forecast(self, ticker: str, interval: str) -> Dict[str, Any]:
        """
        Fetch OHLC / forecast data for a ticker & interval via the MCP client.
        """
        if not self._forecasting_client:
            raise RuntimeError("Forecasting client is not initialised")

        symbol = asset_registry.get_symbol(ticker.upper())
        result = await self._forecasting_client.get_stock_forecast(symbol, interval)
        return {"success": True, "ticker": ticker, "interval": interval, "forecast": result}

    async def _tool_get_action_recommendation(self, ticker: str, interval: str) -> Dict[str, Any]:
        """
        Fetch DQN action and confidence for the supplied ticker/interval.
        """
        if not self._forecasting_client:
            raise RuntimeError("Forecasting client is not initialised")

        symbol = asset_registry.get_symbol(ticker.upper())
        result = await self._forecasting_client.get_action_recommendation(symbol, interval)
        return {"success": True, "ticker": ticker, "interval": interval, "action": result}

    async def _tool_list_supported_assets(self) -> Dict[str, Any]:
        """
        List available asset metadata from the forecasting service.
        """
        if not self._forecasting_client:
            raise RuntimeError("Forecasting client is not initialised")

        assets = await self._forecasting_client.get_available_tickers()
        enabled = await self._forecasting_client.get_enabled_assets()
        await asset_registry.update_assets(assets, enabled)
        return {"success": True, "assets": asset_registry.get_assets()}

    async def _tool_get_ohlc(self, ticker: str, interval: str, limit: int = 200) -> Dict[str, Any]:
        """
        Retrieve OHLC candles for the supplied ticker/interval.
        """
        if not self._forecasting_client:
            raise RuntimeError("Forecasting client is not initialised")

        symbol = asset_registry.get_symbol(ticker.upper())
        candles = await self._forecasting_client.get_ohlc(symbol, interval, limit=limit)
        return {"success": True, "ticker": ticker, "interval": interval, "candles": candles}

    async def _tool_get_model_metrics(self, ticker: str, interval: str) -> Dict[str, Any]:
        """
        Retrieve model diagnostics (Sharpe, accuracy, etc.) for a ticker.
        """
        if not self._forecasting_client:
            raise RuntimeError("Forecasting client is not initialised")

        symbol = asset_registry.get_symbol(ticker.upper())
        metrics = await self._forecasting_client.get_model_metrics(symbol, interval)
        return {"success": True, "ticker": ticker, "interval": interval, "metrics": metrics}

    # ---------------------------------------------------------------------
    # DEX helpers
    # ---------------------------------------------------------------------

    async def _tool_dex_buy_asset(self, ticker: str, amount_usdc: float) -> Dict[str, Any]:
        """
        Execute a simulated buy via the DEX client.
        """
        if not self._dex_client:
            raise RuntimeError("DEX client is not initialised")
        result = await self._dex_client.buy_asset(ticker.upper(), amount_usdc)
        return {"success": result.get("success", False), "trade": result}

    async def _tool_dex_sell_asset(self, ticker: str, amount: float) -> Dict[str, Any]:
        """Execute a simulated sell via the DEX client."""
        if not self._dex_client:
            raise RuntimeError("DEX client is not initialised")
        result = await self._dex_client.sell_asset(ticker.upper(), amount)
        return {"success": result.get("success", False), "trade": result}

    async def _tool_dex_get_portfolio(self) -> Dict[str, Any]:
        """Query the current simulated portfolio state."""
        if not self._dex_client:
            raise RuntimeError("DEX client is not initialised")
        portfolio = await self._dex_client.get_portfolio_status()
        return {"success": True, "portfolio": portfolio}

    async def _tool_get_guidry_stats(self) -> Dict[str, Any]:
        """Return aggregated telemetry for the forecasting API."""
        return {"success": True, "stats": guidry_cloud_stats.summary()}

    async def _tool_browse_url(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        timeout_seconds: float = 15.0,
        max_characters: int = 2000,
    ) -> Dict[str, Any]:
        """Navigate to a URL via Playwright and return rendered snippet."""
        if not self._playwright_toolkit:
            raise RuntimeError("Playwright toolkit is not initialised")
        return await self._playwright_toolkit.browse_url(
            url=url,
            wait_for_selector=wait_for_selector,
            timeout_seconds=timeout_seconds,
            max_characters=max_characters,
        )


# shared singleton to avoid repeated initialisation
toolkit_registry = ToolkitRegistry()


async def build_default_tools() -> List[FunctionTool]:
    """Convenience helper mirroring ToolkitRegistry.get_default_toolset."""
    return await toolkit_registry.get_default_toolset()

