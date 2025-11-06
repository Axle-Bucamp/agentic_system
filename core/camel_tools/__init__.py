"""
CAMEL-compatible tools for trading system operations.
"""
from core.camel_tools.mcp_forecasting_toolkit import MCPForecastingToolkit
from core.camel_tools.dex_trading_toolkit import DEXTradingToolkit
from core.camel_tools.market_data_toolkit import MarketDataToolkit
from core.camel_tools.crypto_tools import CryptoTools

__all__ = [
    "MCPForecastingToolkit",
    "DEXTradingToolkit",
    "MarketDataToolkit",
    "CryptoTools",
]

