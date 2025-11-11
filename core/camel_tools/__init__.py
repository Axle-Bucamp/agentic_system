"""
CAMEL-compatible tools for trading system operations.
"""
from core.camel_tools.mcp_forecasting_toolkit import MCPForecastingToolkit
from core.camel_tools.blockscout_toolkit import BlockscoutMCPToolkit, get_blockscout_toolkit
from core.camel_tools.dex_trading_toolkit import DEXTradingToolkit
from core.camel_tools.market_data_toolkit import MarketDataToolkit
from core.camel_tools.crypto_tools import CryptoTools
from core.camel_tools.guidry_stats_toolkit import GuidryStatsToolkit
from core.camel_tools.review_pipeline_toolkit import ReviewPipelineToolkit

try:
    from core.camel_tools.asknews_toolkit import AskNewsToolkit
except ImportError:  # pragma: no cover
    AskNewsToolkit = None  # type: ignore

try:
    from core.camel_tools.google_research_toolkit import GoogleResearchToolkit
except ImportError:  # pragma: no cover
    GoogleResearchToolkit = None  # type: ignore

__all__ = [
    "MCPForecastingToolkit",
    "BlockscoutMCPToolkit",
    "get_blockscout_toolkit",
    "DEXTradingToolkit",
    "MarketDataToolkit",
    "CryptoTools",
    "GuidryStatsToolkit",
    "ReviewPipelineToolkit",
]

