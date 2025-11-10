"""
Market Data Toolkit for CAMEL

Provides CAMEL-compatible tools for retrieving market data and performing analysis.
"""
from typing import Dict, Any, List, Optional, Annotated
from pydantic import Field
from core.config import settings
from core.logging import log
from core.forecasting_client import ForecastingClient

try:
    from camel.toolkits import FunctionTool  # type: ignore
    CAMEL_TOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FunctionTool = None  # type: ignore
    CAMEL_TOOLS_AVAILABLE = False
    log.warning("CAMEL function tools not available. Install with: pip install 'camel-ai[tools]'")


class MarketDataToolkit:
    """Toolkit for market data retrieval and analysis."""
    
    def __init__(self, forecasting_client: Optional[ForecastingClient] = None):
        """
        Initialize the toolkit.
        
        Args:
            forecasting_client: Optional ForecastingClient instance
        """
        self.forecasting_client = forecasting_client or ForecastingClient({
            "base_url": settings.mcp_api_url,
            "api_key": settings.mcp_api_key,
            "mock_mode": settings.use_mock_services,
        })
    
    async def initialize(self):
        """Initialize the forecasting client."""
        await self.forecasting_client.connect()
    
    def get_ticker_info_tool(self):
        """Get tool for retrieving ticker information."""
        if not CAMEL_TOOLS_AVAILABLE:
            raise ImportError("CAMEL tools not installed")
        
        toolkit_instance = self
        
        async def get_ticker_info(
            ticker: Annotated[str, Field(description="Stock ticker symbol (e.g., BTC-USD)")]
        ) -> Dict[str, Any]:
            """
            Get detailed information about a specific ticker.
            
            Args:
                ticker: Stock ticker symbol
                
            Returns:
                Ticker information including available intervals and metadata
            """
            try:
                api_ticker = f"{ticker}-USD" if not ticker.endswith("-USD") else ticker
                
                result = await toolkit_instance.forecasting_client.get_ticker_info(api_ticker)
                
                return {
                    "success": True,
                    "ticker": ticker,
                    "info": result
                }
            except Exception as e:
                log.error(f"Error getting ticker info: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "info": {}
                }
        
        get_ticker_info.__name__ = "get_ticker_info"
        get_ticker_info.__doc__ = "Get detailed information about a ticker including available intervals"
        return get_ticker_info
    
    def get_available_intervals_tool(self):
        """Get tool for listing available intervals."""
        if not CAMEL_TOOLS_AVAILABLE:
            raise ImportError("CAMEL tools not installed")
        
        toolkit_instance = self
        
        async def get_available_intervals() -> Dict[str, Any]:
            """
            Get list of all available time intervals.
            
            Returns:
                List of available intervals
            """
            try:
                intervals = await toolkit_instance.forecasting_client.get_available_intervals()
                
                return {
                    "success": True,
                    "intervals": intervals,
                    "count": len(intervals)
                }
            except Exception as e:
                log.error(f"Error getting available intervals: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "intervals": [],
                    "count": 0
                }
        
        get_available_intervals.__name__ = "get_available_intervals"
        get_available_intervals.__doc__ = "Get list of all available time intervals for forecasting"
        return get_available_intervals
    
    def get_all_tools(self) -> List:
        """Get all tools in this toolkit."""
        base_tools = [
            self.get_ticker_info_tool(),
            self.get_available_intervals_tool(),
        ]
        return [self._wrap_tool(func) for func in base_tools]

    @staticmethod
    def _wrap_tool(func):
        if not CAMEL_TOOLS_AVAILABLE or FunctionTool is None:
            raise ImportError("CAMEL function tools not installed")

        tool = FunctionTool(func)
        try:
            schema = dict(tool.get_openai_tool_schema())
        except Exception:
            schema = {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": func.__doc__ or func.__name__,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }

        function_schema = schema.setdefault("function", {})
        function_schema["name"] = func.__name__
        function_schema.setdefault("description", func.__doc__ or func.__name__)
        tool.openai_tool_schema = schema
        return tool

