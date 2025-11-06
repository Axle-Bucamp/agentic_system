"""
MCP Forecasting Toolkit for CAMEL

Provides CAMEL-compatible tools for interacting with the MCP forecasting API.
"""
from typing import Dict, Any, List, Optional, Annotated
from pydantic import Field
from core.config import settings
from core.logging import log
from core.forecasting_client import ForecastingClient, ForecastingAPIError

try:
    CAMEL_TOOLS_AVAILABLE = True
except ImportError:
    CAMEL_TOOLS_AVAILABLE = False
    log.warning("CAMEL tools not available. Install with: pip install camel-ai")


class MCPForecastingToolkit:
    """Toolkit for MCP forecasting API endpoints."""
    
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
    
    def get_stock_forecast_tool(self):
        """Get tool for retrieving stock forecasts."""
        if not CAMEL_TOOLS_AVAILABLE:
            raise ImportError("CAMEL tools not installed")
        
        toolkit_instance = self
        
        async def get_stock_forecast(
            ticker: Annotated[str, Field(description="Stock ticker symbol (e.g., BTC-USD)")],
            interval: Annotated[str, Field(description="Time interval: minutes, thirty, hours, or days")]
        ) -> Dict[str, Any]:
            """
            Get price forecast for a specific ticker and interval.
            
            Args:
                ticker: Stock ticker symbol
                interval: Time interval
                
            Returns:
                Forecast data including price predictions and confidence
            """
            try:
                # Convert ticker format if needed
                api_ticker = f"{ticker}-USD" if not ticker.endswith("-USD") else ticker
                
                # Access forecasting_client from closure
                result = await toolkit_instance.forecasting_client.get_stock_forecast(api_ticker, interval)
                
                return {
                    "success": True,
                    "ticker": ticker,
                    "interval": interval,
                    "forecast": result
                }
            except ForecastingAPIError as e:
                log.error(f"Forecasting API error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "interval": interval
                }
            except Exception as e:
                log.error(f"Error getting stock forecast: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "interval": interval
                }
        
        get_stock_forecast.__name__ = "get_stock_forecast"
        get_stock_forecast.__doc__ = "Get price forecast for a specific ticker and interval from the MCP forecasting API"
        return get_stock_forecast
    
    def get_action_recommendation_tool(self):
        """Get tool for retrieving action recommendations."""
        if not CAMEL_TOOLS_AVAILABLE:
            raise ImportError("CAMEL tools not installed")
        
        toolkit_instance = self
        
        async def get_action_recommendation(
            ticker: Annotated[str, Field(description="Stock ticker symbol (e.g., BTC-USD)")],
            interval: Annotated[str, Field(description="Time interval: minutes, thirty, hours, or days")]
        ) -> Dict[str, Any]:
            """
            Get DQN action recommendation (BUY/SELL/HOLD) with confidence.
            
            Args:
                ticker: Stock ticker symbol
                interval: Time interval
                
            Returns:
                Action recommendation with confidence and Q-values
            """
            try:
                api_ticker = f"{ticker}-USD" if not ticker.endswith("-USD") else ticker
                
                result = await toolkit_instance.forecasting_client.get_action_recommendation(api_ticker, interval)
                
                # Map action values to names
                action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
                action_value = result.get("action", 1)
                action_name = action_map.get(action_value, "HOLD")
                
                return {
                    "success": True,
                    "ticker": ticker,
                    "interval": interval,
                    "action": action_value,
                    "action_name": action_name,
                    "confidence": result.get("action_confidence", 0.5),
                    "forecast_price": result.get("forecast_price"),
                    "q_values": result.get("q_values", []),
                    "current_price": result.get("current_price")
                }
            except ForecastingAPIError as e:
                log.error(f"Forecasting API error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "interval": interval
                }
            except Exception as e:
                log.error(f"Error getting action recommendation: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "interval": interval
                }
        
        get_action_recommendation.__name__ = "get_action_recommendation"
        get_action_recommendation.__doc__ = "Get DQN action recommendation (BUY/SELL/HOLD) with confidence for a ticker"
        return get_action_recommendation
    
    def list_available_tickers_tool(self):
        """Get tool for listing available tickers."""
        if not CAMEL_TOOLS_AVAILABLE:
            raise ImportError("CAMEL tools not installed")
        
        toolkit_instance = self
        
        async def list_available_tickers() -> Dict[str, Any]:
            """
            Get list of all available tickers with DQN data.
            
            Returns:
                List of available tickers with their metadata
            """
            try:
                tickers = await toolkit_instance.forecasting_client.get_available_tickers()
                
                # Filter tickers with DQN data if available
                tickers_with_dqn = [
                    ticker for ticker in tickers
                    if ticker.get("has_dqn", False) or isinstance(ticker, str)
                ]
                
                return {
                    "success": True,
                    "tickers": tickers_with_dqn,
                    "count": len(tickers_with_dqn)
                }
            except ForecastingAPIError as e:
                log.error(f"Forecasting API error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "tickers": [],
                    "count": 0
                }
            except Exception as e:
                log.error(f"Error listing tickers: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "tickers": [],
                    "count": 0
                }
        
        list_available_tickers.__name__ = "list_available_tickers"
        list_available_tickers.__doc__ = "Get list of all available tickers with DQN forecasting data"
        return list_available_tickers
    
    def get_metrics_tool(self):
        """Get tool for retrieving model metrics."""
        if not CAMEL_TOOLS_AVAILABLE:
            raise ImportError("CAMEL tools not installed")
        
        toolkit_instance = self
        
        async def get_metrics(
            ticker: Annotated[str, Field(description="Stock ticker symbol (e.g., BTC-USD)")],
            interval: Annotated[str, Field(description="Time interval: minutes, thirty, hours, or days")]
        ) -> Dict[str, Any]:
            """
            Get model performance metrics for a ticker.
            
            Args:
                ticker: Stock ticker symbol
                interval: Time interval
                
            Returns:
                Model performance metrics including accuracy, Sharpe ratio, etc.
            """
            try:
                api_ticker = f"{ticker}-USD" if not ticker.endswith("-USD") else ticker
                
                result = await toolkit_instance.forecasting_client.get_model_metrics(api_ticker, interval)
                
                return {
                    "success": True,
                    "ticker": ticker,
                    "interval": interval,
                    "metrics": result
                }
            except ForecastingAPIError as e:
                log.error(f"Forecasting API error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "interval": interval
                }
            except Exception as e:
                log.error(f"Error getting metrics: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "interval": interval
                }
        
        get_metrics.__name__ = "get_metrics"
        get_metrics.__doc__ = "Get model performance metrics (accuracy, Sharpe ratio, etc.) for a ticker"
        return get_metrics
    
    def get_all_tools(self) -> List:
        """Get all tools in this toolkit."""
        return [
            self.get_stock_forecast_tool(),
            self.get_action_recommendation_tool(),
            self.list_available_tickers_tool(),
            self.get_metrics_tool(),
        ]

