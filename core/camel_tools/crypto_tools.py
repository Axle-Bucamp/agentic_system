"""
Crypto Tools for CAMEL

Provides CAMEL-compatible tools for cryptocurrency-specific operations,
inspired by uniswap-trader-mcp, crypto-trending-mcp, and crypto-sentiment-mcp.
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


class CryptoTools:
    """Toolkit for cryptocurrency-specific operations."""
    
    def __init__(self, forecasting_client: Optional[ForecastingClient] = None):
        """
        Initialize the crypto tools.
        
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
    
    def get_crypto_trend_tool(self):
        """Get tool for analyzing crypto trends."""
        if not CAMEL_TOOLS_AVAILABLE:
            raise ImportError("CAMEL tools not installed")
        
        toolkit_instance = self
        
        async def get_crypto_trend(
            ticker: Annotated[str, Field(description="Cryptocurrency ticker symbol (e.g., BTC-USD)")],
            interval: Annotated[str, Field(description="Time interval: minutes, thirty, hours, or days")]
        ) -> Dict[str, Any]:
            """
            Get cryptocurrency trend analysis including price movement and momentum.
            
            Args:
                ticker: Cryptocurrency ticker symbol
                interval: Time interval for analysis
                
            Returns:
                Trend analysis with direction, strength, and momentum indicators
            """
            try:
                api_ticker = f"{ticker}-USD" if not ticker.endswith("-USD") else ticker
                
                # Get forecast and action recommendation
                forecast = await toolkit_instance.forecasting_client.get_stock_forecast(api_ticker, interval)
                action = await toolkit_instance.forecasting_client.get_action_recommendation(api_ticker, interval)
                
                # Determine trend direction
                action_map = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
                trend_direction = action_map.get(action.get("action", 1), "NEUTRAL")
                confidence = action.get("action_confidence", 0.5)
                
                return {
                    "success": True,
                    "ticker": ticker,
                    "interval": interval,
                    "trend_direction": trend_direction,
                    "trend_strength": confidence,
                    "forecast_price": forecast.get("forecast_price"),
                    "current_price": action.get("current_price"),
                    "momentum": "STRONG" if confidence > 0.7 else "MODERATE" if confidence > 0.5 else "WEAK"
                }
            except Exception as e:
                log.error(f"Error getting crypto trend: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "interval": interval
                }
        
        get_crypto_trend.__name__ = "get_crypto_trend"
        get_crypto_trend.__doc__ = "Get cryptocurrency trend analysis with direction, strength, and momentum"
        return get_crypto_trend
    
    def get_crypto_sentiment_tool(self):
        """Get tool for analyzing crypto market sentiment."""
        if not CAMEL_TOOLS_AVAILABLE:
            raise ImportError("CAMEL tools not installed")
        
        toolkit_instance = self
        
        async def get_crypto_sentiment(
            ticker: Annotated[str, Field(description="Cryptocurrency ticker symbol (e.g., BTC-USD)")]
        ) -> Dict[str, Any]:
            """
            Get market sentiment for a cryptocurrency based on DQN predictions and metrics.
            
            Args:
                ticker: Cryptocurrency ticker symbol
                
            Returns:
                Sentiment analysis with overall sentiment, confidence, and indicators
            """
            try:
                api_ticker = f"{ticker}-USD" if not ticker.endswith("-USD") else ticker
                
                # Get action recommendations for multiple intervals
                intervals = ["hours", "days"]
                sentiments = []
                
                for interval in intervals:
                    try:
                        action = await toolkit_instance.forecasting_client.get_action_recommendation(api_ticker, interval)
                        action_value = action.get("action", 1)
                        confidence = action.get("action_confidence", 0.5)
                        
                        # Map to sentiment
                        sentiment_map = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
                        sentiment = sentiment_map.get(action_value, "NEUTRAL")
                        
                        sentiments.append({
                            "interval": interval,
                            "sentiment": sentiment,
                            "confidence": confidence
                        })
                    except Exception:
                        continue
                
                # Aggregate sentiment
                weighted_score = 0.0
                if not sentiments:
                    overall_sentiment = "NEUTRAL"
                    avg_confidence = 0.5
                else:
                    sentiment_scores = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}
                    weighted_score = sum(
                        sentiment_scores.get(s["sentiment"], 0) * s["confidence"]
                        for s in sentiments
                    ) / len(sentiments)
                    
                    if weighted_score > 0.3:
                        overall_sentiment = "BULLISH"
                    elif weighted_score < -0.3:
                        overall_sentiment = "BEARISH"
                    else:
                        overall_sentiment = "NEUTRAL"
                    
                    avg_confidence = sum(s["confidence"] for s in sentiments) / len(sentiments)
                
                return {
                    "success": True,
                    "ticker": ticker,
                    "overall_sentiment": overall_sentiment,
                    "sentiment_confidence": avg_confidence,
                    "interval_sentiments": sentiments,
                    "sentiment_score": weighted_score
                }
            except Exception as e:
                log.error(f"Error getting crypto sentiment: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "overall_sentiment": "NEUTRAL"
                }
        
        get_crypto_sentiment.__name__ = "get_crypto_sentiment"
        get_crypto_sentiment.__doc__ = "Get market sentiment for a cryptocurrency based on DQN predictions"
        return get_crypto_sentiment
    
    def get_all_tools(self) -> List:
        """Get all tools in this toolkit."""
        base_tools = [
            self.get_crypto_trend_tool(),
            self.get_crypto_sentiment_tool(),
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

