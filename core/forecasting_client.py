"""
Forecasting API client for guidry-cloud.com integration.
"""
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import httpx
from core.logging import log
from core.mocks.mock_forecasting_service import get_mock_forecasting_service


class ForecastingAPIError(Exception):
    """Base exception for forecasting API operations."""
    pass


class ForecastingClient:
    """Client for interacting with the forecasting API at guidry-cloud.com."""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get("base_url", "https://forecasting.guidry-cloud.com")
        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 30.0)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        
        # Cache for frequently accessed data
        self.cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        self.default_cache_ttl = timedelta(minutes=5)
        
        # Mock mode
        self.is_mock = config.get("mock_mode", config.get("use_mock_services", True))
        self.mock_data: Dict[str, Any] = {}
        self.mock_service = None
    
    async def connect(self) -> None:
        """Initialize the HTTP client."""
        try:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "AgenticTradingSystem/1.0.0"
            }
            
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                verify=False  # Disable SSL verification for development/testing
            )
            
            if self.is_mock:
                self.mock_service = await get_mock_forecasting_service()
                await self._setup_mock_data()
            
            log.info(f"Forecasting API client connected to {self.base_url}")     
              
        except Exception as e:
            log.error(f"Failed to connect to forecasting API: {e}")
            raise ForecastingAPIError(f"Connection failed: {e}")
    
    async def initialize(self) -> None:
        """Initialize the forecasting client (alias for connect)."""
        try:
            await self.connect()        
        except Exception as e:
            log.error(f"Failed to connect to forecasting API: {e}")
            raise ForecastingAPIError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
        log.info("Forecasting API client disconnected")
    
    async def _setup_mock_data(self) -> None:
        """Setup mock data for testing."""
        self.mock_data = {
            "tickers": [
                {"symbol": "BTC-USD", "name": "Bitcoin", "type": "crypto", "intervals": ["minutes", "hours", "days", "thirty"], "has_dqn": True},
                {"symbol": "ETH-USD", "name": "Ethereum", "type": "crypto", "intervals": ["minutes", "hours", "days", "thirty"], "has_dqn": True},
                {"symbol": "SOL-USD", "name": "Solana", "type": "crypto", "intervals": ["minutes", "hours", "days", "thirty"], "has_dqn": True},
            ],
            "actions": {
                "BTC-USD": {"hours": {"action": 2, "action_confidence": 0.85, "forecast_price": 47000.0}, "days": {"action": 2, "action_confidence": 0.78, "forecast_price": 50000.0}},
                "ETH-USD": {"hours": {"action": 1, "action_confidence": 0.72, "forecast_price": 2100.0}, "days": {"action": 2, "action_confidence": 0.81, "forecast_price": 2300.0}},
                "SOL-USD": {"hours": {"action": 0, "action_confidence": 0.68, "forecast_price": 95.0}, "days": {"action": 1, "action_confidence": 0.75, "forecast_price": 100.0}},
            },
            "metrics": {
                "BTC-USD": {"hours": {"accuracy": 0.82, "sharpe_ratio": 1.45, "max_drawdown": 0.12}, "days": {"accuracy": 0.78, "sharpe_ratio": 1.32, "max_drawdown": 0.15}},
                "ETH-USD": {"hours": {"accuracy": 0.79, "sharpe_ratio": 1.28, "max_drawdown": 0.14}, "days": {"accuracy": 0.81, "sharpe_ratio": 1.41, "max_drawdown": 0.13}},
                "SOL-USD": {"hours": {"accuracy": 0.75, "sharpe_ratio": 1.15, "max_drawdown": 0.18}, "days": {"accuracy": 0.77, "sharpe_ratio": 1.22, "max_drawdown": 0.16}},
            }
        }
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        if not self.client:
            raise ForecastingAPIError("Not connected to forecasting API")
        
        for attempt in range(self.retry_attempts):
            try:
                if method == "GET":
                    response = await self.client.get(endpoint, params=params)
                elif method == "POST":
                    response = await self.client.post(endpoint, params=params, json=data)
                else:
                    raise ForecastingAPIError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code in [429, 502, 503, 504] and attempt < self.retry_attempts - 1:
                    # Retry on rate limit or server errors
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise ForecastingAPIError(f"HTTP error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise ForecastingAPIError(f"Request failed: {e}")
        
        raise ForecastingAPIError("Max retry attempts exceeded")
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if not expired."""
        if key in self.cache and key in self.cache_ttl:
            if datetime.utcnow() < self.cache_ttl[key]:
                return self.cache[key]
            else:
                # Remove expired cache
                del self.cache[key]
                del self.cache_ttl[key]
        return None
    
    def _set_cache(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Set cached data with TTL."""
        self.cache[key] = value
        self.cache_ttl[key] = datetime.utcnow() + (ttl or self.default_cache_ttl)
    
    async def get_available_tickers(self) -> List[Dict[str, Any]]:
        """Get list of available tickers."""
        cache_key = "available_tickers"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.is_mock:
            if self.mock_service:
                # Use comprehensive mock service
                tickers = await self.mock_service.get_available_tickers()
                result = [{"symbol": ticker, "name": ticker.replace("-", " "), "active": True} for ticker in tickers]
            else:
                result = self.mock_data["tickers"]
        else:
            try:
                response = await self._make_request("GET", "/api/tickers/available")
                result = response.get("tickers", [])
            except Exception as e:
                log.error(f"Failed to get available tickers: {e}")
                return []
        
        self._set_cache(cache_key, result)
        return result
    
    async def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Get detailed information about a specific ticker."""
        cache_key = f"ticker_info_{ticker}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.is_mock:
            # Find ticker in mock data
            ticker_data = next((t for t in self.mock_data["tickers"] if t["symbol"] == ticker), None)
            if not ticker_data:
                raise ForecastingAPIError(f"Ticker {ticker} not found")
            result = ticker_data
        else:
            try:
                response = await self._make_request("GET", f"/api/tickers/{ticker}/info")
                result = response
            except Exception as e:
                log.error(f"Failed to get ticker info for {ticker}: {e}")
                raise ForecastingAPIError(f"Failed to get ticker info: {e}")
        
        self._set_cache(cache_key, result, timedelta(hours=1))  # Cache for 1 hour
        return result
    
    async def get_action_recommendation(self, ticker: str, interval: str) -> Dict[str, Any]:
        """Get DQN action recommendation for a ticker and interval."""
        cache_key = f"action_{ticker}_{interval}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.is_mock:
            if self.mock_service:
                # Use comprehensive mock service
                mock_result = await self.mock_service.get_action_recommendation(ticker, interval)
                result = {
                    "action": 0 if mock_result["recommendation"] == "SELL" else 1 if mock_result["recommendation"] == "HOLD" else 2,
                    "action_confidence": mock_result["confidence"],
                    "forecast_price": mock_result.get("price_target", 100.0),
                    "q_values": [0.3, 0.5, 0.2],  # [SELL, HOLD, BUY]
                    "current_price": 100.0,
                    "recommendation": mock_result["recommendation"],
                    "reasoning": mock_result["reasoning"]
                }
            elif ticker in self.mock_data["actions"] and interval in self.mock_data["actions"][ticker]:
                result = self.mock_data["actions"][ticker][interval]
            else:
                # Default mock response
                result = {
                    "action": 1,  # HOLD
                    "action_confidence": 0.5,
                    "forecast_price": 100.0,
                    "q_values": [0.3, 0.5, 0.2],  # [SELL, HOLD, BUY]
                    "current_price": 100.0
                }
        else:
            try:
                response = await self._make_request("GET", f"/api/json/action/{ticker}/{interval}")
                result = response
            except Exception as e:
                log.error(f"Failed to get action recommendation for {ticker}/{interval}: {e}")
                raise ForecastingAPIError(f"Failed to get action recommendation: {e}")
        
        self._set_cache(cache_key, result, timedelta(minutes=2))  # Cache for 2 minutes
        return result
    
    async def get_stock_forecast(self, ticker: str, interval: str) -> Dict[str, Any]:
        """Get detailed stock forecast data."""
        cache_key = f"forecast_{ticker}_{interval}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.is_mock:
            if self.mock_service:
                # Use comprehensive mock service
                mock_result = await self.mock_service.get_stock_forecast(ticker, interval)
                result = {
                    "ticker": ticker,
                    "interval": interval,
                    "forecast_price": mock_result["forecast"][0]["price"] if mock_result["forecast"] else 100.0,
                    "confidence": mock_result["forecast"][0]["confidence"] if mock_result["forecast"] else 0.75,
                    "trend": mock_result["forecast"][0]["trend"] if mock_result["forecast"] else "bullish",
                    "support_levels": [95.0, 90.0],
                    "resistance_levels": [105.0, 110.0],
                    "timestamp": datetime.utcnow().isoformat(),
                    "forecast_data": mock_result["forecast"],
                    "model_version": mock_result.get("model_version", "v1.0.0")
                }
            else:
                # Generate mock forecast data
                result = {
                    "ticker": ticker,
                    "interval": interval,
                    "forecast_price": 100.0,
                    "confidence": 0.75,
                    "trend": "bullish",
                    "support_levels": [95.0, 90.0],
                    "resistance_levels": [105.0, 110.0],
                    "timestamp": datetime.utcnow().isoformat()
                }
        else:
            try:
                response = await self._make_request("GET", f"/api/json/stock/{interval}/{ticker}")
                result = response
            except Exception as e:
                log.error(f"Failed to get stock forecast for {ticker}/{interval}: {e}")
                raise ForecastingAPIError(f"Failed to get stock forecast: {e}")
        
        self._set_cache(cache_key, result, timedelta(minutes=5))  # Cache for 5 minutes
        return result
    
    async def get_model_metrics(self, ticker: str, interval: str) -> Dict[str, Any]:
        """Get model performance metrics."""
        cache_key = f"metrics_{ticker}_{interval}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.is_mock:
            if ticker in self.mock_data["metrics"] and interval in self.mock_data["metrics"][ticker]:
                result = self.mock_data["metrics"][ticker][interval]
            else:
                # Default mock metrics
                result = {
                    "accuracy": 0.75,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": 0.15,
                    "win_rate": 0.65,
                    "total_trades": 100,
                    "avg_return": 0.02
                }
        else:
            try:
                response = await self._make_request("GET", f"/api/json/metrics/{ticker}/{interval}")
                result = response
            except Exception as e:
                log.error(f"Failed to get model metrics for {ticker}/{interval}: {e}")
                raise ForecastingAPIError(f"Failed to get model metrics: {e}")
        
        self._set_cache(cache_key, result, timedelta(hours=1))  # Cache for 1 hour
        return result
    
    async def get_available_intervals(self) -> List[str]:
        """Get list of available intervals."""
        cache_key = "available_intervals"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.is_mock:
            result = ["minutes", "hours", "days", "thirty"]
        else:
            try:
                response = await self._make_request("GET", "/api/tickers/intervals")
                result = response.get("intervals", [])
            except Exception as e:
                log.error(f"Failed to get available intervals: {e}")
                return ["minutes", "hours", "days", "thirty"]  # Fallback
        
        self._set_cache(cache_key, result, timedelta(hours=24))  # Cache for 24 hours
        return result
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment."""
        cache_key = "market_sentiment"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.is_mock:
            result = {
                "overall_sentiment": "bullish",
                "sentiment_score": 0.65,
                "fear_greed_index": 72,
                "market_trend": "upward",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            try:
                response = await self._make_request("GET", "/api/news/market-sentiment")
                result = response
            except Exception as e:
                log.error(f"Failed to get market sentiment: {e}")
                # Return neutral sentiment as fallback
                result = {
                    "overall_sentiment": "neutral",
                    "sentiment_score": 0.5,
                    "fear_greed_index": 50,
                    "market_trend": "sideways",
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        self._set_cache(cache_key, result, timedelta(minutes=15))  # Cache for 15 minutes
        return result
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get API health status."""
        try:
            if self.is_mock:
                return {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "version": "1.0.0",
                    "uptime": "99.9%"
                }
            
            response = await self._make_request("GET", "/health")
            return response
            
        except Exception as e:
            log.error(f"Failed to get health status: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.cache_ttl.clear()
        log.info("Forecasting API cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cached_keys": list(self.cache.keys()),
            "cache_ttl_entries": len(self.cache_ttl)
        }


# Global forecasting client instance
from core.config import settings

forecasting_client = ForecastingClient({
    "base_url": settings.mcp_api_url,
    "api_key": settings.mcp_api_key,
    "mock_mode": settings.use_mock_services,
    "timeout": 30.0,
    "retry_attempts": 3,
    "retry_delay": 1.0
})
