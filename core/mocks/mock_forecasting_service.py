"""
Mock Forecasting Service

Provides a comprehensive mock implementation of the forecasting API
that simulates real behavior for testing and development.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from core.logging import log


class RecommendationType(Enum):
    """Trading recommendation types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class ForecastData:
    """Forecast data structure."""
    timestamp: datetime
    price: float
    confidence: float
    volume: float
    trend: str


@dataclass
class RecommendationData:
    """Recommendation data structure."""
    recommendation: RecommendationType
    confidence: float
    reasoning: str
    price_target: Optional[float]
    stop_loss: Optional[float]
    risk_score: float


class MockForecastingService:
    """Mock forecasting service that simulates real API behavior."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.base_url = self.config.get("base_url", "https://forecasting.guidry-cloud.com")
        self.api_key = self.config.get("api_key", "mock_api_key")
        self.rate_limit = self.config.get("rate_limit", 100)  # requests per minute
        self.response_delay = self.config.get("response_delay", 0.1)  # seconds
        
        # Mock data storage
        self.tickers = [
            "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD",
            "MATIC-USD", "AVAX-USD", "LINK-USD", "UNI-USD", "ATOM-USD"
        ]
        
        # Price ranges for realistic simulation
        self.price_ranges = {
            "BTC-USD": (40000, 70000),
            "ETH-USD": (2000, 4000),
            "SOL-USD": (50, 200),
            "ADA-USD": (0.3, 1.0),
            "DOT-USD": (5, 15),
            "MATIC-USD": (0.5, 2.0),
            "AVAX-USD": (10, 50),
            "LINK-USD": (5, 25),
            "UNI-USD": (3, 15),
            "ATOM-USD": (5, 20)
        }
        
        # Historical data cache
        self.historical_data = {}
        self.forecast_cache = {}
        self.recommendation_cache = {}
        
        # Initialize with some historical data
        self._initialize_historical_data()
        
        log.info("Mock Forecasting Service initialized")
    
    def _initialize_historical_data(self):
        """Initialize historical data for realistic simulation."""
        for ticker in self.tickers:
            base_price = random.uniform(*self.price_ranges[ticker])
            historical = []
            
            # Generate 100 historical data points
            for i in range(100):
                timestamp = datetime.utcnow() - timedelta(hours=100-i)
                # Add some realistic price movement
                price_change = random.uniform(-0.05, 0.05)  # ±5% change
                base_price *= (1 + price_change)
                
                historical.append({
                    "timestamp": timestamp.isoformat(),
                    "price": round(base_price, 2),
                    "volume": random.uniform(1000000, 10000000),
                    "high": round(base_price * random.uniform(1.0, 1.02), 2),
                    "low": round(base_price * random.uniform(0.98, 1.0), 2),
                    "change_24h": round(price_change * 100, 2)
                })
            
            self.historical_data[ticker] = historical
    
    async def _simulate_api_delay(self):
        """Simulate API response delay."""
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
    
    async def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits."""
        # Simple rate limiting implementation
        current_time = time.time()
        minute_key = int(current_time // 60)
        
        if not hasattr(self, '_rate_limit_tracker'):
            self._rate_limit_tracker = {}
        
        # Clean old entries
        self._rate_limit_tracker = {
            k: v for k, v in self._rate_limit_tracker.items()
            if k >= minute_key - 1
        }
        
        # Check current minute
        current_count = self._rate_limit_tracker.get(minute_key, 0)
        if current_count >= self.rate_limit:
            return False
        
        # Increment counter
        self._rate_limit_tracker[minute_key] = current_count + 1
        return True
    
    async def get_available_tickers(self) -> List[str]:
        """Get list of available tickers."""
        await self._simulate_api_delay()
        
        if not await self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        return self.tickers.copy()

    async def get_ohlc(
        self,
        ticker: str,
        interval: str = "hours",
        limit: int = 120,
    ) -> List[Dict[str, Any]]:
        """Return mock OHLC candles."""
        await self._simulate_api_delay()

        if not await self._check_rate_limit():
            raise Exception("Rate limit exceeded")

        if ticker not in self.historical_data:
            raise Exception(f"Ticker {ticker} not found")

        historical = self.historical_data[ticker][-limit:]
        candles: List[Dict[str, Any]] = []
        for entry in historical:
            candles.append(
                {
                    "timestamp": entry["timestamp"],
                    "open": entry["price"] * 0.99,
                    "high": entry["high"],
                    "low": entry["low"],
                    "close": entry["price"],
                    "volume": entry["volume"],
                }
            )
        return candles
    
    async def get_stock_forecast(
        self, 
        ticker: str, 
        interval: str = "hours",
        periods: int = 24
    ) -> Dict[str, Any]:
        """Get stock forecast for a ticker."""
        await self._simulate_api_delay()
        
        if not await self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        if ticker not in self.tickers:
            raise Exception(f"Ticker {ticker} not found")
        
        # Check cache first
        cache_key = f"{ticker}_{interval}_{periods}"
        if cache_key in self.forecast_cache:
            cached_data = self.forecast_cache[cache_key]
            if time.time() - cached_data["timestamp"] < 300:  # 5 minute cache
                return cached_data["data"]
        
        # Generate forecast
        base_price = random.uniform(*self.price_ranges[ticker])
        forecast_data = []
        
        for i in range(periods):
            # Add realistic price movement
            price_change = random.uniform(-0.02, 0.02)  # ±2% change per period
            base_price *= (1 + price_change)
            
            # Determine trend
            if price_change > 0.01:
                trend = "bullish"
            elif price_change < -0.01:
                trend = "bearish"
            else:
                trend = "sideways"
            
            forecast_data.append({
                "timestamp": (datetime.utcnow() + timedelta(hours=i)).isoformat(),
                "price": round(base_price, 2),
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "volume": random.uniform(1000000, 10000000),
                "trend": trend,
                "high": round(base_price * random.uniform(1.0, 1.03), 2),
                "low": round(base_price * random.uniform(0.97, 1.0), 2)
            })
        
        result = {
            "ticker": ticker,
            "interval": interval,
            "periods": periods,
            "forecast": forecast_data,
            "model_version": "v1.0.0",
            "generated_at": datetime.utcnow().isoformat(),
            "accuracy": round(random.uniform(0.75, 0.92), 2)
        }
        
        # Cache result
        self.forecast_cache[cache_key] = {
            "data": result,
            "timestamp": time.time()
        }
        
        return result
    
    async def get_action_recommendation(
        self, 
        ticker: str, 
        interval: str = "hours"
    ) -> Dict[str, Any]:
        """Get trading action recommendation for a ticker."""
        await self._simulate_api_delay()
        
        if not await self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        if ticker not in self.tickers:
            raise Exception(f"Ticker {ticker} not found")
        
        # Check cache first
        cache_key = f"{ticker}_{interval}_recommendation"
        if cache_key in self.recommendation_cache:
            cached_data = self.recommendation_cache[cache_key]
            if time.time() - cached_data["timestamp"] < 180:  # 3 minute cache
                return cached_data["data"]
        
        # Generate recommendation
        confidence = random.uniform(0.65, 0.95)
        base_price = random.uniform(*self.price_ranges[ticker])
        
        # Determine recommendation based on confidence and some logic
        if confidence > 0.85:
            if random.random() > 0.5:
                recommendation = RecommendationType.BUY
                price_target = base_price * random.uniform(1.05, 1.15)
                stop_loss = base_price * random.uniform(0.90, 0.95)
                reasoning = "Strong bullish signals from technical analysis and market sentiment"
            else:
                recommendation = RecommendationType.SELL
                price_target = base_price * random.uniform(0.85, 0.95)
                stop_loss = base_price * random.uniform(1.05, 1.10)
                reasoning = "Bearish momentum detected with high confidence"
        elif confidence > 0.70:
            recommendation = RecommendationType.HOLD
            price_target = None
            stop_loss = None
            reasoning = "Mixed signals suggest waiting for clearer direction"
        else:
            recommendation = RecommendationType.HOLD
            price_target = None
            stop_loss = None
            reasoning = "Low confidence in market direction, maintaining current position"
        
        # Calculate risk score
        risk_score = random.uniform(0.2, 0.8)
        
        result = {
            "ticker": ticker,
            "recommendation": recommendation.value,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
            "price_target": round(price_target, 2) if price_target else None,
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "risk_score": round(risk_score, 2),
            "interval": interval,
            "model_version": "v1.0.0",
            "generated_at": datetime.utcnow().isoformat(),
            "market_conditions": {
                "volatility": random.uniform(0.1, 0.4),
                "trend_strength": random.uniform(0.3, 0.9),
                "volume_trend": random.choice(["increasing", "decreasing", "stable"])
            }
        }
        
        # Cache result
        self.recommendation_cache[cache_key] = {
            "data": result,
            "timestamp": time.time()
        }
        
        return result
    
    async def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        await self._simulate_api_delay()
        
        return {
            "model_version": "v1.0.0",
            "accuracy": round(random.uniform(0.75, 0.92), 2),
            "precision": round(random.uniform(0.70, 0.90), 2),
            "recall": round(random.uniform(0.65, 0.88), 2),
            "f1_score": round(random.uniform(0.68, 0.89), 2),
            "total_predictions": random.randint(10000, 100000),
            "last_updated": datetime.utcnow().isoformat(),
            "performance_by_ticker": {
                ticker: {
                    "accuracy": round(random.uniform(0.70, 0.95), 2),
                    "predictions": random.randint(100, 1000)
                }
                for ticker in self.tickers[:5]  # Top 5 tickers
            }
        }
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment."""
        await self._simulate_api_delay()
        
        return {
            "overall_sentiment": round(random.uniform(-1.0, 1.0), 2),
            "fear_greed_index": random.randint(0, 100),
            "market_phase": random.choice(["bull", "bear", "sideways", "volatile"]),
            "confidence": round(random.uniform(0.6, 0.9), 2),
            "timestamp": datetime.utcnow().isoformat(),
            "sector_sentiment": {
                "crypto": round(random.uniform(-1.0, 1.0), 2),
                "defi": round(random.uniform(-1.0, 1.0), 2),
                "nft": round(random.uniform(-1.0, 1.0), 2),
                "gaming": round(random.uniform(-1.0, 1.0), 2)
            },
            "news_sentiment": {
                "positive": random.randint(20, 60),
                "negative": random.randint(10, 40),
                "neutral": random.randint(20, 50)
            }
        }
    
    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        await self._simulate_api_delay()
        return True
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get detailed service status."""
        await self._simulate_api_delay()
        
        return {
            "status": "healthy",
            "uptime": random.randint(1000, 10000),  # seconds
            "version": "1.0.0",
            "rate_limit_remaining": self.rate_limit - self._rate_limit_tracker.get(int(time.time() // 60), 0),
            "cache_size": {
                "forecasts": len(self.forecast_cache),
                "recommendations": len(self.recommendation_cache)
            },
            "supported_tickers": len(self.tickers),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def reset_cache(self):
        """Reset all caches."""
        self.forecast_cache.clear()
        self.recommendation_cache.clear()
        log.info("Mock forecasting service cache reset")
    
    async def simulate_market_crash(self):
        """Simulate a market crash for testing."""
        # Reduce all prices by 20-30%
        for ticker in self.price_ranges:
            current_range = self.price_ranges[ticker]
            new_low = current_range[0] * 0.7
            new_high = current_range[1] * 0.8
            self.price_ranges[ticker] = (new_low, new_high)
        
        # Clear caches to force new data generation
        await self.reset_cache()
        
        log.warning("Simulated market crash - prices reduced by 20-30%")
    
    async def simulate_market_rally(self):
        """Simulate a market rally for testing."""
        # Increase all prices by 20-30%
        for ticker in self.price_ranges:
            current_range = self.price_ranges[ticker]
            new_low = current_range[0] * 1.2
            new_high = current_range[1] * 1.3
            self.price_ranges[ticker] = (new_low, new_high)
        
        # Clear caches to force new data generation
        await self.reset_cache()
        
        log.info("Simulated market rally - prices increased by 20-30%")


# Global mock service instance
_mock_forecasting_service: Optional[MockForecastingService] = None


async def get_mock_forecasting_service() -> MockForecastingService:
    """Get global mock forecasting service instance."""
    global _mock_forecasting_service
    if _mock_forecasting_service is None:
        _mock_forecasting_service = MockForecastingService()
    return _mock_forecasting_service
