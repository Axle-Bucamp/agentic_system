"""
Unit tests for forecasting client.
"""
import pytest
from datetime import datetime
from core.forecasting_client import ForecastingClient, ForecastingAPIError
from tests.mocks.mock_forecasting_api import MockForecastingClient


class TestForecastingClient:
    """Test forecasting client functionality."""
    
    @pytest.mark.asyncio
    async def test_forecasting_client_connection(self):
        """Test forecasting client connection."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        
        assert not client.is_connected
        await client.connect()
        assert client.is_connected
        
        await client.disconnect()
        assert not client.is_connected
    
    @pytest.mark.asyncio
    async def test_get_available_tickers(self):
        """Test getting available tickers."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        tickers = await client.get_available_tickers()
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        
        # Check ticker structure
        ticker = tickers[0]
        assert "symbol" in ticker
        assert "name" in ticker
        assert "type" in ticker
        assert "intervals" in ticker
        assert "has_dqn" in ticker
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_ticker_info(self):
        """Test getting ticker info."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        ticker_info = await client.get_ticker_info("BTC-USD")
        assert ticker_info["symbol"] == "BTC-USD"
        assert ticker_info["name"] == "Bitcoin"
        assert ticker_info["type"] == "crypto"
        assert "intervals" in ticker_info
        assert ticker_info["has_dqn"] is True
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_ticker_info_not_found(self):
        """Test getting ticker info for non-existent ticker."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        with pytest.raises(ForecastingAPIError):
            await client.get_ticker_info("NONEXISTENT")
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_action_recommendation(self):
        """Test getting action recommendation."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        action = await client.get_action_recommendation("BTC-USD", "hours")
        assert "action" in action
        assert "action_confidence" in action
        assert "forecast_price" in action
        assert "q_values" in action
        assert "current_price" in action
        assert "timestamp" in action
        
        # Check action values
        assert action["action"] in [0, 1, 2]  # SELL, HOLD, BUY
        assert 0 <= action["action_confidence"] <= 1
        assert action["forecast_price"] > 0
        assert len(action["q_values"]) == 3  # [SELL, HOLD, BUY]
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_action_recommendation_different_intervals(self):
        """Test getting action recommendation for different intervals."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        intervals = ["minutes", "hours", "days", "thirty"]
        for interval in intervals:
            action = await client.get_action_recommendation("BTC-USD", interval)
            assert "action" in action
            assert "action_confidence" in action
            assert "forecast_price" in action
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_action_recommendation_ticker_not_found(self):
        """Test getting action recommendation for non-existent ticker."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        with pytest.raises(ForecastingAPIError):
            await client.get_action_recommendation("NONEXISTENT", "hours")
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_action_recommendation_interval_not_found(self):
        """Test getting action recommendation for non-existent interval."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        with pytest.raises(ForecastingAPIError):
            await client.get_action_recommendation("BTC-USD", "invalid_interval")
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_stock_forecast(self):
        """Test getting stock forecast."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        forecast = await client.get_stock_forecast("BTC-USD", "hours")
        assert "ticker" in forecast
        assert "interval" in forecast
        assert "forecast_price" in forecast
        assert "confidence" in forecast
        assert "trend" in forecast
        assert "support_levels" in forecast
        assert "resistance_levels" in forecast
        assert "timestamp" in forecast
        
        # Check forecast values
        assert forecast["ticker"] == "BTC-USD"
        assert forecast["interval"] == "hours"
        assert forecast["forecast_price"] > 0
        assert 0 <= forecast["confidence"] <= 1
        assert forecast["trend"] in ["bullish", "bearish", "neutral"]
        assert isinstance(forecast["support_levels"], list)
        assert isinstance(forecast["resistance_levels"], list)
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_model_metrics(self):
        """Test getting model metrics."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        metrics = await client.get_model_metrics("BTC-USD", "hours")
        assert "accuracy" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "total_trades" in metrics
        
        # Check metric values
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["sharpe_ratio"] > 0
        assert 0 <= metrics["max_drawdown"] <= 1
        assert 0 <= metrics["win_rate"] <= 1
        assert metrics["total_trades"] > 0
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_model_metrics_different_intervals(self):
        """Test getting model metrics for different intervals."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        intervals = ["minutes", "hours", "days", "thirty"]
        for interval in intervals:
            metrics = await client.get_model_metrics("BTC-USD", interval)
            assert "accuracy" in metrics
            assert "sharpe_ratio" in metrics
            assert "max_drawdown" in metrics
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_model_metrics_ticker_not_found(self):
        """Test getting model metrics for non-existent ticker."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        with pytest.raises(ForecastingAPIError):
            await client.get_model_metrics("NONEXISTENT", "hours")
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_available_intervals(self):
        """Test getting available intervals."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        intervals = await client.get_available_intervals()
        assert isinstance(intervals, list)
        assert len(intervals) > 0
        assert "minutes" in intervals
        assert "hours" in intervals
        assert "days" in intervals
        assert "thirty" in intervals
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_market_sentiment(self):
        """Test getting market sentiment."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        sentiment = await client.get_market_sentiment()
        assert "overall_sentiment" in sentiment
        assert "sentiment_score" in sentiment
        assert "fear_greed_index" in sentiment
        assert "market_trend" in sentiment
        assert "timestamp" in sentiment
        
        # Check sentiment values
        assert sentiment["overall_sentiment"] in ["bullish", "bearish", "neutral"]
        assert 0 <= sentiment["sentiment_score"] <= 1
        assert 0 <= sentiment["fear_greed_index"] <= 100
        assert sentiment["market_trend"] in ["upward", "downward", "sideways"]
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_health_status(self):
        """Test getting health status."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        health = await client.get_health_status()
        assert "status" in health
        assert "timestamp" in health
        assert "version" in health
        assert "uptime" in health
        
        # Check health values
        assert health["status"] == "healthy"
        assert health["version"] == "1.0.0"
        assert "99.9%" in health["uptime"]
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self):
        """Test caching functionality."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        # First call should cache the result
        tickers1 = await client.get_available_tickers()
        tickers2 = await client.get_available_tickers()
        
        # Should return the same result (cached)
        assert tickers1 == tickers2
        
        # Check cache stats
        cache_stats = client.get_cache_stats()
        assert cache_stats["cache_size"] > 0
        assert "available_tickers" in cache_stats["cached_keys"]
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test clearing cache."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        # Get some data to populate cache
        await client.get_available_tickers()
        await client.get_action_recommendation("BTC-USD", "hours")
        
        # Check cache has data
        cache_stats = client.get_cache_stats()
        assert cache_stats["cache_size"] > 0
        
        # Clear cache
        client.clear_cache()
        
        # Check cache is empty
        cache_stats = client.get_cache_stats()
        assert cache_stats["cache_size"] == 0
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_set_action_recommendation(self):
        """Test setting custom action recommendation for testing."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        # Set custom action recommendation
        custom_action = {
            "action": 2,  # BUY
            "action_confidence": 0.95,
            "forecast_price": 50000.0,
            "q_values": [0.02, 0.03, 0.95]
        }
        client.set_action_recommendation("BTC-USD", "hours", custom_action)
        
        # Get the custom action
        action = await client.get_action_recommendation("BTC-USD", "hours")
        assert action["action"] == 2
        assert action["action_confidence"] == 0.95
        assert action["forecast_price"] == 50000.0
        assert action["q_values"] == [0.02, 0.03, 0.95]
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_set_model_metrics(self):
        """Test setting custom model metrics for testing."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        # Set custom model metrics
        custom_metrics = {
            "accuracy": 0.90,
            "sharpe_ratio": 2.0,
            "max_drawdown": 0.05,
            "win_rate": 0.80,
            "total_trades": 200
        }
        client.set_model_metrics("BTC-USD", "hours", custom_metrics)
        
        # Get the custom metrics
        metrics = await client.get_model_metrics("BTC-USD", "hours")
        assert metrics["accuracy"] == 0.90
        assert metrics["sharpe_ratio"] == 2.0
        assert metrics["max_drawdown"] == 0.05
        assert metrics["win_rate"] == 0.80
        assert metrics["total_trades"] == 200
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_set_market_sentiment(self):
        """Test setting custom market sentiment for testing."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        # Set custom market sentiment
        custom_sentiment = {
            "overall_sentiment": "bearish",
            "sentiment_score": 0.25,
            "fear_greed_index": 20,
            "market_trend": "downward",
            "timestamp": datetime.utcnow().isoformat()
        }
        client.set_market_sentiment(custom_sentiment)
        
        # Get the custom sentiment
        sentiment = await client.get_market_sentiment()
        assert sentiment["overall_sentiment"] == "bearish"
        assert sentiment["sentiment_score"] == 0.25
        assert sentiment["fear_greed_index"] == 20
        assert sentiment["market_trend"] == "downward"
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_clear_data(self):
        """Test clearing all mock data."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        # Get some data first
        tickers = await client.get_available_tickers()
        assert len(tickers) > 0
        
        # Clear all data
        client.clear_data()
        
        # Try to get data again (should return empty)
        tickers = await client.get_available_tickers()
        assert len(tickers) == 0
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_multiple_tickers_action_recommendations(self):
        """Test getting action recommendations for multiple tickers."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]
        interval = "hours"
        
        for ticker in tickers:
            action = await client.get_action_recommendation(ticker, interval)
            assert "action" in action
            assert "action_confidence" in action
            assert "forecast_price" in action
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_multiple_intervals_for_ticker(self):
        """Test getting data for multiple intervals for a single ticker."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        ticker = "BTC-USD"
        intervals = ["minutes", "hours", "days", "thirty"]
        
        for interval in intervals:
            action = await client.get_action_recommendation(ticker, interval)
            metrics = await client.get_model_metrics(ticker, interval)
            
            assert "action" in action
            assert "accuracy" in metrics
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_ticker(self):
        """Test error handling for invalid ticker."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        with pytest.raises(ForecastingAPIError):
            await client.get_ticker_info("INVALID-TICKER")
        
        with pytest.raises(ForecastingAPIError):
            await client.get_action_recommendation("INVALID-TICKER", "hours")
        
        with pytest.raises(ForecastingAPIError):
            await client.get_model_metrics("INVALID-TICKER", "hours")
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_interval(self):
        """Test error handling for invalid interval."""
        config = {"mock_mode": True}
        client = MockForecastingClient(config)
        await client.connect()
        
        with pytest.raises(ForecastingAPIError):
            await client.get_action_recommendation("BTC-USD", "invalid_interval")
        
        with pytest.raises(ForecastingAPIError):
            await client.get_model_metrics("BTC-USD", "invalid_interval")
        
        await client.disconnect()
