"""
Mock forecasting API implementation for testing.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from core.forecasting_client import ForecastingClient, ForecastingAPIError


class MockForecastingAPI:
    """Mock forecasting API for testing purposes."""
    
    def __init__(self):
        self.is_connected = False
        self.mock_data = {
            "tickers": [
                {"symbol": "BTC-USD", "name": "Bitcoin", "type": "crypto", "intervals": ["minutes", "hours", "days", "thirty"], "has_dqn": True},
                {"symbol": "ETH-USD", "name": "Ethereum", "type": "crypto", "intervals": ["minutes", "hours", "days", "thirty"], "has_dqn": True},
                {"symbol": "SOL-USD", "name": "Solana", "type": "crypto", "intervals": ["minutes", "hours", "days", "thirty"], "has_dqn": True},
            ],
            "actions": {
                "BTC-USD": {
                    "hours": {"action": 2, "action_confidence": 0.85, "forecast_price": 47000.0, "q_values": [0.1, 0.05, 0.85]},
                    "days": {"action": 2, "action_confidence": 0.78, "forecast_price": 50000.0, "q_values": [0.15, 0.07, 0.78]},
                    "minutes": {"action": 1, "action_confidence": 0.65, "forecast_price": 45500.0, "q_values": [0.2, 0.65, 0.15]},
                    "thirty": {"action": 2, "action_confidence": 0.82, "forecast_price": 52000.0, "q_values": [0.08, 0.10, 0.82]}
                },
                "ETH-USD": {
                    "hours": {"action": 1, "action_confidence": 0.72, "forecast_price": 2100.0, "q_values": [0.15, 0.72, 0.13]},
                    "days": {"action": 2, "action_confidence": 0.81, "forecast_price": 2300.0, "q_values": [0.10, 0.09, 0.81]},
                    "minutes": {"action": 0, "action_confidence": 0.68, "forecast_price": 1980.0, "q_values": [0.68, 0.25, 0.07]},
                    "thirty": {"action": 2, "action_confidence": 0.85, "forecast_price": 2500.0, "q_values": [0.05, 0.10, 0.85]}
                },
                "SOL-USD": {
                    "hours": {"action": 0, "action_confidence": 0.68, "forecast_price": 95.0, "q_values": [0.68, 0.25, 0.07]},
                    "days": {"action": 1, "action_confidence": 0.75, "forecast_price": 100.0, "q_values": [0.15, 0.75, 0.10]},
                    "minutes": {"action": 0, "action_confidence": 0.62, "forecast_price": 92.0, "q_values": [0.62, 0.30, 0.08]},
                    "thirty": {"action": 1, "action_confidence": 0.78, "forecast_price": 105.0, "q_values": [0.12, 0.78, 0.10]}
                }
            },
            "metrics": {
                "BTC-USD": {
                    "hours": {"accuracy": 0.82, "sharpe_ratio": 1.45, "max_drawdown": 0.12, "win_rate": 0.68, "total_trades": 150},
                    "days": {"accuracy": 0.78, "sharpe_ratio": 1.32, "max_drawdown": 0.15, "win_rate": 0.65, "total_trades": 45},
                    "minutes": {"accuracy": 0.75, "sharpe_ratio": 1.20, "max_drawdown": 0.18, "win_rate": 0.62, "total_trades": 500},
                    "thirty": {"accuracy": 0.80, "sharpe_ratio": 1.38, "max_drawdown": 0.14, "win_rate": 0.70, "total_trades": 12}
                },
                "ETH-USD": {
                    "hours": {"accuracy": 0.79, "sharpe_ratio": 1.28, "max_drawdown": 0.14, "win_rate": 0.65, "total_trades": 120},
                    "days": {"accuracy": 0.81, "sharpe_ratio": 1.41, "max_drawdown": 0.13, "win_rate": 0.68, "total_trades": 38},
                    "minutes": {"accuracy": 0.76, "sharpe_ratio": 1.15, "max_drawdown": 0.16, "win_rate": 0.60, "total_trades": 400},
                    "thirty": {"accuracy": 0.83, "sharpe_ratio": 1.45, "max_drawdown": 0.12, "win_rate": 0.72, "total_trades": 10}
                },
                "SOL-USD": {
                    "hours": {"accuracy": 0.75, "sharpe_ratio": 1.15, "max_drawdown": 0.18, "win_rate": 0.58, "total_trades": 100},
                    "days": {"accuracy": 0.77, "sharpe_ratio": 1.22, "max_drawdown": 0.16, "win_rate": 0.62, "total_trades": 32},
                    "minutes": {"accuracy": 0.73, "sharpe_ratio": 1.08, "max_drawdown": 0.20, "win_rate": 0.55, "total_trades": 350},
                    "thirty": {"accuracy": 0.79, "sharpe_ratio": 1.30, "max_drawdown": 0.15, "win_rate": 0.65, "total_trades": 8}
                }
            },
            "market_sentiment": {
                "overall_sentiment": "bullish",
                "sentiment_score": 0.65,
                "fear_greed_index": 72,
                "market_trend": "upward",
                "timestamp": datetime.utcnow().isoformat()
            },
            "health": {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "uptime": "99.9%"
            }
        }
    
    async def connect(self) -> None:
        """Mock connection."""
        self.is_connected = True
    
    async def disconnect(self) -> None:
        """Mock disconnection."""
        self.is_connected = False
    
    async def get_available_tickers(self) -> List[Dict[str, Any]]:
        """Get mock available tickers."""
        return self.mock_data["tickers"]
    
    async def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Get mock ticker info."""
        ticker_data = next((t for t in self.mock_data["tickers"] if t["symbol"] == ticker), None)
        if not ticker_data:
            raise ForecastingAPIError(f"Ticker {ticker} not found")
        return ticker_data
    
    async def get_action_recommendation(self, ticker: str, interval: str) -> Dict[str, Any]:
        """Get mock action recommendation."""
        if ticker not in self.mock_data["actions"]:
            raise ForecastingAPIError(f"Ticker {ticker} not found")
        
        if interval not in self.mock_data["actions"][ticker]:
            raise ForecastingAPIError(f"Interval {interval} not found for {ticker}")
        
        action_data = self.mock_data["actions"][ticker][interval]
        return {
            **action_data,
            "current_price": 45000.0 if "BTC" in ticker else 2000.0 if "ETH" in ticker else 100.0,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_stock_forecast(self, ticker: str, interval: str) -> Dict[str, Any]:
        """Get mock stock forecast."""
        action_data = await self.get_action_recommendation(ticker, interval)
        return {
            "ticker": ticker,
            "interval": interval,
            "forecast_price": action_data["forecast_price"],
            "confidence": action_data["action_confidence"],
            "trend": "bullish" if action_data["action"] == 2 else "bearish" if action_data["action"] == 0 else "neutral",
            "support_levels": [action_data["forecast_price"] * 0.95, action_data["forecast_price"] * 0.90],
            "resistance_levels": [action_data["forecast_price"] * 1.05, action_data["forecast_price"] * 1.10],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_model_metrics(self, ticker: str, interval: str) -> Dict[str, Any]:
        """Get mock model metrics."""
        if ticker not in self.mock_data["metrics"]:
            raise ForecastingAPIError(f"Ticker {ticker} not found")
        
        if interval not in self.mock_data["metrics"][ticker]:
            raise ForecastingAPIError(f"Interval {interval} not found for {ticker}")
        
        return self.mock_data["metrics"][ticker][interval]
    
    async def get_available_intervals(self) -> List[str]:
        """Get mock available intervals."""
        return ["minutes", "hours", "days", "thirty"]
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """Get mock market sentiment."""
        return self.mock_data["market_sentiment"]
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get mock health status."""
        return self.mock_data["health"]
    
    def set_action_recommendation(self, ticker: str, interval: str, action_data: Dict[str, Any]) -> None:
        """Set mock action recommendation for testing."""
        if ticker not in self.mock_data["actions"]:
            self.mock_data["actions"][ticker] = {}
        self.mock_data["actions"][ticker][interval] = action_data
    
    def set_model_metrics(self, ticker: str, interval: str, metrics: Dict[str, Any]) -> None:
        """Set mock model metrics for testing."""
        if ticker not in self.mock_data["metrics"]:
            self.mock_data["metrics"][ticker] = {}
        self.mock_data["metrics"][ticker][interval] = metrics
    
    def set_market_sentiment(self, sentiment: Dict[str, Any]) -> None:
        """Set mock market sentiment for testing."""
        self.mock_data["market_sentiment"] = sentiment
    
    def clear_data(self) -> None:
        """Clear all mock data for testing."""
        self.mock_data = {
            "tickers": [],
            "actions": {},
            "metrics": {},
            "market_sentiment": {},
            "health": {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        }


class MockForecastingClient(ForecastingClient):
    """Mock forecasting client that uses MockForecastingAPI internally."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mock_api = MockForecastingAPI()
    
    async def connect(self) -> None:
        """Connect to mock API."""
        await self.mock_api.connect()
        self.is_connected = True
    
    async def disconnect(self) -> None:
        """Disconnect from mock API."""
        await self.mock_api.disconnect()
        self.is_connected = False
    
    async def get_available_tickers(self) -> List[Dict[str, Any]]:
        """Get available tickers from mock API."""
        return await self.mock_api.get_available_tickers()
    
    async def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Get ticker info from mock API."""
        return await self.mock_api.get_ticker_info(ticker)
    
    async def get_action_recommendation(self, ticker: str, interval: str) -> Dict[str, Any]:
        """Get action recommendation from mock API."""
        return await self.mock_api.get_action_recommendation(ticker, interval)
    
    async def get_stock_forecast(self, ticker: str, interval: str) -> Dict[str, Any]:
        """Get stock forecast from mock API."""
        return await self.mock_api.get_stock_forecast(ticker, interval)
    
    async def get_model_metrics(self, ticker: str, interval: str) -> Dict[str, Any]:
        """Get model metrics from mock API."""
        return await self.mock_api.get_model_metrics(ticker, interval)
    
    async def get_available_intervals(self) -> List[str]:
        """Get available intervals from mock API."""
        return await self.mock_api.get_available_intervals()
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """Get market sentiment from mock API."""
        return await self.mock_api.get_market_sentiment()
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status from mock API."""
        return await self.mock_api.get_health_status()
    
    def set_action_recommendation(self, ticker: str, interval: str, action_data: Dict[str, Any]) -> None:
        """Set action recommendation for testing."""
        self.mock_api.set_action_recommendation(ticker, interval, action_data)
    
    def set_model_metrics(self, ticker: str, interval: str, metrics: Dict[str, Any]) -> None:
        """Set model metrics for testing."""
        self.mock_api.set_model_metrics(ticker, interval, metrics)
    
    def set_market_sentiment(self, sentiment: Dict[str, Any]) -> None:
        """Set market sentiment for testing."""
        self.mock_api.set_market_sentiment(sentiment)
    
    def clear_data(self) -> None:
        """Clear all mock data for testing."""
        self.mock_api.clear_data()
