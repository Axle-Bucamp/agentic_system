"""
Pytest configuration and shared fixtures.
"""
import json
from collections import defaultdict
import asyncio
import pytest
import pytest_asyncio
from typing import Dict, Any, AsyncGenerator, List
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta
from core.forecasting_client import ForecastingClient
import httpx

try:
    import fakeredis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    fakeredis = None
from core.exchange_interface import (
    ExchangeType,
    OrderSide,
    OrderType,
    OrderStatus,
    Balance,
    Ticker,
    Order,
)
from core.exchange_manager import ExchangeManager
from core.redis_client import RedisClient


class FakeRedis:
    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._lists: Dict[str, List[str]] = defaultdict(list)

    async def get_json(self, key: str):
        value = self._store.get(key)
        if value is None:
            return None
        return json.loads(json.dumps(value))

    async def set_json(self, key: str, value, expire: int | None = None):
        self._store[key] = json.loads(json.dumps(value))

    async def lpush(self, key: str, *values: str):
        for value in values:
            self._lists[key].insert(0, value)

    async def rpush(self, key: str, value: str):
        self._lists[key].append(value)

    async def ltrim(self, key: str, start: int, end: int):
        if end == -1:
            trimmed = self._lists[key][start:]
        else:
            trimmed = self._lists[key][start : end + 1]
        self._lists[key] = trimmed

    async def lrange(self, key: str, start: int, end: int):
        if end == -1:
            end = None
        else:
            end += 1
        return self._lists[key][start:end]

    async def delete(self, key: str):
        self._store.pop(key, None)
        self._lists.pop(key, None)

    async def set(self, key: str, value: str, expire: int | None = None):
        self._store[key] = value

    async def get(self, key: str):
        return self._store.get(key)

    async def exists(self, key: str) -> bool:
        return key in self._store or key in self._lists

    async def ping(self):
        return True

    async def hset(self, name: str, key: str, value: str):
        bucket = self._store.setdefault(name, {})
        if not isinstance(bucket, dict):
            bucket = {}
        bucket[key] = value
        self._store[name] = bucket

    async def hgetall(self, name: str):
        bucket = self._store.get(name, {})
        if isinstance(bucket, dict):
            return bucket.copy()
        return {}

    async def hdel(self, name: str, key: str):
        bucket = self._store.get(name)
        if isinstance(bucket, dict):
            bucket.pop(key, None)


@pytest_asyncio.fixture
async def fake_redis():
    return FakeRedis()


@pytest_asyncio.fixture(autouse=True)
async def patch_api_redis_client(fake_redis, monkeypatch):
    from api import main as api_main

    async def get_json(key, *_args, **_kwargs):
        return await fake_redis.get_json(key)

    async def set_json(key, value, expire=None):
        await fake_redis.set_json(key, value, expire)

    async def set_(key, value, expire=None):
        await fake_redis.set(key, value, expire)

    async def get_(key, *_args, **_kwargs):
        return await fake_redis.get(key)

    async def delete(key):
        await fake_redis.delete(key)

    async def hset(name, key, value):
        await fake_redis.hset(name, key, value)

    async def hgetall(name):
        return await fake_redis.hgetall(name)

    async def hdel(name, key):
        await fake_redis.hdel(name, key)

    monkeypatch.setattr(api_main.redis_client, "get_json", get_json)
    monkeypatch.setattr(api_main.redis_client, "set_json", set_json)
    monkeypatch.setattr(api_main.redis_client, "set", set_)
    monkeypatch.setattr(api_main.redis_client, "get", get_)
    monkeypatch.setattr(api_main.redis_client, "delete", delete)
    monkeypatch.setattr(api_main.redis_client, "hset", hset)
    monkeypatch.setattr(api_main.redis_client, "hgetall", hgetall)
    monkeypatch.setattr(api_main.redis_client, "hdel", hdel)
    yield


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_redis() -> AsyncGenerator[RedisClient, None]:
    """Create a mock Redis client for testing."""
    if fakeredis is None:
        pytest.skip("fakeredis is required for mock_redis fixture")
    redis_client = RedisClient()
    redis_client.redis = fakeredis.FakeAsyncRedis()
    yield redis_client


@pytest.fixture
async def mock_forecasting_client() -> AsyncGenerator[ForecastingClient, None]:
    """Create a mock forecasting client for testing."""
    config = {
        "base_url": "https://forecasting.guidry-cloud.com",
        "api_key": "test_key",
        "mock_mode": True
    }
    client = ForecastingClient(config)
    await client.connect()
    yield client
    await client.disconnect()


@pytest.fixture
async def mock_exchange_manager() -> AsyncGenerator[ExchangeManager, None]:
    """Create a mock exchange manager for testing."""
    manager = ExchangeManager()
    manager.set_paper_trading(True)  # Enable paper trading for tests
    yield manager


@pytest.fixture
def mock_http_client() -> MagicMock:
    """Create a mock HTTP client for testing."""
    client = MagicMock(spec=httpx.AsyncClient)
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.delete = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def sample_balance() -> Balance:
    """Create a sample balance for testing."""
    return Balance(
        asset="USDT",
        free=10000.0,
        locked=0.0,
        total=10000.0
    )


@pytest.fixture
def sample_ticker() -> Ticker:
    """Create a sample ticker for testing."""
    return Ticker(
        symbol="BTCUSDT",
        price=45000.0,
        volume=1000000.0,
        timestamp=datetime.utcnow(),
        bid=44950.0,
        ask=45050.0,
        high_24h=46000.0,
        low_24h=44000.0,
        change_24h=1000.0,
        change_percent_24h=2.27
    )


@pytest.fixture
def sample_order() -> Order:
    """Create a sample order for testing."""
    return Order(
        id="test_order_123",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        amount=0.001,
        price=45000.0,
        status=OrderStatus.FILLED,
        filled_amount=0.001,
        remaining_amount=0.0,
        average_price=45000.0,
        fee=0.045,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        client_order_id="test_client_123"
    )


@pytest.fixture
def sample_forecasting_response() -> Dict[str, Any]:
    """Create a sample forecasting API response for testing."""
    return {
        "action": 2,  # BUY
        "action_confidence": 0.85,
        "forecast_price": 47000.0,
        "q_values": [0.1, 0.05, 0.85],  # [SELL, HOLD, BUY]
        "current_price": 45000.0,
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def sample_market_data() -> Dict[str, Any]:
    """Create sample market data for testing."""
    return {
        "BTCUSDT": {
            "price": 45000.0,
            "volume": 1000000.0,
            "timestamp": datetime.utcnow().isoformat(),
            "bid": 44950.0,
            "ask": 45050.0,
            "high_24h": 46000.0,
            "low_24h": 44000.0,
            "change_24h": 1000.0,
            "change_percent_24h": 2.27
        },
        "ETHUSDT": {
            "price": 2000.0,
            "volume": 500000.0,
            "timestamp": datetime.utcnow().isoformat(),
            "bid": 1995.0,
            "ask": 2005.0,
            "high_24h": 2100.0,
            "low_24h": 1950.0,
            "change_24h": 50.0,
            "change_percent_24h": 2.56
        }
    }


@pytest.fixture
def sample_portfolio_data() -> Dict[str, Any]:
    """Create sample portfolio data for testing."""
    return {
        "balance_usdc": 10000.0,
        "holdings": {
            "BTC": 0.1,
            "ETH": 2.0,
            "SOL": 50.0
        },
        "total_value_usdc": 15000.0,
        "daily_pnl": 500.0,
        "total_pnl": 2000.0,
        "positions": [
            {
                "ticker": "BTC",
                "quantity": 0.1,
                "avg_price": 40000.0,
                "current_price": 45000.0,
                "unrealized_pnl": 500.0
            },
            {
                "ticker": "ETH",
                "quantity": 2.0,
                "avg_price": 1800.0,
                "current_price": 2000.0,
                "unrealized_pnl": 400.0
            }
        ],
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def sample_agent_signal() -> Dict[str, Any]:
    """Create a sample agent signal for testing."""
    return {
        "agent_type": "DQN",
        "signal_type": "DQN_PREDICTION",
        "ticker": "BTCUSDT",
        "action": "BUY",
        "confidence": 0.85,
        "data": {
            "forecast_price": 47000.0,
            "q_values": [0.1, 0.05, 0.85],
            "intervals_analyzed": ["hours", "days"]
        },
        "reasoning": "Strong bullish signal from DQN model",
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def sample_trade_decision() -> Dict[str, Any]:
    """Create a sample trade decision for testing."""
    return {
        "ticker": "BTCUSDT",
        "action": "BUY",
        "quantity": 0.001,
        "expected_price": 45000.0,
        "confidence": 0.85,
        "reasoning": "DQN: BUY (0.85), Chart: BUY (0.72), Risk: APPROVED",
        "contributing_signals": [],
        "risk_approved": True,
        "requires_human_validation": False,
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """Create a mock LLM response for testing."""
    return {
        "sentiment": "bullish",
        "confidence": 0.78,
        "summary": "Positive market sentiment with strong buying pressure",
        "key_points": [
            "Institutional adoption increasing",
            "Technical indicators showing bullish signals",
            "Regulatory clarity improving"
        ],
        "risk_factors": [
            "Market volatility remains high",
            "Regulatory uncertainty in some jurisdictions"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def sample_technical_indicators() -> Dict[str, Any]:
    """Create sample technical indicators for testing."""
    return {
        "rsi": 65.5,
        "macd": 150.0,
        "macd_signal": 120.0,
        "bollinger_upper": 46000.0,
        "bollinger_lower": 44000.0,
        "volume_sma": 800000.0,
        "recommendation": "BUY",
        "strength": 0.72,
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def sample_risk_metrics() -> Dict[str, Any]:
    """Create sample risk metrics for testing."""
    return {
        "var_95": 0.05,
        "cvar_95": 0.08,
        "position_size": 0.15,
        "max_position_size": 0.20,
        "current_drawdown": 0.03,
        "max_drawdown": 0.12,
        "daily_pnl": 250.0,
        "risk_level": "MEDIUM",
        "warnings": [],
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def mock_news_data() -> Dict[str, Any]:
    """Create mock news data for testing."""
    return {
        "articles": [
            {
                "title": "Bitcoin Reaches New All-Time High",
                "content": "Bitcoin has reached a new all-time high of $50,000...",
                "sentiment": "positive",
                "confidence": 0.85,
                "source": "CoinDesk",
                "published_at": datetime.utcnow().isoformat()
            },
            {
                "title": "Ethereum 2.0 Upgrade Shows Promising Results",
                "content": "The Ethereum 2.0 upgrade is showing promising results...",
                "sentiment": "positive",
                "confidence": 0.78,
                "source": "Ethereum Foundation",
                "published_at": (datetime.utcnow() - timedelta(hours=2)).isoformat()
            }
        ],
        "overall_sentiment": "bullish",
        "sentiment_score": 0.72,
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def sample_performance_metrics() -> Dict[str, Any]:
    """Create sample performance metrics for testing."""
    return {
        "total_trades": 150,
        "winning_trades": 95,
        "losing_trades": 55,
        "win_rate": 0.633,
        "total_pnl": 2500.0,
        "sharpe_ratio": 1.45,
        "max_drawdown": 0.12,
        "current_drawdown": 0.03,
        "roi": 0.25,
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def mock_web3() -> MagicMock:
    """Create a mock Web3 instance for testing."""
    web3 = MagicMock()
    web3.is_connected.return_value = True
    web3.eth.get_balance.return_value = 1000000000000000000  # 1 ETH in wei
    web3.eth.gas_price = 20000000000  # 20 gwei
    return web3


@pytest.fixture
def mock_account() -> MagicMock:
    """Create a mock Ethereum account for testing."""
    account = MagicMock()
    account.address = "0x1234567890123456789012345678901234567890"
    account.private_key = b"test_private_key"
    return account


@pytest.fixture
def sample_dex_config() -> Dict[str, Any]:
    """Create sample DEX configuration for testing."""
    return {
        "network": "ethereum",
        "rpc_url": "https://eth-mainnet.g.alchemy.com/v2/test",
        "chain_id": 1,
        "private_key": "test_private_key",
        "wallet_address": "0x1234567890123456789012345678901234567890",
        "dex_type": "uniswap_v3",
        "router_address": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        "factory_address": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "weth_address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "slippage_tolerance": 0.005,
        "gas_limit": 300000,
        "mock_mode": True
    }


@pytest.fixture
def sample_mexc_config() -> Dict[str, Any]:
    """Create sample MEXC configuration for testing."""
    return {
        "api_key": "test_api_key",
        "secret_key": "test_secret_key",
        "base_url": "https://api.mexc.com",
        "rate_limit_delay": 0.1,
        "mock_mode": True
    }


@pytest.fixture
def sample_forecasting_config() -> Dict[str, Any]:
    """Create sample forecasting API configuration for testing."""
    return {
        "base_url": "https://forecasting.guidry-cloud.com",
        "api_key": "test_forecasting_key",
        "timeout": 30.0,
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "mock_mode": True
    }


@pytest.fixture
def mock_redis_data() -> Dict[str, Any]:
    """Create mock Redis data for testing."""
    return {
        "state:portfolio": {
            "balance_usdc": 10000.0,
            "holdings": {"BTC": 0.1, "ETH": 2.0},
            "total_value_usdc": 15000.0,
            "daily_pnl": 500.0,
            "total_pnl": 2000.0,
            "positions": []
        },
        "market:BTCUSDT": {
            "price": 45000.0,
            "volume": 1000000.0,
            "timestamp": datetime.utcnow().isoformat()
        },
        "dqn:prediction:BTC": {
            "action": "BUY",
            "confidence": 0.85,
            "forecast_price": 47000.0,
            "timestamp": datetime.utcnow().isoformat()
        }
    }


@pytest.fixture
async def setup_redis_data(mock_redis: RedisClient, mock_redis_data: Dict[str, Any]) -> None:
    """Setup Redis with test data."""
    for key, value in mock_redis_data.items():
        await mock_redis.set_json(key, value)


@pytest.fixture
def mock_httpx_response() -> MagicMock:
    """Create a mock httpx response for testing."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"status": "success"}
    response.raise_for_status.return_value = None
    return response


@pytest.fixture
def mock_httpx_client(mock_httpx_response: MagicMock) -> MagicMock:
    """Create a mock httpx client for testing."""
    client = MagicMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=mock_httpx_response)
    client.post = AsyncMock(return_value=mock_httpx_response)
    client.delete = AsyncMock(return_value=mock_httpx_response)
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def sample_error_response() -> Dict[str, Any]:
    """Create a sample error response for testing."""
    return {
        "error": "Invalid request",
        "code": 400,
        "message": "The request parameters are invalid",
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def mock_circuit_breaker() -> MagicMock:
    """Create a mock circuit breaker for testing."""
    circuit_breaker = MagicMock()
    circuit_breaker.call.return_value = "success"
    circuit_breaker.state = "CLOSED"
    return circuit_breaker


@pytest.fixture
def sample_alert_data() -> Dict[str, Any]:
    """Create sample alert data for testing."""
    return {
        "alert_type": "risk_warning",
        "severity": "medium",
        "message": "Portfolio risk level has increased",
        "ticker": "BTCUSDT",
        "current_value": 0.15,
        "threshold": 0.20,
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def mock_prometheus_metrics() -> MagicMock:
    """Create mock Prometheus metrics for testing."""
    metrics = MagicMock()
    metrics.trade_counter = MagicMock()
    metrics.trade_counter.inc.return_value = None
    metrics.trade_value_histogram = MagicMock()
    metrics.trade_value_histogram.observe.return_value = None
    metrics.error_counter = MagicMock()
    metrics.error_counter.inc.return_value = None
    return metrics
