# Testing Guide

This guide covers the comprehensive testing strategy for the Agentic Trading System, including unit tests, integration tests, and end-to-end testing.

## Testing Philosophy

Our testing strategy follows the testing pyramid:

1. **Unit Tests** (70%): Fast, isolated tests for individual components
2. **Integration Tests** (20%): Tests for component interactions
3. **Functional Tests** (10%): End-to-end workflow tests

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_config.py       # Configuration tests
│   ├── test_exchange_interface.py
│   ├── test_forecasting_client.py
│   └── ...
├── integration/             # Integration tests
│   ├── test_exchange_manager.py
│   ├── test_redis_integration.py
│   └── ...
├── functional/              # Functional tests
│   ├── test_trading_system.py
│   ├── test_full_trading_flow.py
│   └── ...
└── mocks/                   # Mock implementations
    ├── mock_dex_exchange.py
    ├── mock_mexc_exchange.py
    └── mock_forecasting_api.py
```

## Running Tests

### All Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=api --cov=agents --cov=core --cov-report=html

# Run with verbose output
uv run pytest -v
```

### Specific Test Categories

```bash
# Unit tests only
uv run pytest tests/unit/

# Integration tests only
uv run pytest tests/integration/

# Functional tests only
uv run pytest tests/functional/

# Specific test file
uv run pytest tests/unit/test_config.py

# Specific test function
uv run pytest tests/unit/test_config.py::test_settings_validation
```

### Test Filtering

```bash
# Run tests matching pattern
uv run pytest -k "test_trade"

# Run tests with specific marker
uv run pytest -m "slow"

# Skip slow tests
uv run pytest -m "not slow"
```

## Test Configuration

### pytest.ini

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --cov=api
    --cov=agents
    --cov=core
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=80
asyncio_mode = auto
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    functional: marks tests as functional tests
    unit: marks tests as unit tests
```

### Test Fixtures

The `conftest.py` file provides common fixtures:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from core.redis_client import RedisClient
from core.exchange_manager import ExchangeManager
from core.forecasting_client import ForecastingClient

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mock_redis():
    """Mock Redis client."""
    redis = AsyncMock(spec=RedisClient)
    redis.get_json.return_value = {}
    redis.set_json.return_value = True
    redis.publish.return_value = True
    redis.subscribe.return_value = AsyncMock()
    return redis

@pytest.fixture
async def mock_exchange_manager():
    """Mock exchange manager."""
    manager = AsyncMock(spec=ExchangeManager)
    manager.place_order.return_value = MagicMock()
    manager.get_balance.return_value = {"USDC": 10000.0}
    return manager

@pytest.fixture
async def mock_forecasting_client():
    """Mock forecasting client."""
    client = AsyncMock(spec=ForecastingClient)
    client.get_action_recommendation.return_value = {
        "recommendation": "BUY",
        "confidence": 0.85
    }
    client.get_stock_forecast.return_value = {
        "forecast": [{"price": 50000.0, "confidence": 0.85}]
    }
    return client
```

## Unit Tests

### Configuration Tests

```python
# tests/unit/test_config.py
import pytest
from core.config import settings

def test_settings_validation():
    """Test that settings are properly validated."""
    assert settings.app_name == "Agentic Trading System"
    assert settings.initial_capital > 0
    assert 0 < settings.max_position_size <= 1

def test_environment_variables():
    """Test environment variable loading."""
    import os
    os.environ["INITIAL_CAPITAL"] = "5000.0"
    from core.config import Settings
    new_settings = Settings()
    assert new_settings.initial_capital == 5000.0
```

### Exchange Interface Tests

```python
# tests/unit/test_exchange_interface.py
import pytest
from core.exchange_interface import AbstractExchange, TradeAction, OrderType

class TestExchange(AbstractExchange):
    async def connect(self):
        pass
    
    async def disconnect(self):
        pass
    
    async def place_order(self, symbol, side, order_type, amount):
        return MagicMock()

@pytest.mark.asyncio
async def test_exchange_interface():
    """Test exchange interface implementation."""
    exchange = TestExchange()
    await exchange.connect()
    
    order = await exchange.place_order(
        symbol="BTCUSDC",
        side=TradeAction.BUY,
        order_type=OrderType.MARKET,
        amount=0.1
    )
    
    assert order is not None
    await exchange.disconnect()
```

### Forecasting Client Tests

```python
# tests/unit/test_forecasting_client.py
import pytest
import respx
from core.forecasting_client import ForecastingClient

@pytest.mark.asyncio
async def test_get_action_recommendation():
    """Test getting action recommendation from forecasting API."""
    with respx.mock:
        respx.get("https://forecasting.guidry-cloud.com/api/action_recommendation").mock(
            return_value=httpx.Response(
                200,
                json={
                    "recommendation": "BUY",
                    "confidence": 0.85
                }
            )
        )
        
        client = ForecastingClient()
        result = await client.get_action_recommendation("BTC-USD")
        
        assert result["recommendation"] == "BUY"
        assert result["confidence"] == 0.85
```

## Integration Tests

### Exchange Manager Tests

```python
# tests/integration/test_exchange_manager.py
import pytest
from core.exchange_manager import ExchangeManager
from tests.mocks.mock_dex_exchange import MockDEXExchange
from tests.mocks.mock_mexc_exchange import MockMEXCExchange

@pytest.mark.asyncio
async def test_exchange_manager_initialization():
    """Test exchange manager initialization."""
    manager = ExchangeManager()
    
    # Register mock exchanges
    dex_exchange = MockDEXExchange()
    mexc_exchange = MockMEXCExchange()
    
    manager.register_exchange("DEX", dex_exchange)
    manager.register_exchange("MEXC", mexc_exchange)
    
    await manager.initialize()
    
    assert manager.is_connected()
    assert "DEX" in manager.exchanges
    assert "MEXC" in manager.exchanges

@pytest.mark.asyncio
async def test_place_order():
    """Test placing orders through exchange manager."""
    manager = ExchangeManager()
    dex_exchange = MockDEXExchange()
    manager.register_exchange("DEX", dex_exchange)
    await manager.initialize()
    
    order = await manager.place_order(
        symbol="BTCUSDC",
        side=TradeAction.BUY,
        order_type=OrderType.MARKET,
        amount=0.1,
        exchange_type=ExchangeType.DEX
    )
    
    assert order is not None
    assert order.symbol == "BTCUSDC"
    assert order.side == TradeAction.BUY
```

### Redis Integration Tests

```python
# tests/integration/test_redis_integration.py
import pytest
import fakeredis
from core.redis_client import RedisClient

@pytest.mark.asyncio
async def test_redis_connection():
    """Test Redis connection and basic operations."""
    redis = RedisClient(host="localhost", port=6379, db=0)
    
    # Use fakeredis for testing
    redis.redis = fakeredis.FakeAsyncRedis()
    
    await redis.connect()
    
    # Test basic operations
    await redis.set("test_key", "test_value")
    value = await redis.get("test_key")
    assert value == "test_value"
    
    # Test JSON operations
    data = {"key": "value", "number": 42}
    await redis.set_json("test_json", data)
    retrieved = await redis.get_json("test_json")
    assert retrieved == data
    
    await redis.disconnect()
```

## Functional Tests

### Trading System Tests

```python
# tests/functional/test_trading_system.py
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.functional
@pytest.mark.asyncio
async def test_trading_system_initialization():
    """Test complete trading system initialization."""
    # This test would initialize the entire system
    # and verify all components are working together
    pass

@pytest.mark.functional
@pytest.mark.asyncio
async def test_market_data_flow():
    """Test market data flow through the system."""
    # Test that market data flows from exchanges
    # through agents to the API
    pass

@pytest.mark.functional
@pytest.mark.asyncio
async def test_trading_decision_flow():
    """Test complete trading decision flow."""
    # Test that signals from agents are aggregated
    # and trading decisions are made correctly
    pass
```

### Full Trading Flow Tests

```python
# tests/functional/test_full_trading_flow.py
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.functional
@pytest.mark.asyncio
async def test_complete_trading_flow():
    """Test complete trading flow from signal to execution."""
    # 1. Market data arrives
    # 2. Agents process data
    # 3. Signals are generated
    # 4. Orchestrator makes decision
    # 5. Trade is executed
    # 6. Portfolio is updated
    pass

@pytest.mark.functional
@pytest.mark.asyncio
async def test_error_recovery():
    """Test system recovery from errors."""
    # Test that the system recovers gracefully
    # from various error conditions
    pass
```

## Mock Implementations

### Mock DEX Exchange

```python
# tests/mocks/mock_dex_exchange.py
from core.exchange_interface import AbstractExchange, TradeAction, OrderType, Order
from typing import Dict, Any

class MockDEXExchange(AbstractExchange):
    def __init__(self):
        self.connected = False
        self.balances = {"USDC": 10000.0, "BTC": 0.0}
        self.orders = []
    
    async def connect(self):
        self.connected = True
    
    async def disconnect(self):
        self.connected = False
    
    async def place_order(self, symbol: str, side: TradeAction, order_type: OrderType, amount: float) -> Order:
        if not self.connected:
            raise ConnectionError("Not connected")
        
        # Simulate order execution
        order = Order(
            id=f"order_{len(self.orders)}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=50000.0,  # Mock price
            status="filled",
            timestamp=datetime.utcnow()
        )
        
        self.orders.append(order)
        return order
    
    async def get_balance(self) -> Dict[str, float]:
        return self.balances.copy()
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "price": 50000.0,
            "volume": 1000000.0,
            "timestamp": datetime.utcnow()
        }
```

### Mock Forecasting API

```python
# tests/mocks/mock_forecasting_api.py
from core.forecasting_client import ForecastingClient
from typing import Dict, Any

class MockForecastingClient(ForecastingClient):
    def __init__(self):
        self.connected = False
        self.config = {"model_version": "v1.0.0"}
    
    async def initialize(self):
        self.connected = True
    
    async def get_action_recommendation(self, ticker: str, interval: str = "hours") -> Dict[str, Any]:
        if not self.connected:
            raise ConnectionError("Not connected")
        
        return {
            "recommendation": "BUY",
            "confidence": 0.85,
            "ticker": ticker,
            "interval": interval
        }
    
    async def get_stock_forecast(self, ticker: str, interval: str = "hours") -> Dict[str, Any]:
        if not self.connected:
            raise ConnectionError("Not connected")
        
        return {
            "ticker": ticker,
            "forecast": [
                {"timestamp": "2024-01-01T01:00:00Z", "price": 51000.0, "confidence": 0.85}
            ],
            "interval": interval,
            "model_version": "v1.0.0"
        }
```

## Test Data Management

### Fixture Data

```python
# tests/fixtures/trading_data.py
import pytest
from datetime import datetime, timedelta

@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "BTC": {
            "price": 50000.0,
            "volume": 1000000.0,
            "change_24h": 0.05,
            "high_24h": 52000.0,
            "low_24h": 48000.0,
            "timestamp": datetime.utcnow()
        }
    }

@pytest.fixture
def sample_trade_data():
    """Sample trade data for testing."""
    return {
        "id": "trade_123",
        "ticker": "BTC",
        "action": "BUY",
        "quantity": 0.1,
        "price": 50000.0,
        "timestamp": datetime.utcnow()
    }

@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing."""
    return {
        "balance_usdc": 5000.0,
        "holdings": {"BTC": 0.1, "ETH": 1.0},
        "total_value_usdc": 10000.0,
        "daily_pnl": 100.0,
        "total_pnl": 1000.0
    }
```

## Performance Testing

### Load Testing

```python
# tests/performance/test_load.py
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

@pytest.mark.slow
@pytest.mark.asyncio
async def test_api_load():
    """Test API under load."""
    import httpx
    
    async def make_request():
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            return response.status_code
    
    # Run 100 concurrent requests
    tasks = [make_request() for _ in range(100)]
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    # All requests should succeed
    assert all(status == 200 for status in results)
    
    # Should complete within reasonable time
    assert end_time - start_time < 10.0
```

### Memory Testing

```python
# tests/performance/test_memory.py
import pytest
import psutil
import os

@pytest.mark.slow
def test_memory_usage():
    """Test memory usage during operation."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Run some operations
    # ... your code here ...
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable
    assert memory_increase < 100 * 1024 * 1024  # 100MB
```

## Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install UV
      run: curl -LsSf https://astral.sh/uv/install.sh | sh
    
    - name: Install dependencies
      run: uv sync
    
    - name: Run tests
      run: uv run pytest --cov=api --cov=agents --cov=core --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Test Best Practices

### 1. Test Isolation

- Each test should be independent
- Use fixtures for setup and teardown
- Mock external dependencies
- Clean up after each test

### 2. Test Naming

- Use descriptive test names
- Follow the pattern: `test_<functionality>_<scenario>`
- Examples:
  - `test_place_order_success`
  - `test_place_order_insufficient_balance`
  - `test_get_balance_connection_error`

### 3. Assertions

- Use specific assertions
- Test both positive and negative cases
- Verify error conditions
- Check return values and side effects

### 4. Async Testing

- Use `@pytest.mark.asyncio` for async tests
- Use `AsyncMock` for mocking async functions
- Handle async context managers properly

### 5. Test Data

- Use realistic test data
- Create reusable fixtures
- Avoid hardcoded values
- Use factories for complex objects

## Debugging Tests

### Running Specific Tests

```bash
# Run specific test with verbose output
uv run pytest tests/unit/test_config.py::test_settings_validation -v

# Run tests with debugging
uv run pytest --pdb tests/unit/test_config.py

# Run tests with logging
uv run pytest --log-cli-level=DEBUG tests/unit/test_config.py
```

### Test Debugging

```python
import pytest
import logging

def test_with_debugging():
    """Test with debugging information."""
    logging.basicConfig(level=logging.DEBUG)
    
    # Your test code here
    result = some_function()
    
    # Add breakpoints
    import pdb; pdb.set_trace()
    
    assert result == expected_value
```

## Coverage Requirements

- **Overall Coverage**: 80%+
- **Critical Components**: 90%+
- **New Code**: 95%+

### Coverage Exclusions

```python
# .coveragerc
[run]
omit = 
    tests/*
    */migrations/*
    */venv/*
    */env/*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## Test Maintenance

### Regular Updates

- Update tests when code changes
- Remove obsolete tests
- Refactor duplicate test code
- Update test data regularly

### Performance Monitoring

- Monitor test execution time
- Identify slow tests
- Optimize test performance
- Use parallel execution when possible

### Documentation

- Document test scenarios
- Explain complex test logic
- Keep test documentation up to date
- Include test examples in documentation
