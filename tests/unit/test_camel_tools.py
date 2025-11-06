"""
Unit tests for CAMEL tools integration.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.camel_tools.mcp_forecasting_toolkit import MCPForecastingToolkit
from core.camel_tools.dex_trading_toolkit import DEXTradingToolkit
from core.camel_tools.market_data_toolkit import MarketDataToolkit


@pytest.fixture
def mock_forecasting_client():
    """Mock forecasting client."""
    client = AsyncMock()
    client.connect = AsyncMock()
    client.get_stock_forecast = AsyncMock(return_value={"forecast_price": 50000.0})
    client.get_action_recommendation = AsyncMock(return_value={"action": 2, "action_confidence": 0.8})
    client.get_available_tickers = AsyncMock(return_value=[{"symbol": "BTC-USD", "has_dqn": True}])
    client.get_model_metrics = AsyncMock(return_value={"accuracy": 0.85})
    return client


@pytest.fixture
def mock_dex_client():
    """Mock DEX client."""
    client = AsyncMock()
    client.connect = AsyncMock()
    client.buy_asset = AsyncMock(return_value={"success": True})
    client.sell_asset = AsyncMock(return_value={"success": True})
    client.get_portfolio_status = AsyncMock(return_value={"balance_usdc": 1000.0})
    client.get_ticker_prices = AsyncMock(return_value={"BTC": 50000.0})
    return client


@pytest.mark.asyncio
async def test_mcp_forecasting_toolkit_initialization(mock_forecasting_client):
    """Test MCP forecasting toolkit initialization."""
    toolkit = MCPForecastingToolkit(forecasting_client=mock_forecasting_client)
    await toolkit.initialize()
    
    assert toolkit.forecasting_client is not None
    mock_forecasting_client.connect.assert_called_once()


@pytest.mark.asyncio
async def test_get_stock_forecast_tool(mock_forecasting_client):
    """Test get stock forecast tool."""
    toolkit = MCPForecastingToolkit(forecasting_client=mock_forecasting_client)
    await toolkit.initialize()
    
    tool = toolkit.get_stock_forecast_tool()
    assert tool is not None
    assert callable(tool)
    
    result = await tool("BTC-USD", "hours")
    assert result["success"] is True
    assert result["ticker"] == "BTC-USD"


@pytest.mark.asyncio
async def test_get_action_recommendation_tool(mock_forecasting_client):
    """Test get action recommendation tool."""
    toolkit = MCPForecastingToolkit(forecasting_client=mock_forecasting_client)
    await toolkit.initialize()
    
    tool = toolkit.get_action_recommendation_tool()
    assert tool is not None
    
    result = await tool("BTC-USD", "hours")
    assert result["success"] is True
    assert result["action"] == 2
    assert result["action_name"] == "BUY"


@pytest.mark.asyncio
async def test_dex_trading_toolkit_initialization(mock_dex_client):
    """Test DEX trading toolkit initialization."""
    toolkit = DEXTradingToolkit(dex_client=mock_dex_client)
    await toolkit.initialize()
    
    assert toolkit.dex_client is not None
    mock_dex_client.connect.assert_called_once()


@pytest.mark.asyncio
async def test_buy_asset_tool(mock_dex_client):
    """Test buy asset tool."""
    toolkit = DEXTradingToolkit(dex_client=mock_dex_client)
    await toolkit.initialize()
    
    tool = toolkit.buy_asset_tool()
    assert tool is not None
    
    result = await tool("BTC", 100.0)
    assert result["success"] is True
    assert result["ticker"] == "BTC"


@pytest.mark.asyncio
async def test_get_portfolio_status_tool(mock_dex_client):
    """Test get portfolio status tool."""
    toolkit = DEXTradingToolkit(dex_client=mock_dex_client)
    await toolkit.initialize()
    
    tool = toolkit.get_portfolio_status_tool()
    assert tool is not None
    
    result = await tool()
    assert result["success"] is True
    assert "portfolio" in result


def test_get_all_tools():
    """Test that all tools can be retrieved."""
    toolkit = MCPForecastingToolkit()
    tools = toolkit.get_all_tools()
    
    assert len(tools) > 0
    assert all(callable(tool) for tool in tools)

