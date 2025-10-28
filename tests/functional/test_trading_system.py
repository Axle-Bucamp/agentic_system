"""
Functional tests for the complete trading system.
"""
import pytest
from datetime import datetime
from core.exchange_manager import ExchangeManager
from core.forecasting_client import ForecastingClient
from core.exchange_interface import ExchangeType, OrderSide, OrderType
from tests.mocks.mock_dex_exchange import MockDEXExchange
from tests.mocks.mock_mexc_exchange import MockMEXCExchange
from tests.mocks.mock_forecasting_api import MockForecastingClient


class TestTradingSystem:
    """Test complete trading system functionality."""
    
    @pytest.mark.asyncio
    async def test_system_initialization(self):
        """Test complete system initialization."""
        # Initialize exchange manager
        exchange_manager = ExchangeManager()
        
        # Add exchanges
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        exchange_manager.add_exchange(dex_exchange, is_primary=True)
        
        mexc_config = {"mock_mode": True}
        mexc_exchange = MockMEXCExchange(mexc_config)
        exchange_manager.add_exchange(mexc_exchange, is_primary=False)
        
        # Initialize forecasting client
        forecasting_config = {"mock_mode": True}
        forecasting_client = MockForecastingClient(forecasting_config)
        
        # Connect all components
        await exchange_manager.connect_all()
        await forecasting_client.connect()
        
        # Verify connections
        assert exchange_manager.exchanges[ExchangeType.DEX].is_connected
        assert exchange_manager.exchanges[ExchangeType.MEXC].is_connected
        assert forecasting_client.is_connected
        
        # Cleanup
        await exchange_manager.disconnect_all()
        await forecasting_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_market_data_flow(self):
        """Test market data flow through the system."""
        # Initialize system
        exchange_manager = ExchangeManager()
        forecasting_client = MockForecastingClient({"mock_mode": True})
        
        # Add exchanges
        dex_exchange = MockDEXExchange({"mock_mode": True})
        mexc_exchange = MockMEXCExchange({"mock_mode": True})
        exchange_manager.add_exchange(dex_exchange, is_primary=True)
        exchange_manager.add_exchange(mexc_exchange, is_primary=False)
        
        await exchange_manager.connect_all()
        await forecasting_client.connect()
        
        # Get market data from exchanges
        dex_ticker = await exchange_manager.get_ticker("BTCUSDC", ExchangeType.DEX)
        mexc_ticker = await exchange_manager.get_ticker("BTCUSDT", ExchangeType.MEXC)
        
        # Get forecasting data
        action = await forecasting_client.get_action_recommendation("BTC-USD", "hours")
        metrics = await forecasting_client.get_model_metrics("BTC-USD", "hours")
        
        # Verify data consistency
        assert dex_ticker.price == 45000.0
        assert mexc_ticker.price == 45000.0
        assert "action" in action
        assert "accuracy" in metrics
        
        # Cleanup
        await exchange_manager.disconnect_all()
        await forecasting_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_trading_decision_flow(self):
        """Test complete trading decision flow."""
        # Initialize system
        exchange_manager = ExchangeManager()
        forecasting_client = MockForecastingClient({"mock_mode": True})
        
        # Add exchanges
        dex_exchange = MockDEXExchange({"mock_mode": True})
        mexc_exchange = MockMEXCExchange({"mock_mode": True})
        exchange_manager.add_exchange(dex_exchange, is_primary=True)
        exchange_manager.add_exchange(mexc_exchange, is_primary=False)
        
        await exchange_manager.connect_all()
        await forecasting_client.connect()
        
        # Enable paper trading
        exchange_manager.set_paper_trading(True)
        
        # Get market data
        ticker = await exchange_manager.get_ticker("BTCUSDC")
        
        # Get forecasting recommendation
        action_data = await forecasting_client.get_action_recommendation("BTC-USD", "hours")
        
        # Make trading decision based on forecast
        if action_data["action"] == 2:  # BUY
            order = await exchange_manager.place_order(
                symbol="BTCUSDC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=0.001
            )
            
            assert order.side == OrderSide.BUY
            assert order.amount == 0.001
            assert order.status.value == "filled"
        
        # Cleanup
        await exchange_manager.disconnect_all()
        await forecasting_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_portfolio_management(self):
        """Test portfolio management functionality."""
        # Initialize system
        exchange_manager = ExchangeManager()
        
        # Add exchanges
        dex_exchange = MockDEXExchange({"mock_mode": True})
        mexc_exchange = MockMEXCExchange({"mock_mode": True})
        exchange_manager.add_exchange(dex_exchange, is_primary=True)
        exchange_manager.add_exchange(mexc_exchange, is_primary=False)
        
        await exchange_manager.connect_all()
        exchange_manager.set_paper_trading(True)
        
        # Get initial portfolio
        initial_balances = await exchange_manager.get_all_balances()
        initial_usdc = initial_balances["USDC"].free
        initial_btc = initial_balances.get("BTC", None)
        initial_btc_free = initial_btc.free if initial_btc else 0.0
        
        # Execute buy order
        order = await exchange_manager.place_order(
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.1
        )
        
        # Get updated portfolio
        updated_balances = await exchange_manager.get_all_balances()
        updated_usdc = updated_balances["USDC"].free
        updated_btc = updated_balances.get("BTC", None)
        updated_btc_free = updated_btc.free if updated_btc else 0.0
        
        # Verify portfolio changes
        assert updated_usdc < initial_usdc  # USDC decreased
        assert updated_btc_free > initial_btc_free  # BTC increased
        
        # Cleanup
        await exchange_manager.disconnect_all()
    
    @pytest.mark.asyncio
    async def test_risk_management(self):
        """Test risk management functionality."""
        # Initialize system
        exchange_manager = ExchangeManager()
        forecasting_client = MockForecastingClient({"mock_mode": True})
        
        # Add exchanges
        dex_exchange = MockDEXExchange({"mock_mode": True})
        exchange_manager.add_exchange(dex_exchange, is_primary=True)
        
        await exchange_manager.connect_all()
        await forecasting_client.connect()
        exchange_manager.set_paper_trading(True)
        
        # Test insufficient balance scenario
        dex_exchange.set_balance("USDC", 10.0)  # Set low balance
        
        # Try to place large order
        with pytest.raises(Exception):  # InsufficientBalanceError
            await exchange_manager.place_order(
                symbol="BTCUSDC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=1.0  # This would cost 45000 USDC
            )
        
        # Test trading disabled scenario
        exchange_manager.set_trading_enabled(False)
        
        with pytest.raises(Exception):  # ExchangeError
            await exchange_manager.place_order(
                symbol="BTCUSDC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=0.001
            )
        
        # Cleanup
        await exchange_manager.disconnect_all()
        await forecasting_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_multi_exchange_arbitrage(self):
        """Test multi-exchange arbitrage detection."""
        # Initialize system
        exchange_manager = ExchangeManager()
        
        # Add exchanges with different prices
        dex_exchange = MockDEXExchange({"mock_mode": True})
        mexc_exchange = MockMEXCExchange({"mock_mode": True})
        
        # Set different prices for arbitrage opportunity
        dex_exchange.set_price("BTCUSDC", 45000.0)
        mexc_exchange.set_price("BTCUSDT", 46000.0)  # Higher price
        
        exchange_manager.add_exchange(dex_exchange, is_primary=True)
        exchange_manager.add_exchange(mexc_exchange, is_primary=False)
        
        await exchange_manager.connect_all()
        
        # Get prices from both exchanges
        dex_ticker = await exchange_manager.get_ticker("BTCUSDC", ExchangeType.DEX)
        mexc_ticker = await exchange_manager.get_ticker("BTCUSDT", ExchangeType.MEXC)
        
        # Calculate price difference
        price_diff = mexc_ticker.price - dex_ticker.price
        price_diff_pct = (price_diff / dex_ticker.price) * 100
        
        # Verify arbitrage opportunity exists
        assert price_diff > 0
        assert price_diff_pct > 0
        
        # Cleanup
        await exchange_manager.disconnect_all()
    
    @pytest.mark.asyncio
    async def test_forecasting_integration(self):
        """Test forecasting API integration."""
        # Initialize forecasting client
        forecasting_client = MockForecastingClient({"mock_mode": True})
        await forecasting_client.connect()
        
        # Test different tickers and intervals
        tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]
        intervals = ["minutes", "hours", "days", "thirty"]
        
        for ticker in tickers:
            for interval in intervals:
                # Get action recommendation
                action = await forecasting_client.get_action_recommendation(ticker, interval)
                assert "action" in action
                assert "action_confidence" in action
                assert "forecast_price" in action
                
                # Get model metrics
                metrics = await forecasting_client.get_model_metrics(ticker, interval)
                assert "accuracy" in metrics
                assert "sharpe_ratio" in metrics
                assert "max_drawdown" in metrics
        
        # Test market sentiment
        sentiment = await forecasting_client.get_market_sentiment()
        assert "overall_sentiment" in sentiment
        assert "sentiment_score" in sentiment
        
        # Cleanup
        await forecasting_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Initialize system
        exchange_manager = ExchangeManager()
        forecasting_client = MockForecastingClient({"mock_mode": True})
        
        # Add exchanges
        dex_exchange = MockDEXExchange({"mock_mode": True})
        exchange_manager.add_exchange(dex_exchange, is_primary=True)
        
        await exchange_manager.connect_all()
        await forecasting_client.connect()
        
        # Test connection recovery
        await exchange_manager.disconnect_all()
        assert not dex_exchange.is_connected
        
        await exchange_manager.connect_all()
        assert dex_exchange.is_connected
        
        # Test forecasting client recovery
        await forecasting_client.disconnect()
        assert not forecasting_client.is_connected
        
        await forecasting_client.connect()
        assert forecasting_client.is_connected
        
        # Cleanup
        await exchange_manager.disconnect_all()
        await forecasting_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Initialize system
        exchange_manager = ExchangeManager()
        forecasting_client = MockForecastingClient({"mock_mode": True})
        
        # Add exchanges
        dex_exchange = MockDEXExchange({"mock_mode": True})
        exchange_manager.add_exchange(dex_exchange, is_primary=True)
        
        await exchange_manager.connect_all()
        await forecasting_client.connect()
        exchange_manager.set_paper_trading(True)
        
        # Execute multiple trades
        trades = []
        for i in range(5):
            order = await exchange_manager.place_order(
                symbol="BTCUSDC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=0.001
            )
            trades.append(order)
        
        # Verify trade execution
        assert len(trades) == 5
        for trade in trades:
            assert trade.status.value == "filled"
            assert trade.side == OrderSide.BUY
        
        # Test forecasting performance
        for ticker in ["BTC-USD", "ETH-USD"]:
            metrics = await forecasting_client.get_model_metrics(ticker, "hours")
            assert metrics["accuracy"] > 0.7
            assert metrics["sharpe_ratio"] > 1.0
        
        # Cleanup
        await exchange_manager.disconnect_all()
        await forecasting_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_system_status_monitoring(self):
        """Test system status monitoring."""
        # Initialize system
        exchange_manager = ExchangeManager()
        forecasting_client = MockForecastingClient({"mock_mode": True})
        
        # Add exchanges
        dex_exchange = MockDEXExchange({"mock_mode": True})
        mexc_exchange = MockMEXCExchange({"mock_mode": True})
        exchange_manager.add_exchange(dex_exchange, is_primary=True)
        exchange_manager.add_exchange(mexc_exchange, is_primary=False)
        
        await exchange_manager.connect_all()
        await forecasting_client.connect()
        
        # Get system status
        exchange_status = exchange_manager.get_status()
        forecasting_health = await forecasting_client.get_health_status()
        
        # Verify exchange status
        assert exchange_status["trading_enabled"] is True
        assert exchange_status["paper_trading"] is False
        assert exchange_status["primary_exchange"] == "DEX"
        assert "MEXC" in exchange_status["fallback_exchanges"]
        
        # Verify forecasting health
        assert forecasting_health["status"] == "healthy"
        assert "version" in forecasting_health
        
        # Test individual exchange status
        dex_status = exchange_status["exchanges"]["DEX"]
        assert dex_status["connected"] is True
        assert dex_status["is_mock"] is True
        
        # Cleanup
        await exchange_manager.disconnect_all()
        await forecasting_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_configuration_management(self):
        """Test configuration management."""
        # Test different configurations
        dex_config = {
            "mock_mode": True,
            "network": "ethereum",
            "slippage_tolerance": 0.01,
            "gas_limit": 500000
        }
        
        mexc_config = {
            "mock_mode": True,
            "api_key": "test_key",
            "rate_limit_delay": 0.2
        }
        
        forecasting_config = {
            "mock_mode": True,
            "base_url": "https://test-api.com",
            "timeout": 60.0,
            "retry_attempts": 5
        }
        
        # Initialize with custom configs
        dex_exchange = MockDEXExchange(dex_config)
        mexc_exchange = MockMEXCExchange(mexc_config)
        forecasting_client = MockForecastingClient(forecasting_config)
        
        # Verify configurations
        assert dex_exchange.config["slippage_tolerance"] == 0.01
        assert mexc_exchange.config["rate_limit_delay"] == 0.2
        assert forecasting_client.config["timeout"] == 60.0
        
        # Test connections
        await dex_exchange.connect()
        await mexc_exchange.connect()
        await forecasting_client.connect()
        
        assert dex_exchange.is_connected
        assert mexc_exchange.is_connected
        assert forecasting_client.is_connected
        
        # Cleanup
        await dex_exchange.disconnect()
        await mexc_exchange.disconnect()
        await forecasting_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations."""
        import asyncio
        
        # Initialize system
        exchange_manager = ExchangeManager()
        forecasting_client = MockForecastingClient({"mock_mode": True})
        
        # Add exchanges
        dex_exchange = MockDEXExchange({"mock_mode": True})
        exchange_manager.add_exchange(dex_exchange, is_primary=True)
        
        await exchange_manager.connect_all()
        await forecasting_client.connect()
        exchange_manager.set_paper_trading(True)
        
        # Define concurrent operations
        async def get_ticker():
            return await exchange_manager.get_ticker("BTCUSDC")
        
        async def get_forecast():
            return await forecasting_client.get_action_recommendation("BTC-USD", "hours")
        
        async def place_order():
            return await exchange_manager.place_order(
                symbol="BTCUSDC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=0.001
            )
        
        # Execute operations concurrently
        results = await asyncio.gather(
            get_ticker(),
            get_forecast(),
            place_order(),
            get_ticker(),
            get_forecast()
        )
        
        # Verify all operations completed successfully
        assert len(results) == 5
        assert results[0].symbol == "BTCUSDC"  # First ticker
        assert "action" in results[1]  # First forecast
        assert results[2].side == OrderSide.BUY  # Order
        assert results[3].symbol == "BTCUSDC"  # Second ticker
        assert "action" in results[4]  # Second forecast
        
        # Cleanup
        await exchange_manager.disconnect_all()
        await forecasting_client.disconnect()
