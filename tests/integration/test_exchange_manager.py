"""
Integration tests for exchange manager.
"""
import pytest
from core.exchange_manager import ExchangeManager
from core.exchange_interface import ExchangeType, OrderSide, OrderType
from tests.mocks.mock_dex_exchange import MockDEXExchange
from tests.mocks.mock_mexc_exchange import MockMEXCExchange


class TestExchangeManager:
    """Test exchange manager functionality."""
    
    @pytest.mark.asyncio
    async def test_exchange_manager_initialization(self):
        """Test exchange manager initialization."""
        manager = ExchangeManager()
        
        assert manager.primary_exchange is None
        assert len(manager.fallback_exchanges) == 0
        assert manager.trading_enabled is True
        assert manager.paper_trading is False
    
    @pytest.mark.asyncio
    async def test_add_exchange(self):
        """Test adding exchanges to manager."""
        manager = ExchangeManager()
        
        # Add DEX exchange as primary
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        
        assert manager.primary_exchange == ExchangeType.DEX
        assert ExchangeType.DEX in manager.exchanges
        
        # Add MEXC exchange as fallback
        mexc_config = {"mock_mode": True}
        mexc_exchange = MockMEXCExchange(mexc_config)
        manager.add_exchange(mexc_exchange, is_primary=False)
        
        assert ExchangeType.MEXC in manager.fallback_exchanges
        assert ExchangeType.MEXC in manager.exchanges
    
    @pytest.mark.asyncio
    async def test_connect_all_exchanges(self):
        """Test connecting to all exchanges."""
        manager = ExchangeManager()
        
        # Add exchanges
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        
        mexc_config = {"mock_mode": True}
        mexc_exchange = MockMEXCExchange(mexc_config)
        manager.add_exchange(mexc_exchange, is_primary=False)
        
        # Connect to all
        await manager.connect_all()
        
        # Check all exchanges are connected
        for exchange in manager.exchanges.values():
            assert exchange.is_connected
    
    @pytest.mark.asyncio
    async def test_disconnect_all_exchanges(self):
        """Test disconnecting from all exchanges."""
        manager = ExchangeManager()
        
        # Add and connect exchanges
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        
        mexc_config = {"mock_mode": True}
        mexc_exchange = MockMEXCExchange(mexc_config)
        manager.add_exchange(mexc_exchange, is_primary=False)
        
        await manager.connect_all()
        
        # Disconnect all
        await manager.disconnect_all()
        
        # Check all exchanges are disconnected
        for exchange in manager.exchanges.values():
            assert not exchange.is_connected
    
    @pytest.mark.asyncio
    async def test_get_exchange_primary(self):
        """Test getting primary exchange."""
        manager = ExchangeManager()
        
        # Add DEX as primary
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        await manager.connect_all()
        
        # Get primary exchange
        exchange = manager.get_exchange()
        assert exchange.exchange_type == ExchangeType.DEX
    
    @pytest.mark.asyncio
    async def test_get_exchange_specific(self):
        """Test getting specific exchange."""
        manager = ExchangeManager()
        
        # Add both exchanges
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        
        mexc_config = {"mock_mode": True}
        mexc_exchange = MockMEXCExchange(mexc_config)
        manager.add_exchange(mexc_exchange, is_primary=False)
        
        await manager.connect_all()
        
        # Get specific exchange
        dex = manager.get_exchange(ExchangeType.DEX)
        assert dex.exchange_type == ExchangeType.DEX
        
        mexc = manager.get_exchange(ExchangeType.MEXC)
        assert mexc.exchange_type == ExchangeType.MEXC
    
    @pytest.mark.asyncio
    async def test_get_exchange_not_found(self):
        """Test getting non-existent exchange."""
        manager = ExchangeManager()
        
        with pytest.raises(Exception):  # ExchangeError
            manager.get_exchange(ExchangeType.DEX)
    
    @pytest.mark.asyncio
    async def test_get_exchange_not_connected(self):
        """Test getting exchange that's not connected."""
        manager = ExchangeManager()
        
        # Add exchange but don't connect
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        
        with pytest.raises(Exception):  # ExchangeError
            manager.get_exchange()
    
    @pytest.mark.asyncio
    async def test_get_balance_primary(self):
        """Test getting balance from primary exchange."""
        manager = ExchangeManager()
        
        # Add DEX as primary
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        await manager.connect_all()
        
        # Get balance
        balance = await manager.get_balance("USDC")
        assert balance.asset == "USDC"
        assert balance.free == 10000.0
    
    @pytest.mark.asyncio
    async def test_get_balance_specific_exchange(self):
        """Test getting balance from specific exchange."""
        manager = ExchangeManager()
        
        # Add both exchanges
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        
        mexc_config = {"mock_mode": True}
        mexc_exchange = MockMEXCExchange(mexc_config)
        manager.add_exchange(mexc_exchange, is_primary=False)
        
        await manager.connect_all()
        
        # Get balance from MEXC
        balance = await manager.get_balance("USDT", ExchangeType.MEXC)
        assert balance.asset == "USDT"
        assert balance.free == 10000.0
    
    @pytest.mark.asyncio
    async def test_get_all_balances(self):
        """Test getting all balances."""
        manager = ExchangeManager()
        
        # Add DEX as primary
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        await manager.connect_all()
        
        # Get all balances
        balances = await manager.get_all_balances()
        assert "USDC" in balances
        assert "WETH" in balances
        assert "USDT" in balances
    
    @pytest.mark.asyncio
    async def test_get_ticker_primary(self):
        """Test getting ticker from primary exchange."""
        manager = ExchangeManager()
        
        # Add DEX as primary
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        await manager.connect_all()
        
        # Get ticker
        ticker = await manager.get_ticker("BTCUSDC")
        assert ticker.symbol == "BTCUSDC"
        assert ticker.price == 45000.0
    
    @pytest.mark.asyncio
    async def test_get_ticker_specific_exchange(self):
        """Test getting ticker from specific exchange."""
        manager = ExchangeManager()
        
        # Add both exchanges
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        
        mexc_config = {"mock_mode": True}
        mexc_exchange = MockMEXCExchange(mexc_config)
        manager.add_exchange(mexc_exchange, is_primary=False)
        
        await manager.connect_all()
        
        # Get ticker from MEXC
        ticker = await manager.get_ticker("BTCUSDT", ExchangeType.MEXC)
        assert ticker.symbol == "BTCUSDT"
        assert ticker.price == 45000.0
    
    @pytest.mark.asyncio
    async def test_get_tickers(self):
        """Test getting multiple tickers."""
        manager = ExchangeManager()
        
        # Add DEX as primary
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        await manager.connect_all()
        
        # Get tickers
        tickers = await manager.get_tickers(["BTCUSDC", "ETHUSDC"])
        assert "BTCUSDC" in tickers
        assert "ETHUSDC" in tickers
        assert tickers["BTCUSDC"].price == 45000.0
        assert tickers["ETHUSDC"].price == 2000.0
    
    @pytest.mark.asyncio
    async def test_place_order_paper_trading(self):
        """Test placing order in paper trading mode."""
        manager = ExchangeManager()
        
        # Add DEX as primary
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        await manager.connect_all()
        
        # Enable paper trading
        manager.set_paper_trading(True)
        
        # Place order
        order = await manager.place_order(
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001
        )
        
        assert order.symbol == "BTCUSDC"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.MARKET
        assert order.amount == 0.001
    
    @pytest.mark.asyncio
    async def test_place_order_trading_disabled(self):
        """Test placing order when trading is disabled."""
        manager = ExchangeManager()
        
        # Add DEX as primary
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        await manager.connect_all()
        
        # Disable trading
        manager.set_trading_enabled(False)
        
        # Try to place order
        with pytest.raises(Exception):  # ExchangeError
            await manager.place_order(
                symbol="BTCUSDC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=0.001
            )
    
    @pytest.mark.asyncio
    async def test_cancel_order_paper_trading(self):
        """Test canceling order in paper trading mode."""
        manager = ExchangeManager()
        
        # Add DEX as primary
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        await manager.connect_all()
        
        # Enable paper trading
        manager.set_paper_trading(True)
        
        # Cancel order (should succeed in paper trading)
        cancelled = await manager.cancel_order("test_order_id")
        assert cancelled is True
    
    @pytest.mark.asyncio
    async def test_get_order_paper_trading(self):
        """Test getting order in paper trading mode."""
        manager = ExchangeManager()
        
        # Add DEX as primary
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        await manager.connect_all()
        
        # Enable paper trading
        manager.set_paper_trading(True)
        
        # Get order
        order = await manager.get_order("test_order_id")
        assert order.symbol == "BTCUSDT"  # Mock order symbol
        assert order.side == OrderSide.BUY  # Mock order side
    
    @pytest.mark.asyncio
    async def test_get_open_orders_paper_trading(self):
        """Test getting open orders in paper trading mode."""
        manager = ExchangeManager()
        
        # Add DEX as primary
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        await manager.connect_all()
        
        # Enable paper trading
        manager.set_paper_trading(True)
        
        # Get open orders (should be empty in paper trading)
        orders = await manager.get_open_orders()
        assert len(orders) == 0
    
    @pytest.mark.asyncio
    async def test_get_trades_paper_trading(self):
        """Test getting trades in paper trading mode."""
        manager = ExchangeManager()
        
        # Add DEX as primary
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        await manager.connect_all()
        
        # Enable paper trading
        manager.set_paper_trading(True)
        
        # Get trades (should be empty in paper trading)
        trades = await manager.get_trades()
        assert len(trades) == 0
    
    @pytest.mark.asyncio
    async def test_set_trading_enabled(self):
        """Test setting trading enabled/disabled."""
        manager = ExchangeManager()
        
        # Initially enabled
        assert manager.trading_enabled is True
        
        # Disable trading
        manager.set_trading_enabled(False)
        assert manager.trading_enabled is False
        
        # Enable trading
        manager.set_trading_enabled(True)
        assert manager.trading_enabled is True
    
    @pytest.mark.asyncio
    async def test_set_paper_trading(self):
        """Test setting paper trading mode."""
        manager = ExchangeManager()
        
        # Initially disabled
        assert manager.paper_trading is False
        
        # Enable paper trading
        manager.set_paper_trading(True)
        assert manager.paper_trading is True
        
        # Disable paper trading
        manager.set_paper_trading(False)
        assert manager.paper_trading is False
    
    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test getting manager status."""
        manager = ExchangeManager()
        
        # Add exchanges
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        
        mexc_config = {"mock_mode": True}
        mexc_exchange = MockMEXCExchange(mexc_config)
        manager.add_exchange(mexc_exchange, is_primary=False)
        
        await manager.connect_all()
        
        # Get status
        status = manager.get_status()
        
        assert "trading_enabled" in status
        assert "paper_trading" in status
        assert "primary_exchange" in status
        assert "fallback_exchanges" in status
        assert "exchanges" in status
        
        assert status["trading_enabled"] is True
        assert status["paper_trading"] is False
        assert status["primary_exchange"] == "DEX"
        assert "MEXC" in status["fallback_exchanges"]
        assert "DEX" in status["exchanges"]
        assert "MEXC" in status["exchanges"]
        
        # Check exchange details
        dex_status = status["exchanges"]["DEX"]
        assert "connected" in dex_status
        assert "is_mock" in dex_status
        assert "info" in dex_status
        assert dex_status["connected"] is True
        assert dex_status["is_mock"] is True
    
    @pytest.mark.asyncio
    async def test_multiple_exchanges_balance_comparison(self):
        """Test comparing balances across multiple exchanges."""
        manager = ExchangeManager()
        
        # Add both exchanges
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        
        mexc_config = {"mock_mode": True}
        mexc_exchange = MockMEXCExchange(mexc_config)
        manager.add_exchange(mexc_exchange, is_primary=False)
        
        await manager.connect_all()
        
        # Get balances from both exchanges
        dex_balances = await manager.get_all_balances(ExchangeType.DEX)
        mexc_balances = await manager.get_all_balances(ExchangeType.MEXC)
        
        # Check that both exchanges have different base currencies
        assert "USDC" in dex_balances
        assert "USDT" in mexc_balances
        
        # Check that both have BTC
        assert "BTC" in dex_balances
        assert "BTC" in mexc_balances
    
    @pytest.mark.asyncio
    async def test_multiple_exchanges_ticker_comparison(self):
        """Test comparing tickers across multiple exchanges."""
        manager = ExchangeManager()
        
        # Add both exchanges
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        
        mexc_config = {"mock_mode": True}
        mexc_exchange = MockMEXCExchange(mexc_config)
        manager.add_exchange(mexc_exchange, is_primary=False)
        
        await manager.connect_all()
        
        # Get tickers from both exchanges
        dex_ticker = await manager.get_ticker("BTCUSDC", ExchangeType.DEX)
        mexc_ticker = await manager.get_ticker("BTCUSDT", ExchangeType.MEXC)
        
        # Both should have same price (in mock mode)
        assert dex_ticker.price == mexc_ticker.price
        assert dex_ticker.symbol == "BTCUSDC"
        assert mexc_ticker.symbol == "BTCUSDT"
    
    @pytest.mark.asyncio
    async def test_exchange_fallback_mechanism(self):
        """Test exchange fallback mechanism."""
        manager = ExchangeManager()
        
        # Add DEX as primary
        dex_config = {"mock_mode": True}
        dex_exchange = MockDEXExchange(dex_config)
        manager.add_exchange(dex_exchange, is_primary=True)
        
        # Add MEXC as fallback
        mexc_config = {"mock_mode": True}
        mexc_exchange = MockMEXCExchange(mexc_config)
        manager.add_exchange(mexc_exchange, is_primary=False)
        
        await manager.connect_all()
        
        # Test that we can get data from both exchanges
        # Primary exchange (DEX)
        dex_balance = await manager.get_balance("USDC")
        assert dex_balance.asset == "USDC"
        
        # Fallback exchange (MEXC)
        mexc_balance = await manager.get_balance("USDT", ExchangeType.MEXC)
        assert mexc_balance.asset == "USDT"
    
    @pytest.mark.asyncio
    async def test_exchange_manager_error_handling(self):
        """Test exchange manager error handling."""
        manager = ExchangeManager()
        
        # Try to get exchange when none are added
        with pytest.raises(Exception):  # ExchangeError
            manager.get_exchange()
        
        # Try to get balance when no exchanges
        with pytest.raises(Exception):  # ExchangeError
            await manager.get_balance("USDC")
        
        # Try to place order when no exchanges
        with pytest.raises(Exception):  # ExchangeError
            await manager.place_order(
                symbol="BTCUSDC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=0.001
            )
