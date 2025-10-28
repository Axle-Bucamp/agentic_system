"""
Unit tests for exchange interface and implementations.
"""
import pytest
from datetime import datetime
from core.exchange_interface import (
    ExchangeType, OrderSide, OrderType, OrderStatus,
    Balance, Ticker, Order, ExchangeError, InsufficientBalanceError
)
from tests.mocks.mock_dex_exchange import MockDEXExchange
from tests.mocks.mock_mexc_exchange import MockMEXCExchange


class TestExchangeInterface:
    """Test exchange interface functionality."""
    
    @pytest.mark.asyncio
    async def test_dex_exchange_connection(self):
        """Test DEX exchange connection."""
        config = {"mock_mode": True}
        exchange = MockDEXExchange(config)
        
        assert not exchange.is_connected
        await exchange.connect()
        assert exchange.is_connected
        assert exchange.exchange_type == ExchangeType.DEX
        
        await exchange.disconnect()
        assert not exchange.is_connected
    
    @pytest.mark.asyncio
    async def test_mexc_exchange_connection(self):
        """Test MEXC exchange connection."""
        config = {"mock_mode": True}
        exchange = MockMEXCExchange(config)
        
        assert not exchange.is_connected
        await exchange.connect()
        assert exchange.is_connected
        assert exchange.exchange_type == ExchangeType.MEXC
        
        await exchange.disconnect()
        assert not exchange.is_connected
    
    @pytest.mark.asyncio
    async def test_dex_balance_operations(self):
        """Test DEX exchange balance operations."""
        config = {"mock_mode": True}
        exchange = MockDEXExchange(config)
        await exchange.connect()
        
        # Test get balance
        balance = await exchange.get_balance("USDC")
        assert balance.asset == "USDC"
        assert balance.free == 10000.0
        assert balance.total == 10000.0
        
        # Test get all balances
        all_balances = await exchange.get_all_balances()
        assert "USDC" in all_balances
        assert "WETH" in all_balances
        assert "USDT" in all_balances
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_mexc_balance_operations(self):
        """Test MEXC exchange balance operations."""
        config = {"mock_mode": True}
        exchange = MockMEXCExchange(config)
        await exchange.connect()
        
        # Test get balance
        balance = await exchange.get_balance("USDT")
        assert balance.asset == "USDT"
        assert balance.free == 10000.0
        assert balance.total == 10000.0
        
        # Test get all balances
        all_balances = await exchange.get_all_balances()
        assert "USDT" in all_balances
        assert "BTC" in all_balances
        assert "ETH" in all_balances
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_dex_ticker_operations(self):
        """Test DEX exchange ticker operations."""
        config = {"mock_mode": True}
        exchange = MockDEXExchange(config)
        await exchange.connect()
        
        # Test get single ticker
        ticker = await exchange.get_ticker("BTCUSDC")
        assert ticker.symbol == "BTCUSDC"
        assert ticker.price == 45000.0
        assert ticker.volume == 1000000.0
        
        # Test get multiple tickers
        tickers = await exchange.get_tickers(["BTCUSDC", "ETHUSDC"])
        assert "BTCUSDC" in tickers
        assert "ETHUSDC" in tickers
        assert tickers["BTCUSDC"].price == 45000.0
        assert tickers["ETHUSDC"].price == 2000.0
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_mexc_ticker_operations(self):
        """Test MEXC exchange ticker operations."""
        config = {"mock_mode": True}
        exchange = MockMEXCExchange(config)
        await exchange.connect()
        
        # Test get single ticker
        ticker = await exchange.get_ticker("BTCUSDT")
        assert ticker.symbol == "BTCUSDT"
        assert ticker.price == 45000.0
        assert ticker.volume == 1000000.0
        
        # Test get multiple tickers
        tickers = await exchange.get_tickers(["BTCUSDT", "ETHUSDT"])
        assert "BTCUSDT" in tickers
        assert "ETHUSDT" in tickers
        assert tickers["BTCUSDT"].price == 45000.0
        assert tickers["ETHUSDT"].price == 2000.0
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_dex_order_operations(self):
        """Test DEX exchange order operations."""
        config = {"mock_mode": True}
        exchange = MockDEXExchange(config)
        await exchange.connect()
        
        # Test place buy order
        order = await exchange.place_order(
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001
        )
        
        assert order.symbol == "BTCUSDC"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.MARKET
        assert order.amount == 0.001
        assert order.status == OrderStatus.FILLED
        assert order.filled_amount == 0.001
        
        # Test get order
        retrieved_order = await exchange.get_order(order.id)
        assert retrieved_order.id == order.id
        assert retrieved_order.symbol == order.symbol
        
        # Test cancel order (should fail for filled order)
        cancelled = await exchange.cancel_order(order.id)
        assert not cancelled
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_mexc_order_operations(self):
        """Test MEXC exchange order operations."""
        config = {"mock_mode": True}
        exchange = MockMEXCExchange(config)
        await exchange.connect()
        
        # Test place buy order
        order = await exchange.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001
        )
        
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.MARKET
        assert order.amount == 0.001
        assert order.status == OrderStatus.FILLED
        assert order.filled_amount == 0.001
        
        # Test get order
        retrieved_order = await exchange.get_order(order.id)
        assert retrieved_order.id == order.id
        assert retrieved_order.symbol == order.symbol
        
        # Test cancel order (should fail for filled order)
        cancelled = await exchange.cancel_order(order.id)
        assert not cancelled
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_dex_insufficient_balance(self):
        """Test DEX exchange insufficient balance error."""
        config = {"mock_mode": True}
        exchange = MockDEXExchange(config)
        await exchange.connect()
        
        # Set low balance
        exchange.set_balance("USDC", 10.0)
        
        # Try to buy with insufficient balance
        with pytest.raises(InsufficientBalanceError):
            await exchange.place_order(
                symbol="BTCUSDC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=1.0  # This would cost 45000 USDC
            )
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_mexc_insufficient_balance(self):
        """Test MEXC exchange insufficient balance error."""
        config = {"mock_mode": True}
        exchange = MockMEXCExchange(config)
        await exchange.connect()
        
        # Set low balance
        exchange.set_balance("USDT", 10.0)
        
        # Try to buy with insufficient balance
        with pytest.raises(InsufficientBalanceError):
            await exchange.place_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=1.0  # This would cost 45000 USDT
            )
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_dex_order_not_found(self):
        """Test DEX exchange order not found error."""
        config = {"mock_mode": True}
        exchange = MockDEXExchange(config)
        await exchange.connect()
        
        # Try to get non-existent order
        with pytest.raises(ExchangeError):
            await exchange.get_order("non_existent_order")
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_mexc_order_not_found(self):
        """Test MEXC exchange order not found error."""
        config = {"mock_mode": True}
        exchange = MockMEXCExchange(config)
        await exchange.connect()
        
        # Try to get non-existent order
        with pytest.raises(ExchangeError):
            await exchange.get_order("non_existent_order")
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_dex_trading_fees(self):
        """Test DEX exchange trading fees."""
        config = {"mock_mode": True}
        exchange = MockDEXExchange(config)
        await exchange.connect()
        
        fees = await exchange.get_trading_fees("BTCUSDC")
        assert "maker" in fees
        assert "taker" in fees
        assert fees["maker"] == 0.003
        assert fees["taker"] == 0.003
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_mexc_trading_fees(self):
        """Test MEXC exchange trading fees."""
        config = {"mock_mode": True}
        exchange = MockMEXCExchange(config)
        await exchange.connect()
        
        fees = await exchange.get_trading_fees("BTCUSDT")
        assert "maker" in fees
        assert "taker" in fees
        assert fees["maker"] == 0.001
        assert fees["taker"] == 0.001
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_dex_balance_update_after_trade(self):
        """Test DEX exchange balance update after trade."""
        config = {"mock_mode": True}
        exchange = MockDEXExchange(config)
        await exchange.connect()
        
        # Get initial balance
        initial_balance = await exchange.get_balance("USDC")
        initial_btc_balance = await exchange.get_balance("BTC")
        
        # Place buy order
        order = await exchange.place_order(
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.1
        )
        
        # Check updated balances
        updated_balance = await exchange.get_balance("USDC")
        updated_btc_balance = await exchange.get_balance("BTC")
        
        # USDC should decrease, BTC should increase
        assert updated_balance.free < initial_balance.free
        assert updated_btc_balance.free > initial_btc_balance.free
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_mexc_balance_update_after_trade(self):
        """Test MEXC exchange balance update after trade."""
        config = {"mock_mode": True}
        exchange = MockMEXCExchange(config)
        await exchange.connect()
        
        # Get initial balance
        initial_balance = await exchange.get_balance("USDT")
        initial_btc_balance = await exchange.get_balance("BTC")
        
        # Place buy order
        order = await exchange.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.1
        )
        
        # Check updated balances
        updated_balance = await exchange.get_balance("USDT")
        updated_btc_balance = await exchange.get_balance("BTC")
        
        # USDT should decrease, BTC should increase
        assert updated_balance.free < initial_balance.free
        assert updated_btc_balance.free > initial_btc_balance.free
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_dex_open_orders(self):
        """Test DEX exchange open orders."""
        config = {"mock_mode": True}
        exchange = MockDEXExchange(config)
        await exchange.connect()
        
        # Initially no open orders
        open_orders = await exchange.get_open_orders()
        assert len(open_orders) == 0
        
        # Place a filled order (should not appear in open orders)
        await exchange.place_order(
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001
        )
        
        # Still no open orders (order was filled)
        open_orders = await exchange.get_open_orders()
        assert len(open_orders) == 0
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_mexc_open_orders(self):
        """Test MEXC exchange open orders."""
        config = {"mock_mode": True}
        exchange = MockMEXCExchange(config)
        await exchange.connect()
        
        # Initially no open orders
        open_orders = await exchange.get_open_orders()
        assert len(open_orders) == 0
        
        # Place a filled order (should not appear in open orders)
        await exchange.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.001
        )
        
        # Still no open orders (order was filled)
        open_orders = await exchange.get_open_orders()
        assert len(open_orders) == 0
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_dex_order_book(self):
        """Test DEX exchange order book."""
        config = {"mock_mode": True}
        exchange = MockDEXExchange(config)
        await exchange.connect()
        
        order_book = await exchange.get_order_book("BTCUSDC")
        assert "bids" in order_book
        assert "asks" in order_book
        assert "timestamp" in order_book
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_mexc_order_book(self):
        """Test MEXC exchange order book."""
        config = {"mock_mode": True}
        exchange = MockMEXCExchange(config)
        await exchange.connect()
        
        order_book = await exchange.get_order_book("BTCUSDT")
        assert "bids" in order_book
        assert "asks" in order_book
        assert "timestamp" in order_book
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_dex_klines(self):
        """Test DEX exchange klines."""
        config = {"mock_mode": True}
        exchange = MockDEXExchange(config)
        await exchange.connect()
        
        klines = await exchange.get_klines("BTCUSDC", "1h")
        assert isinstance(klines, list)
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_mexc_klines(self):
        """Test MEXC exchange klines."""
        config = {"mock_mode": True}
        exchange = MockMEXCExchange(config)
        await exchange.connect()
        
        klines = await exchange.get_klines("BTCUSDT", "1h")
        assert isinstance(klines, list)
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_dex_validate_symbol(self):
        """Test DEX exchange symbol validation."""
        config = {"mock_mode": True}
        exchange = MockDEXExchange(config)
        await exchange.connect()
        
        # Valid symbol
        is_valid = await exchange.validate_symbol("BTCUSDC")
        assert is_valid
        
        # Invalid symbol (would fail in real implementation)
        is_valid = await exchange.validate_symbol("INVALID")
        assert is_valid  # Mock always returns True
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_mexc_validate_symbol(self):
        """Test MEXC exchange symbol validation."""
        config = {"mock_mode": True}
        exchange = MockMEXCExchange(config)
        await exchange.connect()
        
        # Valid symbol
        is_valid = await exchange.validate_symbol("BTCUSDT")
        assert is_valid
        
        # Invalid symbol (would fail in real implementation)
        is_valid = await exchange.validate_symbol("INVALID")
        assert is_valid  # Mock always returns True
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_dex_exchange_info(self):
        """Test DEX exchange info."""
        config = {"mock_mode": True}
        exchange = MockDEXExchange(config)
        await exchange.connect()
        
        info = exchange.get_exchange_info()
        assert info["exchange_type"] == "DEX"
        assert info["is_connected"] is True
        assert info["is_mock"] is True
        
        await exchange.disconnect()
    
    @pytest.mark.asyncio
    async def test_mexc_exchange_info(self):
        """Test MEXC exchange info."""
        config = {"mock_mode": True}
        exchange = MockMEXCExchange(config)
        await exchange.connect()
        
        info = exchange.get_exchange_info()
        assert info["exchange_type"] == "MEXC"
        assert info["is_connected"] is True
        assert info["is_mock"] is True
        
        await exchange.disconnect()
