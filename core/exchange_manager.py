"""
Exchange manager for handling multiple exchange connections.
"""
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from core.exchange_interface import (
    ExchangeInterface, ExchangeType, OrderSide, OrderType,
    Balance, Ticker, Order, Trade, ExchangeError
)
from core.logging import log


class ExchangeManager:
    """Manages multiple exchange connections and provides unified interface."""
    
    def __init__(self):
        self.exchanges: Dict[ExchangeType, ExchangeInterface] = {}
        self.primary_exchange: Optional[ExchangeType] = None
        self.fallback_exchanges: List[ExchangeType] = []
        self.trading_enabled = True
        self.paper_trading = False
    
    async def initialize(self) -> None:
        """Initialize the exchange manager (connects to all exchanges)."""
        await self.connect_all()
    
    def add_exchange(self, exchange: ExchangeInterface, is_primary: bool = False) -> None:
        """Add an exchange to the manager."""
        self.exchanges[exchange.exchange_type] = exchange
        
        if is_primary:
            self.primary_exchange = exchange.exchange_type
        else:
            self.fallback_exchanges.append(exchange.exchange_type)
        
        log.info(f"Added {exchange.exchange_type.value} exchange (primary: {is_primary})")
    
    async def connect_all(self) -> None:
        """Connect to all registered exchanges."""
        tasks = []
        for exchange in self.exchanges.values():
            tasks.append(self._connect_exchange(exchange))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        connected_count = sum(1 for exchange in self.exchanges.values() if exchange.is_connected)
        log.info(f"Connected to {connected_count}/{len(self.exchanges)} exchanges")
    
    async def disconnect_all(self) -> None:
        """Disconnect from all exchanges."""
        tasks = []
        for exchange in self.exchanges.values():
            if exchange.is_connected:
                tasks.append(exchange.disconnect())
        
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("Disconnected from all exchanges")
    
    async def _connect_exchange(self, exchange: ExchangeInterface) -> None:
        """Connect to a single exchange with error handling."""
        try:
            await exchange.connect()
            log.info(f"Successfully connected to {exchange.exchange_type.value}")
        except Exception as e:
            log.error(f"Failed to connect to {exchange.exchange_type.value}: {e}")
    
    def get_exchange(self, exchange_type: Optional[ExchangeType] = None) -> ExchangeInterface:
        """Get exchange instance."""
        if exchange_type is None:
            exchange_type = self.primary_exchange
        
        if exchange_type is None:
            raise ExchangeError("No primary exchange configured")
        
        if exchange_type not in self.exchanges:
            raise ExchangeError(f"Exchange {exchange_type.value} not found")
        
        exchange = self.exchanges[exchange_type]
        if not exchange.is_connected:
            raise ExchangeError(f"Exchange {exchange_type.value} not connected")
        
        return exchange
    
    async def get_balance(self, asset: str, exchange_type: Optional[ExchangeType] = None) -> Balance:
        """Get balance from primary or specified exchange."""
        exchange = self.get_exchange(exchange_type)
        return await exchange.get_balance(asset)
    
    async def get_all_balances(self, exchange_type: Optional[ExchangeType] = None) -> Dict[str, Balance]:
        """Get all balances from primary or specified exchange."""
        exchange = self.get_exchange(exchange_type)
        return await exchange.get_all_balances()
    
    async def get_ticker(self, symbol: str, exchange_type: Optional[ExchangeType] = None) -> Ticker:
        """Get ticker from primary or specified exchange."""
        exchange = self.get_exchange(exchange_type)
        return await exchange.get_ticker(symbol)
    
    async def get_tickers(self, symbols: List[str], exchange_type: Optional[ExchangeType] = None) -> Dict[str, Ticker]:
        """Get tickers from primary or specified exchange."""
        exchange = self.get_exchange(exchange_type)
        return await exchange.get_tickers(symbols)
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: float,
        price: Optional[float] = None,
        exchange_type: Optional[ExchangeType] = None,
        **kwargs
    ) -> Order:
        """Place order on primary or specified exchange."""
        if not self.trading_enabled:
            raise ExchangeError("Trading is disabled")
        
        if self.paper_trading:
            log.info(f"PAPER TRADING: {side.value} {amount} {symbol} @ {price or 'market'}")
            # In paper trading mode, simulate the order
            return await self._simulate_order(symbol, side, order_type, amount, price, **kwargs)
        
        exchange = self.get_exchange(exchange_type)
        return await exchange.place_order(symbol, side, order_type, amount, price, **kwargs)
    
    async def cancel_order(self, order_id: str, exchange_type: Optional[ExchangeType] = None) -> bool:
        """Cancel order on primary or specified exchange."""
        if self.paper_trading:
            log.info(f"PAPER TRADING: Cancel order {order_id}")
            return True
        
        exchange = self.get_exchange(exchange_type)
        return await exchange.cancel_order(order_id)
    
    async def get_order(self, order_id: str, exchange_type: Optional[ExchangeType] = None) -> Order:
        """Get order from primary or specified exchange."""
        if self.paper_trading:
            # Return simulated order
            return await self._get_simulated_order(order_id)
        
        exchange = self.get_exchange(exchange_type)
        return await exchange.get_order(order_id)
    
    async def get_open_orders(self, symbol: Optional[str] = None, exchange_type: Optional[ExchangeType] = None) -> List[Order]:
        """Get open orders from primary or specified exchange."""
        if self.paper_trading:
            return []  # No open orders in paper trading
        
        exchange = self.get_exchange(exchange_type)
        return await exchange.get_open_orders(symbol)
    
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        limit: int = 100,
        exchange_type: Optional[ExchangeType] = None
    ) -> List[Trade]:
        """Get trades from primary or specified exchange."""
        if self.paper_trading:
            return []  # No trade history in paper trading
        
        exchange = self.get_exchange(exchange_type)
        return await exchange.get_trades(symbol, order_id, limit)
    
    async def _simulate_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Simulate order execution for paper trading."""
        # Get current price if not provided
        if price is None:
            try:
                ticker = await self.get_ticker(symbol)
                price = ticker.price
            except ExchangeError:
                price = 100.0  # Default price for simulation
        
        # Simulate order execution
        order_id = f"paper_{int(datetime.utcnow().timestamp() * 1000)}"
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            amount=amount,
            price=price,
            status="filled",  # Simulate immediate fill
            filled_amount=amount,
            remaining_amount=0.0,
            average_price=price,
            fee=amount * price * 0.001,  # 0.1% fee
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            client_order_id=kwargs.get("client_order_id")
        )
        
        log.info(f"Simulated order: {order_id} - {side.value} {amount} {symbol} @ {price}")
        return order
    
    async def _get_simulated_order(self, order_id: str) -> Order:
        """Get simulated order for paper trading."""
        # This would typically be stored in memory or database
        # For now, return a basic simulated order
        return Order(
            id=order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            amount=0.001,
            price=50000.0,
            status="filled",
            filled_amount=0.001,
            remaining_amount=0.0,
            average_price=50000.0,
            fee=0.05,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    def set_trading_enabled(self, enabled: bool) -> None:
        """Enable or disable trading."""
        self.trading_enabled = enabled
        log.info(f"Trading {'enabled' if enabled else 'disabled'}")
    
    def set_paper_trading(self, enabled: bool) -> None:
        """Enable or disable paper trading mode."""
        self.paper_trading = enabled
        log.info(f"Paper trading {'enabled' if enabled else 'disabled'}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "trading_enabled": self.trading_enabled,
            "paper_trading": self.paper_trading,
            "primary_exchange": self.primary_exchange.value if self.primary_exchange else None,
            "fallback_exchanges": [ex.value for ex in self.fallback_exchanges],
            "exchanges": {
                ex_type.value: {
                    "connected": exchange.is_connected,
                    "is_mock": exchange.is_mock,
                    "info": exchange.get_exchange_info()
                }
                for ex_type, exchange in self.exchanges.items()
            }
        }


# Global exchange manager instance
exchange_manager = ExchangeManager()