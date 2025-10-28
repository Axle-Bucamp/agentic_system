"""
Mock MEXC exchange implementation for testing.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from core.exchange_interface import (
    ExchangeInterface, ExchangeType, OrderSide, OrderType, OrderStatus,
    Balance, Ticker, Order, Trade, ExchangeError, InsufficientBalanceError
)


class MockMEXCExchange(ExchangeInterface):
    """Mock MEXC exchange for testing purposes."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ExchangeType.MEXC, config)
        self.is_connected = False
        self.is_mock = True
        
        # Mock data
        self.balances: Dict[str, float] = {
            "USDT": 10000.0,
            "BTC": 0.5,
            "ETH": 5.0,
            "USDC": 5000.0,
        }
        self.prices: Dict[str, float] = {
            "BTCUSDT": 45000.0,
            "ETHUSDT": 2000.0,
            "USDCUSDT": 1.0,
        }
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.order_counter = 0
    
    async def connect(self) -> None:
        """Mock connection."""
        self.is_connected = True
    
    async def disconnect(self) -> None:
        """Mock disconnection."""
        self.is_connected = False
    
    async def get_balance(self, asset: str) -> Balance:
        """Get mock balance."""
        balance = self.balances.get(asset, 0.0)
        return Balance(
            asset=asset,
            free=balance,
            locked=0.0,
            total=balance
        )
    
    async def get_all_balances(self) -> Dict[str, Balance]:
        """Get all mock balances."""
        return {
            asset: Balance(
                asset=asset,
                free=balance,
                locked=0.0,
                total=balance
            )
            for asset, balance in self.balances.items()
        }
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get mock ticker."""
        price = self.prices.get(symbol, 100.0)
        return Ticker(
            symbol=symbol,
            price=price,
            volume=1000000.0,
            timestamp=datetime.utcnow(),
            bid=price * 0.999,
            ask=price * 1.001,
            high_24h=price * 1.05,
            low_24h=price * 0.95,
            change_24h=price * 0.02,
            change_percent_24h=2.0
        )
    
    async def get_tickers(self, symbols: List[str]) -> Dict[str, Ticker]:
        """Get mock tickers."""
        return {symbol: await self.get_ticker(symbol) for symbol in symbols}
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        **kwargs
    ) -> Order:
        """Place mock order."""
        self.order_counter += 1
        order_id = f"mock_mexc_{self.order_counter}"
        
        # Get current price if not provided
        if price is None:
            ticker = await self.get_ticker(symbol)
            price = ticker.price
        
        # Check balance for buy orders
        if side == OrderSide.BUY:
            base_asset = "USDT"
            required_balance = amount * price
            current_balance = self.balances.get(base_asset, 0.0)
            
            if current_balance < required_balance:
                raise InsufficientBalanceError(f"Insufficient {base_asset} balance")
            
            # Deduct balance
            self.balances[base_asset] -= required_balance
        
        # Create order
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            amount=amount,
            price=price,
            status=OrderStatus.FILLED,
            filled_amount=amount,
            remaining_amount=0.0,
            average_price=price,
            fee=amount * price * 0.001,  # 0.1% MEXC fee
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            client_order_id=client_order_id
        )
        
        # Store order
        self.orders[order_id] = order
        
        # Update balances
        if side == OrderSide.BUY:
            quote_asset = symbol.replace("USDT", "")
            self.balances[quote_asset] = self.balances.get(quote_asset, 0.0) + amount
        else:
            quote_asset = symbol.replace("USDT", "")
            base_asset = "USDT"
            self.balances[quote_asset] -= amount
            self.balances[base_asset] += amount * price
        
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel mock order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.utcnow()
                return True
        return False
    
    async def get_order(self, order_id: str) -> Order:
        """Get mock order."""
        if order_id in self.orders:
            return self.orders[order_id]
        else:
            raise ExchangeError(f"Order {order_id} not found")
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get mock open orders."""
        orders = []
        for order in self.orders.values():
            if order.status == OrderStatus.PENDING:
                if symbol is None or order.symbol == symbol:
                    orders.append(order)
        return orders
    
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Trade]:
        """Get mock trades."""
        return []
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get mock order book."""
        return {
            "bids": [],
            "asks": [],
            "timestamp": datetime.utcnow()
        }
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Get mock klines."""
        return []
    
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get mock trading fees."""
        return {
            "maker": 0.001,
            "taker": 0.001,
        }
    
    def set_balance(self, asset: str, amount: float) -> None:
        """Set mock balance for testing."""
        self.balances[asset] = amount
    
    def set_price(self, symbol: str, price: float) -> None:
        """Set mock price for testing."""
        self.prices[symbol] = price
    
    def get_order_count(self) -> int:
        """Get number of orders for testing."""
        return len(self.orders)
    
    def clear_orders(self) -> None:
        """Clear all orders for testing."""
        self.orders.clear()
        self.trades.clear()
