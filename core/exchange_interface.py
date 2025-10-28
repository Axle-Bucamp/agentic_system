"""
Abstract exchange interface for unified trading operations.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


class ExchangeType(str, Enum):
    """Supported exchange types."""
    DEX = "DEX"
    MEXC = "MEXC"
    MOCK = "MOCK"


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(str, Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order statuses."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"


class Balance(BaseModel):
    """Account balance for a specific asset."""
    asset: str
    free: float = Field(description="Available balance")
    locked: float = Field(description="Locked in orders")
    total: float = Field(description="Total balance")
    
    @property
    def available(self) -> float:
        """Available balance for trading."""
        return self.free


class Ticker(BaseModel):
    """Market ticker information."""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None
    change_percent_24h: Optional[float] = None


class Order(BaseModel):
    """Order information."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    amount: float
    price: Optional[float] = None
    status: OrderStatus
    filled_amount: float = 0.0
    remaining_amount: float = 0.0
    average_price: Optional[float] = None
    fee: float = 0.0
    created_at: datetime
    updated_at: datetime
    client_order_id: Optional[str] = None
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is active (pending or partially filled)."""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]


class Trade(BaseModel):
    """Trade execution information."""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    amount: float
    price: float
    fee: float
    fee_asset: str
    timestamp: datetime
    is_maker: bool = False


class ExchangeError(Exception):
    """Base exception for exchange operations."""
    pass


class InsufficientBalanceError(ExchangeError):
    """Insufficient balance for operation."""
    pass


class OrderNotFoundError(ExchangeError):
    """Order not found."""
    pass


class NetworkError(ExchangeError):
    """Network communication error."""
    pass


class ExchangeInterface(ABC):
    """Abstract interface for exchange operations."""
    
    def __init__(self, exchange_type: ExchangeType, config: Dict[str, Any]):
        self.exchange_type = exchange_type
        self.config = config
        self.is_connected = False
        self.is_mock = config.get("mock_mode", False)
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the exchange."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        pass
    
    @abstractmethod
    async def get_balance(self, asset: str) -> Balance:
        """Get balance for a specific asset."""
        pass
    
    @abstractmethod
    async def get_all_balances(self) -> Dict[str, Balance]:
        """Get all account balances."""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker information for a symbol."""
        pass
    
    @abstractmethod
    async def get_tickers(self, symbols: List[str]) -> Dict[str, Ticker]:
        """Get ticker information for multiple symbols."""
        pass
    
    @abstractmethod
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
        """Place a new order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Order:
        """Get order information."""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders."""
        pass
    
    @abstractmethod
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Trade]:
        """Get trade history."""
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book for a symbol."""
        pass
    
    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Get kline/candlestick data."""
        pass
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported by the exchange."""
        try:
            await self.get_ticker(symbol)
            return True
        except ExchangeError:
            return False
    
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for a symbol."""
        # Default implementation - can be overridden
        return {
            "maker": 0.001,  # 0.1%
            "taker": 0.001,  # 0.1%
        }
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information."""
        return {
            "exchange_type": self.exchange_type.value,
            "is_connected": self.is_connected,
            "is_mock": self.is_mock,
            "config": {k: v for k, v in self.config.items() if "key" not in k.lower()}
        }
