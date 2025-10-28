"""
MEXC exchange integration using REST API v3.
"""
import asyncio
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from urllib.parse import urlencode
import httpx
from core.exchange_interface import (
    ExchangeInterface, ExchangeType, OrderSide, OrderType, OrderStatus,
    Balance, Ticker, Order, Trade, ExchangeError, InsufficientBalanceError
)
from core.logging import log


class MEXCExchange(ExchangeInterface):
    """MEXC exchange implementation using REST API v3."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ExchangeType.MEXC, config)
        
        # API configuration
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.base_url = config.get("base_url", "https://api.mexc.com")
        
        # Rate limiting
        self.rate_limit_delay = config.get("rate_limit_delay", 0.1)
        self.last_request_time = 0
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        
        # Mock mode data
        self.mock_balances: Dict[str, float] = {}
        self.mock_prices: Dict[str, float] = {}
        self.mock_orders: Dict[str, Order] = {}
        self.order_counter = 0
    
    async def connect(self) -> None:
        """Connect to MEXC API."""
        try:
            if self.is_mock:
                await self._setup_mock_mode()
                self.is_connected = True
                log.info("MEXC Exchange connected in mock mode")
                return
            
            # Initialize HTTP client
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                headers={
                    "X-MEXC-APIKEY": self.api_key,
                    "Content-Type": "application/json"
                }
            )
            
            # Test connection
            await self._test_connection()
            
            self.is_connected = True
            log.info("MEXC Exchange connected")
            
        except Exception as e:
            log.error(f"Failed to connect to MEXC: {e}")
            raise ExchangeError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        if self.client:
            await self.client.aclose()
            self.client = None
        self.is_connected = False
        log.info("MEXC Exchange disconnected")
    
    async def _setup_mock_mode(self) -> None:
        """Setup mock mode with sample data."""
        # Initialize mock balances
        self.mock_balances = {
            "USDT": 10000.0,
            "BTC": 0.5,
            "ETH": 5.0,
            "USDC": 5000.0,
        }
        
        # Initialize mock prices
        self.mock_prices = {
            "BTCUSDT": 45000.0,
            "ETHUSDT": 2000.0,
            "USDCUSDT": 1.0,
        }
    
    async def _test_connection(self) -> None:
        """Test API connection."""
        try:
            response = await self._make_request("GET", "/api/v3/ping")
            if response.get("code") != 200:
                raise ExchangeError("API connection test failed")
        except Exception as e:
            raise ExchangeError(f"Connection test failed: {e}")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """Make HTTP request to MEXC API."""
        if not self.client:
            raise ExchangeError("Not connected to MEXC")
        
        # Rate limiting
        await self._apply_rate_limit()
        
        # Prepare request
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if signed:
            if not self.api_key or not self.secret_key:
                raise ExchangeError("API credentials required for signed requests")
            
            # Add signature
            timestamp = int(time.time() * 1000)
            query_string = urlencode(params or {})
            signature_string = f"{self.api_key}{timestamp}{query_string}"
            signature = hmac.new(
                self.secret_key.encode('utf-8'),
                signature_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            params = params or {}
            params.update({
                "api_key": self.api_key,
                "timestamp": timestamp,
                "signature": signature
            })
        
        # Make request
        try:
            if method == "GET":
                response = await self.client.get(url, params=params)
            elif method == "POST":
                response = await self.client.post(url, params=params, json=data)
            elif method == "DELETE":
                response = await self.client.delete(url, params=params)
            else:
                raise ExchangeError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            result = response.json()
            
            # Check for API errors
            if "code" in result and result["code"] != 200:
                raise ExchangeError(f"MEXC API error: {result.get('msg', 'Unknown error')}")
            
            return result
            
        except httpx.HTTPStatusError as e:
            raise ExchangeError(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise ExchangeError(f"Request failed: {e}")
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    async def get_balance(self, asset: str) -> Balance:
        """Get balance for a specific asset."""
        if self.is_mock:
            balance = self.mock_balances.get(asset, 0.0)
            return Balance(
                asset=asset,
                free=balance,
                locked=0.0,
                total=balance
            )
        
        try:
            response = await self._make_request("GET", "/api/v3/account", signed=True)
            balances = response.get("balances", [])
            
            for balance_data in balances:
                if balance_data["asset"] == asset:
                    return Balance(
                        asset=asset,
                        free=float(balance_data["free"]),
                        locked=float(balance_data["locked"]),
                        total=float(balance_data["free"]) + float(balance_data["locked"])
                    )
            
            return Balance(asset=asset, free=0.0, locked=0.0, total=0.0)
            
        except Exception as e:
            log.error(f"Failed to get balance for {asset}: {e}")
            raise ExchangeError(f"Failed to get balance: {e}")
    
    async def get_all_balances(self) -> Dict[str, Balance]:
        """Get all account balances."""
        if self.is_mock:
            return {
                asset: Balance(
                    asset=asset,
                    free=balance,
                    locked=0.0,
                    total=balance
                )
                for asset, balance in self.mock_balances.items()
            }
        
        try:
            response = await self._make_request("GET", "/api/v3/account", signed=True)
            balances = response.get("balances", [])
            
            result = {}
            for balance_data in balances:
                asset = balance_data["asset"]
                free = float(balance_data["free"])
                locked = float(balance_data["locked"])
                
                if free > 0 or locked > 0:  # Only include non-zero balances
                    result[asset] = Balance(
                        asset=asset,
                        free=free,
                        locked=locked,
                        total=free + locked
                    )
            
            return result
            
        except Exception as e:
            log.error(f"Failed to get all balances: {e}")
            raise ExchangeError(f"Failed to get balances: {e}")
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker information for a symbol."""
        if self.is_mock:
            price = self.mock_prices.get(symbol, 100.0)
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
        
        try:
            response = await self._make_request("GET", f"/api/v3/ticker/24hr", params={"symbol": symbol})
            
            return Ticker(
                symbol=symbol,
                price=float(response["lastPrice"]),
                volume=float(response["volume"]),
                timestamp=datetime.utcnow(),
                bid=float(response.get("bidPrice", 0)),
                ask=float(response.get("askPrice", 0)),
                high_24h=float(response.get("highPrice", 0)),
                low_24h=float(response.get("lowPrice", 0)),
                change_24h=float(response.get("priceChange", 0)),
                change_percent_24h=float(response.get("priceChangePercent", 0))
            )
            
        except Exception as e:
            log.error(f"Failed to get ticker for {symbol}: {e}")
            raise ExchangeError(f"Failed to get ticker: {e}")
    
    async def get_tickers(self, symbols: List[str]) -> Dict[str, Ticker]:
        """Get ticker information for multiple symbols."""
        tickers = {}
        
        if self.is_mock:
            for symbol in symbols:
                tickers[symbol] = await self.get_ticker(symbol)
            return tickers
        
        try:
            response = await self._make_request("GET", "/api/v3/ticker/24hr")
            
            for ticker_data in response:
                symbol = ticker_data["symbol"]
                if symbol in symbols:
                    tickers[symbol] = Ticker(
                        symbol=symbol,
                        price=float(ticker_data["lastPrice"]),
                        volume=float(ticker_data["volume"]),
                        timestamp=datetime.utcnow(),
                        bid=float(ticker_data.get("bidPrice", 0)),
                        ask=float(ticker_data.get("askPrice", 0)),
                        high_24h=float(ticker_data.get("highPrice", 0)),
                        low_24h=float(ticker_data.get("lowPrice", 0)),
                        change_24h=float(ticker_data.get("priceChange", 0)),
                        change_percent_24h=float(ticker_data.get("priceChangePercent", 0))
                    )
            
            return tickers
            
        except Exception as e:
            log.error(f"Failed to get tickers: {e}")
            raise ExchangeError(f"Failed to get tickers: {e}")
    
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
        if self.is_mock:
            return await self._place_mock_order(symbol, side, order_type, amount, price, client_order_id)
        
        try:
            # Prepare order parameters
            params = {
                "symbol": symbol,
                "side": side.value.upper(),
                "type": order_type.value.upper(),
                "quantity": str(amount),
            }
            
            if price is not None:
                params["price"] = str(price)
            
            if client_order_id:
                params["newClientOrderId"] = client_order_id
            
            # Add time in force if specified
            if "time_in_force" in kwargs:
                params["timeInForce"] = kwargs["time_in_force"]
            
            response = await self._make_request("POST", "/api/v3/order", params=params, signed=True)
            
            return Order(
                id=str(response["orderId"]),
                symbol=symbol,
                side=side,
                type=order_type,
                amount=amount,
                price=price,
                status=OrderStatus(response["status"].lower()),
                filled_amount=float(response.get("executedQty", 0)),
                remaining_amount=float(response.get("origQty", amount)) - float(response.get("executedQty", 0)),
                average_price=float(response.get("avgPrice", 0)) if response.get("avgPrice") else None,
                fee=float(response.get("cummulativeQuoteQty", 0)) * 0.001,  # Estimate fee
                created_at=datetime.fromtimestamp(response["transactTime"] / 1000),
                updated_at=datetime.fromtimestamp(response["transactTime"] / 1000),
                client_order_id=response.get("clientOrderId")
            )
            
        except Exception as e:
            log.error(f"Failed to place order: {e}")
            raise ExchangeError(f"Failed to place order: {e}")
    
    async def _place_mock_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Order:
        """Place a mock order for testing."""
        self.order_counter += 1
        order_id = f"mock_mexc_{self.order_counter}"
        
        # Get current price if not provided
        if price is None:
            ticker = await self.get_ticker(symbol)
            price = ticker.price
        
        # Check balance for buy orders
        if side == OrderSide.BUY:
            base_asset = "USDT"  # Assume USDT as base
            required_balance = amount * price
            current_balance = self.mock_balances.get(base_asset, 0.0)
            
            if current_balance < required_balance:
                raise InsufficientBalanceError(f"Insufficient {base_asset} balance")
            
            # Deduct balance
            self.mock_balances[base_asset] -= required_balance
        
        # Simulate order execution
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
        
        # Store mock order
        self.mock_orders[order_id] = order
        
        # Update balances
        if side == OrderSide.BUY:
            quote_asset = symbol.replace("USDT", "")
            self.mock_balances[quote_asset] = self.mock_balances.get(quote_asset, 0.0) + amount
        else:
            quote_asset = symbol.replace("USDT", "")
            base_asset = "USDT"
            self.mock_balances[quote_asset] -= amount
            self.mock_balances[base_asset] += amount * price
        
        log.info(f"Mock MEXC order placed: {order_id} - {side.value} {amount} {symbol} @ {price}")
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if self.is_mock:
            if order_id in self.mock_orders:
                order = self.mock_orders[order_id]
                if order.status == OrderStatus.PENDING:
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = datetime.utcnow()
                    log.info(f"Mock MEXC order cancelled: {order_id}")
                    return True
            return False
        
        try:
            response = await self._make_request(
                "DELETE", 
                "/api/v3/order", 
                params={"orderId": order_id},
                signed=True
            )
            return response.get("status") == "CANCELED"
            
        except Exception as e:
            log.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Order:
        """Get order information."""
        if self.is_mock:
            if order_id in self.mock_orders:
                return self.mock_orders[order_id]
            else:
                raise ExchangeError(f"Order {order_id} not found")
        
        try:
            response = await self._make_request(
                "GET", 
                "/api/v3/order", 
                params={"orderId": order_id},
                signed=True
            )
            
            return Order(
                id=str(response["orderId"]),
                symbol=response["symbol"],
                side=OrderSide(response["side"].lower()),
                type=OrderType(response["type"].lower()),
                amount=float(response["origQty"]),
                price=float(response["price"]) if response["price"] != "0.00000000" else None,
                status=OrderStatus(response["status"].lower()),
                filled_amount=float(response["executedQty"]),
                remaining_amount=float(response["origQty"]) - float(response["executedQty"]),
                average_price=float(response["avgPrice"]) if response["avgPrice"] != "0.00000000" else None,
                fee=float(response.get("cummulativeQuoteQty", 0)) * 0.001,
                created_at=datetime.fromtimestamp(response["time"] / 1000),
                updated_at=datetime.fromtimestamp(response["updateTime"] / 1000),
                client_order_id=response.get("clientOrderId")
            )
            
        except Exception as e:
            log.error(f"Failed to get order {order_id}: {e}")
            raise ExchangeError(f"Failed to get order: {e}")
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders."""
        if self.is_mock:
            orders = []
            for order in self.mock_orders.values():
                if order.status == OrderStatus.PENDING:
                    if symbol is None or order.symbol == symbol:
                        orders.append(order)
            return orders
        
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol
            
            response = await self._make_request("GET", "/api/v3/openOrders", params=params, signed=True)
            
            orders = []
            for order_data in response:
                orders.append(Order(
                    id=str(order_data["orderId"]),
                    symbol=order_data["symbol"],
                    side=OrderSide(order_data["side"].lower()),
                    type=OrderType(order_data["type"].lower()),
                    amount=float(order_data["origQty"]),
                    price=float(order_data["price"]) if order_data["price"] != "0.00000000" else None,
                    status=OrderStatus(order_data["status"].lower()),
                    filled_amount=float(order_data["executedQty"]),
                    remaining_amount=float(order_data["origQty"]) - float(order_data["executedQty"]),
                    average_price=float(order_data["avgPrice"]) if order_data["avgPrice"] != "0.00000000" else None,
                    fee=0.0,
                    created_at=datetime.fromtimestamp(order_data["time"] / 1000),
                    updated_at=datetime.fromtimestamp(order_data["updateTime"] / 1000),
                    client_order_id=order_data.get("clientOrderId")
                ))
            
            return orders
            
        except Exception as e:
            log.error(f"Failed to get open orders: {e}")
            raise ExchangeError(f"Failed to get open orders: {e}")
    
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Trade]:
        """Get trade history."""
        if self.is_mock:
            return []  # No trade history in mock mode
        
        try:
            params = {"limit": limit}
            if symbol:
                params["symbol"] = symbol
            
            response = await self._make_request("GET", "/api/v3/myTrades", params=params, signed=True)
            
            trades = []
            for trade_data in response:
                trades.append(Trade(
                    id=str(trade_data["id"]),
                    order_id=str(trade_data["orderId"]),
                    symbol=trade_data["symbol"],
                    side=OrderSide(trade_data["isBuyer"] and "buy" or "sell"),
                    amount=float(trade_data["qty"]),
                    price=float(trade_data["price"]),
                    fee=float(trade_data["commission"]),
                    fee_asset=trade_data["commissionAsset"],
                    timestamp=datetime.fromtimestamp(trade_data["time"] / 1000),
                    is_maker=trade_data["isMaker"]
                ))
            
            return trades
            
        except Exception as e:
            log.error(f"Failed to get trades: {e}")
            raise ExchangeError(f"Failed to get trades: {e}")
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book for a symbol."""
        try:
            response = await self._make_request("GET", "/api/v3/depth", params={"symbol": symbol, "limit": limit})
            
            return {
                "bids": [[float(price), float(qty)] for price, qty in response["bids"]],
                "asks": [[float(price), float(qty)] for price, qty in response["asks"]],
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            log.error(f"Failed to get order book for {symbol}: {e}")
            raise ExchangeError(f"Failed to get order book: {e}")
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Get kline/candlestick data."""
        try:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)
            
            response = await self._make_request("GET", "/api/v3/klines", params=params)
            
            klines = []
            for kline_data in response:
                klines.append({
                    "open_time": datetime.fromtimestamp(kline_data[0] / 1000),
                    "close_time": datetime.fromtimestamp(kline_data[6] / 1000),
                    "open": float(kline_data[1]),
                    "high": float(kline_data[2]),
                    "low": float(kline_data[3]),
                    "close": float(kline_data[4]),
                    "volume": float(kline_data[5]),
                    "quote_volume": float(kline_data[7]),
                    "trades": int(kline_data[8]),
                    "taker_buy_volume": float(kline_data[9]),
                    "taker_buy_quote_volume": float(kline_data[10])
                })
            
            return klines
            
        except Exception as e:
            log.error(f"Failed to get klines for {symbol}: {e}")
            raise ExchangeError(f"Failed to get klines: {e}")
    
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for a symbol."""
        try:
            response = await self._make_request("GET", "/api/v3/tradeFee", params={"symbol": symbol}, signed=True)
            
            return {
                "maker": float(response["makerCommission"]),
                "taker": float(response["takerCommission"])
            }
            
        except Exception as e:
            log.error(f"Failed to get trading fees for {symbol}: {e}")
            # Return default fees
            return {
                "maker": 0.001,  # 0.1%
                "taker": 0.001,  # 0.1%
            }
