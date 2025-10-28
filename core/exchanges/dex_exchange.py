"""
DEX (Decentralized Exchange) integration for Uniswap V3 and PancakeSwap V2.
"""
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from core.exchange_interface import (
    ExchangeInterface, ExchangeType, OrderSide, OrderType, OrderStatus,
    Balance, Ticker, Order, Trade, ExchangeError, InsufficientBalanceError
)
from core.logging import log


class DEXExchange(ExchangeInterface):
    """DEX exchange implementation for Uniswap V3 and PancakeSwap V2."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ExchangeType.DEX, config)
        
        # Network configuration
        self.network = config.get("network", "ethereum")
        self.rpc_url = config.get("rpc_url")
        self.chain_id = config.get("chain_id", 1)
        
        # Wallet configuration
        self.private_key = config.get("private_key")
        self.wallet_address = config.get("wallet_address")
        
        # DEX configuration
        self.dex_type = config.get("dex_type", "uniswap_v3")
        self.router_address = config.get("router_address")
        self.factory_address = config.get("factory_address")
        self.weth_address = config.get("weth_address")
        
        # Slippage and gas settings
        self.slippage_tolerance = config.get("slippage_tolerance", 0.005)  # 0.5%
        self.gas_limit = config.get("gas_limit", 300000)
        self.gas_price_multiplier = config.get("gas_price_multiplier", 1.1)
        
        # Web3 instance
        self.w3: Optional[Web3] = None
        self.account: Optional[Account] = None
        
        # Contract instances
        self.router_contract = None
        self.factory_contract = None
        
        # Token addresses cache
        self.token_addresses: Dict[str, str] = {}
        
        # Mock mode data
        self.mock_balances: Dict[str, float] = {}
        self.mock_prices: Dict[str, float] = {}
        self.mock_orders: Dict[str, Order] = {}
        self.order_counter = 0
    
    async def connect(self) -> None:
        """Connect to the blockchain network."""
        try:
            if self.is_mock:
                await self._setup_mock_mode()
                self.is_connected = True
                log.info("DEX Exchange connected in mock mode")
                return
            
            # Initialize Web3
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            
            # Add PoA middleware for BSC/Polygon
            if self.chain_id in [56, 137]:  # BSC or Polygon
                self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Verify connection
            if not self.w3.is_connected():
                raise ExchangeError("Failed to connect to blockchain network")
            
            # Initialize account
            if self.private_key:
                self.account = Account.from_key(self.private_key)
                if self.wallet_address and self.account.address.lower() != self.wallet_address.lower():
                    raise ExchangeError("Private key does not match wallet address")
                self.wallet_address = self.account.address
            
            # Initialize contracts
            await self._initialize_contracts()
            
            self.is_connected = True
            log.info(f"DEX Exchange connected to {self.network} network")
            
        except Exception as e:
            log.error(f"Failed to connect to DEX: {e}")
            raise ExchangeError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        self.w3 = None
        self.account = None
        self.router_contract = None
        self.factory_contract = None
        self.is_connected = False
        log.info("DEX Exchange disconnected")
    
    async def _setup_mock_mode(self) -> None:
        """Setup mock mode with sample data."""
        # Initialize mock balances
        self.mock_balances = {
            "USDC": 10000.0,
            "WETH": 5.0,
            "USDT": 5000.0,
        }
        
        # Initialize mock prices
        self.mock_prices = {
            "ETHUSDC": 2000.0,
            "BTCUSDC": 45000.0,
            "USDTUSDC": 1.0,
        }
    
    async def _initialize_contracts(self) -> None:
        """Initialize DEX contracts."""
        # This would load actual contract ABIs and create contract instances
        # For now, we'll use mock implementations
        log.info("Initializing DEX contracts...")
    
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
        
        # Real implementation would query the blockchain
        # For now, return mock data
        return Balance(
            asset=asset,
            free=0.0,
            locked=0.0,
            total=0.0
        )
    
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
        
        # Real implementation would query multiple tokens
        return {}
    
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
        
        # Real implementation would query DEX for current price
        # This would involve calling the router contract's getAmountsOut
        return Ticker(
            symbol=symbol,
            price=100.0,
            volume=0.0,
            timestamp=datetime.utcnow()
        )
    
    async def get_tickers(self, symbols: List[str]) -> Dict[str, Ticker]:
        """Get ticker information for multiple symbols."""
        tickers = {}
        for symbol in symbols:
            try:
                tickers[symbol] = await self.get_ticker(symbol)
            except Exception as e:
                log.error(f"Failed to get ticker for {symbol}: {e}")
        
        return tickers
    
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
        
        # Real implementation would interact with DEX contracts
        # This involves:
        # 1. Approve token spending if needed
        # 2. Calculate swap parameters
        # 3. Execute swap transaction
        # 4. Wait for confirmation
        
        order_id = f"dex_{int(datetime.utcnow().timestamp() * 1000)}"
        
        return Order(
            id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            amount=amount,
            price=price,
            status=OrderStatus.PENDING,
            filled_amount=0.0,
            remaining_amount=amount,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            client_order_id=client_order_id
        )
    
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
        order_id = f"mock_dex_{self.order_counter}"
        
        # Get current price if not provided
        if price is None:
            ticker = await self.get_ticker(symbol)
            price = ticker.price
        
        # Check balance for buy orders
        if side == OrderSide.BUY:
            base_asset = "USDC"  # Assume USDC as base
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
            fee=amount * price * 0.003,  # 0.3% DEX fee
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            client_order_id=client_order_id
        )
        
        # Store mock order
        self.mock_orders[order_id] = order
        
        # Update balances
        if side == OrderSide.BUY:
            quote_asset = symbol.replace("USDC", "")
            self.mock_balances[quote_asset] = self.mock_balances.get(quote_asset, 0.0) + amount
        else:
            quote_asset = symbol.replace("USDC", "")
            base_asset = "USDC"
            self.mock_balances[quote_asset] -= amount
            self.mock_balances[base_asset] += amount * price
        
        log.info(f"Mock DEX order placed: {order_id} - {side.value} {amount} {symbol} @ {price}")
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if self.is_mock:
            if order_id in self.mock_orders:
                order = self.mock_orders[order_id]
                if order.status == OrderStatus.PENDING:
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = datetime.utcnow()
                    log.info(f"Mock DEX order cancelled: {order_id}")
                    return True
            return False
        
        # Real implementation would cancel the transaction if still pending
        return False
    
    async def get_order(self, order_id: str) -> Order:
        """Get order information."""
        if self.is_mock:
            if order_id in self.mock_orders:
                return self.mock_orders[order_id]
            else:
                raise ExchangeError(f"Order {order_id} not found")
        
        # Real implementation would query blockchain for transaction status
        raise ExchangeError("Order lookup not implemented for real DEX")
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders."""
        if self.is_mock:
            orders = []
            for order in self.mock_orders.values():
                if order.status == OrderStatus.PENDING:
                    if symbol is None or order.symbol == symbol:
                        orders.append(order)
            return orders
        
        # DEX doesn't have traditional open orders
        return []
    
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Trade]:
        """Get trade history."""
        # DEX trades are typically queried from blockchain events
        # This would involve filtering transaction logs
        return []
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book for a symbol."""
        # DEX doesn't have traditional order books
        # This would return liquidity information from pools
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
        """Get kline/candlestick data."""
        # This would query historical price data from a data provider
        # or calculate from DEX transaction history
        return []
    
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for a symbol."""
        # DEX fees are typically fixed per pool
        return {
            "maker": 0.003,  # 0.3%
            "taker": 0.003,  # 0.3%
        }
