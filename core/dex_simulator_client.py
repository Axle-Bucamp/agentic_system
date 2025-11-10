"""
DEX Simulator Client - Interfaces with the DEX simulator API for paper trading.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import httpx

from core import asset_registry
from core.config import settings
from core.logging import log


class DEXSimulatorError(Exception):
    """Base exception for DEX simulator operations."""

    pass


@dataclass
class _FakeTradeRecord:
    timestamp: datetime
    ticker: str
    side: str
    quantity: float
    price: float
    fee: float
    notional: float
    balance_after: float


class _FakeWallet:
    """Simple in-memory wallet used when the external DEX simulator is unavailable."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._trading_fee = float(settings.trading_fee or 0.0)
        self._balance_usdc: float = float(settings.initial_capital or 0.0)
        self._holdings: Dict[str, float] = {}
        self._prices: Dict[str, float] = {}
        self._trades: List[_FakeTradeRecord] = []
        self._initialize_holdings()

    def _initialize_holdings(self) -> None:
        assets = asset_registry.get_assets() or settings.supported_assets
        for asset in assets:
            self._holdings[asset] = 1000.0
            # Provide a sensible baseline price so valuation isn't zeroed out.
            self._prices.setdefault(asset, 1.0)
        self._prices.setdefault("USDC", 1.0)

    async def buy(self, ticker: str, amount_usdc: float) -> Dict[str, Any]:
        async with self._lock:
            price = self._resolve_price(ticker)
            gross_quantity = amount_usdc / price if price > 0 else 0.0
            fee_cost = amount_usdc * self._trading_fee
            total_cost = amount_usdc + fee_cost

            if total_cost > self._balance_usdc + 1e-9:
                raise DEXSimulatorError(
                    f"Insufficient USDC balance for {ticker}: "
                    f"required {total_cost:.4f}, available {self._balance_usdc:.4f}"
                )

            net_quantity = gross_quantity
            self._balance_usdc -= total_cost
            self._holdings[ticker] = self._holdings.get(ticker, 0.0) + net_quantity

            record = _FakeTradeRecord(
                timestamp=datetime.utcnow(),
                ticker=ticker,
                side="BUY",
                quantity=net_quantity,
                price=price,
                fee=fee_cost,
                notional=amount_usdc,
                balance_after=self._balance_usdc,
            )
            self._trades.append(record)

            return {
                "success": True,
                "ticker": ticker,
                "side": "BUY",
                "filled_quantity": net_quantity,
                "requested_amount_usdc": amount_usdc,
                "executed_price": price,
                "fee_paid": fee_cost,
                "total_cost_usdc": total_cost,
                "balance_usdc": self._balance_usdc,
                "timestamp": record.timestamp.isoformat(),
            }

    async def sell(self, ticker: str, quantity: float) -> Dict[str, Any]:
        async with self._lock:
            if quantity <= 0:
                raise DEXSimulatorError("Sell quantity must be positive")

            position = self._holdings.get(ticker, 0.0)
            if quantity > position + 1e-9:
                raise DEXSimulatorError(
                    f"Insufficient {ticker} holdings: attempting to sell {quantity:.4f}, "
                    f"available {position:.4f}"
                )

            price = self._resolve_price(ticker)
            gross_proceeds = quantity * price
            fee_cost = gross_proceeds * self._trading_fee
            net_proceeds = gross_proceeds - fee_cost

            self._holdings[ticker] = position - quantity
            self._balance_usdc += net_proceeds

            record = _FakeTradeRecord(
                timestamp=datetime.utcnow(),
                ticker=ticker,
                side="SELL",
                quantity=quantity,
                price=price,
                fee=fee_cost,
                notional=gross_proceeds,
                balance_after=self._balance_usdc,
            )
            self._trades.append(record)

            return {
                "success": True,
                "ticker": ticker,
                "side": "SELL",
                "filled_quantity": quantity,
                "executed_price": price,
                "fee_paid": fee_cost,
                "gross_proceeds_usdc": gross_proceeds,
                "net_proceeds_usdc": net_proceeds,
                "balance_usdc": self._balance_usdc,
                "timestamp": record.timestamp.isoformat(),
            }

    async def get_balance(self) -> float:
        async with self._lock:
            return self._balance_usdc

    async def get_holdings(self) -> Dict[str, float]:
        async with self._lock:
            # Include zero-quantity entries so downstream agents keep visibility.
            assets = asset_registry.get_assets() or settings.supported_assets
            return {asset: self._holdings.get(asset, 0.0) for asset in assets}

    async def get_prices(self) -> Dict[str, float]:
        async with self._lock:
            prices = dict(self._prices)
            assets = asset_registry.get_assets() or settings.supported_assets
            for asset in assets:
                prices.setdefault(asset, self._prices.get(asset, 1.0))
            return prices

    async def update_price(self, ticker: str, price: float) -> None:
        if price <= 0:
            return
        async with self._lock:
            self._prices[ticker] = price

    async def get_trade_history(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [
                {
                    "timestamp": record.timestamp.isoformat(),
                    "ticker": record.ticker,
                    "side": record.side,
                    "quantity": record.quantity,
                    "price": record.price,
                    "fee": record.fee,
                    "notional": record.notional,
                    "balance_after": record.balance_after,
                }
                for record in self._trades
            ]

    def _resolve_price(self, ticker: str) -> float:
        return max(self._prices.get(ticker, 1.0), 1e-6)

    async def get_total_value(self) -> float:
        async with self._lock:
            assets = asset_registry.get_assets() or settings.supported_assets
            holdings_value = 0.0
            for ticker in assets:
                quantity = self._holdings.get(ticker, 0.0)
                price = self._prices.get(ticker, 1.0)
                holdings_value += quantity * price
            return self._balance_usdc + holdings_value


class DEXSimulatorClient:
    """Client for interacting with the DEX simulator API."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.dex_simulator_url
        self.client: Optional[httpx.AsyncClient] = None
        self._use_fake_wallet: bool = False
        self._wallet: Optional[_FakeWallet] = None

    async def connect(self) -> None:
        """Initialize the HTTP client or the fallback fake wallet."""
        if settings.use_mock_services:
            self._activate_fake_wallet("mock services enabled")
            return

        try:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                verify=False,  # Development only
            )
            # Lightweight sanity check to confirm connectivity.
            await self.client.get("/health")
            log.info("DEX Simulator client connected to %s", self.base_url)
        except Exception as exc:
            log.warning(
                "DEX simulator unavailable (%s); falling back to in-memory wallet.",
                exc,
            )
            self._activate_fake_wallet(str(exc))

    def _activate_fake_wallet(self, reason: str) -> None:
        self._use_fake_wallet = True
        self._wallet = _FakeWallet()
        # Provide a sentinel object so orchestrator checks for `client` pass.
        self.client = object()  # type: ignore[assignment]
        log.info("Using fake DEX wallet (%s).", reason)

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self._use_fake_wallet:
            self._wallet = None
            self.client = None
            log.info("Fake DEX wallet disconnected")
            return

        if self.client and isinstance(self.client, httpx.AsyncClient):
            await self.client.aclose()
        self.client = None
        log.info("DEX Simulator client disconnected")

    async def get_wallet_balance(self) -> float:
        """Get wallet USDC balance."""
        if self._use_fake_wallet and self._wallet:
            return await self._wallet.get_balance()

        try:
            response = await self.client.get("/wallet_usdc")  # type: ignore[union-attr]
            response.raise_for_status()
            data = response.json()
            return data.get("wallet_usdc", 0.0)
        except Exception as exc:
            log.error("Error getting wallet balance: %s", exc)
            return 0.0

    async def get_total_portfolio_value(self) -> float:
        """Get total portfolio value in USDC."""
        if self._use_fake_wallet and self._wallet:
            return await self._wallet.get_total_value()

        try:
            response = await self.client.get("/ticker_sum_usdc")  # type: ignore[union-attr]
            response.raise_for_status()
            data = response.json()
            return data.get("total_value", 0.0)
        except Exception as exc:
            log.error("Error getting portfolio value: %s", exc)
            return 0.0

    async def get_current_holdings(self) -> Dict[str, float]:
        """Get current asset holdings."""
        if self._use_fake_wallet and self._wallet:
            return await self._wallet.get_holdings()

        try:
            response = await self.client.get("/wallet_holding")  # type: ignore[union-attr]
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            log.error("Error getting holdings: %s", exc)
            return {}

    async def get_ticker_prices(self) -> Dict[str, float]:
        """Get current ticker prices."""
        if self._use_fake_wallet and self._wallet:
            return await self._wallet.get_prices()

        try:
            response = await self.client.get("/ticker_prices_usdc")  # type: ignore[union-attr]
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, (int, float))}
            return {}
        except Exception as exc:
            log.error("Error getting ticker prices: %s", exc)
            return {}

    async def buy_asset(self, ticker: str, amount: float) -> Dict[str, Any]:
        """
        Buy an asset on the DEX simulator.

        Args:
            ticker: Asset ticker (e.g., "BTC")
            amount: Amount to buy in USDC

        Returns:
            Dict with trade execution details
        """
        if self._use_fake_wallet and self._wallet:
            try:
                return await self._wallet.buy(ticker.upper(), float(amount))
            except DEXSimulatorError:
                raise
            except Exception as exc:
                raise DEXSimulatorError(f"Fake wallet buy failed: {exc}") from exc

        try:
            response = await self.client.post(  # type: ignore[union-attr]
                "/buy",
                data={"ticker": ticker, "amount": amount},
            )
            response.raise_for_status()
            payload = response.json() if response.content else {}
            price = float(payload.get("price", 0.0)) if isinstance(payload, dict) else 0.0
            quantity = float(payload.get("quantity", amount / price if price else 0.0)) if isinstance(payload, dict) else 0.0
            fee = float(payload.get("fee", quantity * price * settings.trading_fee))

            return {
                "success": True,
                "ticker": ticker,
                "side": "BUY",
                "filled_quantity": quantity,
                "requested_amount_usdc": amount,
                "executed_price": price,
                "fee_paid": fee,
                "total_cost_usdc": amount + fee,
                "raw_response": payload,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as exc:
            log.error("Error buying %s: %s", ticker, exc)
            raise DEXSimulatorError(f"Error buying {ticker}: {exc}") from exc

    async def sell_asset(self, ticker: str, amount: float) -> Dict[str, Any]:
        """
        Sell an asset on the DEX simulator.

        Args:
            ticker: Asset ticker (e.g., "BTC")
            amount: Amount to sell in asset units

        Returns:
            Dict with trade execution details
        """
        if self._use_fake_wallet and self._wallet:
            try:
                return await self._wallet.sell(ticker.upper(), float(amount))
            except DEXSimulatorError:
                raise
            except Exception as exc:
                raise DEXSimulatorError(f"Fake wallet sell failed: {exc}") from exc

        try:
            response = await self.client.post(  # type: ignore[union-attr]
                "/sell",
                data={"ticker": ticker, "amount": amount},
            )
            response.raise_for_status()
            payload = response.json() if response.content else {}
            price = float(payload.get("price", 0.0)) if isinstance(payload, dict) else 0.0
            fee = float(payload.get("fee", amount * price * settings.trading_fee))
            proceeds = float(payload.get("proceeds", amount * price))

            return {
                "success": True,
                "ticker": ticker,
                "side": "SELL",
                "filled_quantity": amount,
                "executed_price": price,
                "fee_paid": fee,
                "gross_proceeds_usdc": proceeds,
                "net_proceeds_usdc": proceeds - fee,
                "raw_response": payload,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as exc:
            log.error("Error selling %s: %s", ticker, exc)
            raise DEXSimulatorError(f"Error selling {ticker}: {exc}") from exc

    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get complete portfolio status."""
        try:
            balance = await self.get_wallet_balance()
            total_value = await self.get_total_portfolio_value()
            holdings = await self.get_current_holdings()
            prices = await self.get_ticker_prices()

            positions = []
            for ticker, quantity in holdings.items():
                price = prices.get(ticker, 0.0)
                positions.append(
                    {
                        "ticker": ticker,
                        "quantity": quantity,
                        "price": price,
                        "value": quantity * price,
                    }
                )

            return {
                "balance_usdc": balance,
                "total_value_usdc": total_value,
                "holdings": holdings,
                "prices": prices,
                "daily_pnl": total_value - settings.initial_capital,
                "total_pnl": total_value - settings.initial_capital,
                "positions": positions,
                "use_fake_wallet": self._use_fake_wallet,
            }
        except Exception as exc:
            log.error("Error getting portfolio status: %s", exc)
            return {
                "balance_usdc": 0.0,
                "total_value_usdc": 0.0,
                "holdings": {},
                "prices": {},
                "daily_pnl": 0.0,
                "total_pnl": 0.0,
                "positions": [],
                "use_fake_wallet": self._use_fake_wallet,
            }


# Global DEX simulator client instance
dex_simulator_client = DEXSimulatorClient()

