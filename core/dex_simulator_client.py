"""
DEX Simulator Client - Interfaces with the DEX simulator API for paper trading.
"""
import httpx
from typing import Optional, Dict, Any
from datetime import datetime
from core.logging import log
from core.config import settings


class DEXSimulatorError(Exception):
    """Base exception for DEX simulator operations."""
    pass


class DEXSimulatorClient:
    """Client for interacting with the DEX simulator API."""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or settings.dex_simulator_url
        self.client: Optional[httpx.AsyncClient] = None
        
    async def connect(self) -> None:
        """Initialize the HTTP client."""
        try:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                verify=False  # Development only
            )
            log.info(f"DEX Simulator client connected to {self.base_url}")
        except Exception as e:
            log.error(f"Failed to connect to DEX simulator: {e}")
            raise DEXSimulatorError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
        log.info("DEX Simulator client disconnected")
    
    async def get_wallet_balance(self) -> float:
        """Get wallet USDC balance."""
        try:
            response = await self.client.get("/wallet_usdc")
            response.raise_for_status()
            data = response.json()
            return data.get("wallet_usdc", 0.0)
        except Exception as e:
            log.error(f"Error getting wallet balance: {e}")
            return 0.0
    
    async def get_total_portfolio_value(self) -> float:
        """Get total portfolio value in USDC."""
        try:
            response = await self.client.get("/ticker_sum_usdc")
            response.raise_for_status()
            data = response.json()
            return data.get("total_value", 0.0)
        except Exception as e:
            log.error(f"Error getting portfolio value: {e}")
            return 0.0
    
    async def get_current_holdings(self) -> Dict[str, float]:
        """Get current asset holdings."""
        try:
            response = await self.client.get("/wallet_holding")
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}
        except Exception as e:
            log.error(f"Error getting holdings: {e}")
            return {}
    
    async def get_ticker_prices(self) -> Dict[str, float]:
        """Get current ticker prices."""
        try:
            response = await self.client.get("/ticker_prices_usdc")
            response.raise_for_status()
            data = response.json()
            # Extract prices from the response
            # Format: {"BTC": 50000.0, "ETH": 3000.0, ...}
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, (int, float))}
            return {}
        except Exception as e:
            log.error(f"Error getting ticker prices: {e}")
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
        try:
            # For POST requests with form data
            response = await self.client.post(
                "/buy",
                data={"ticker": ticker, "amount": amount}
            )
            response.raise_for_status()
            
            return {
                "success": True,
                "ticker": ticker,
                "amount_usdc": amount,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            log.error(f"Error buying {ticker}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def sell_asset(self, ticker: str, amount: float) -> Dict[str, Any]:
        """
        Sell an asset on the DEX simulator.
        
        Args:
            ticker: Asset ticker (e.g., "BTC")
            amount: Amount to sell in asset units
            
        Returns:
            Dict with trade execution details
        """
        try:
            response = await self.client.post(
                "/sell",
                data={"ticker": ticker, "amount": amount}
            )
            response.raise_for_status()
            
            return {
                "success": True,
                "ticker": ticker,
                "amount": amount,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            log.error(f"Error selling {ticker}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get complete portfolio status."""
        try:
            balance = await self.get_wallet_balance()
            total_value = await self.get_total_portfolio_value()
            holdings = await self.get_current_holdings()
            prices = await self.get_ticker_prices()
            
            return {
                "balance_usdc": balance,
                "total_value_usdc": total_value,
                "holdings": holdings,
                "prices": prices,
                "daily_pnl": total_value - settings.initial_capital,
                "total_pnl": total_value - settings.initial_capital,
                "positions": [
                    {
                        "ticker": ticker,
                        "quantity": qty,
                        "price": prices.get(ticker, 0),
                        "value": qty * prices.get(ticker, 0)
                    }
                    for ticker, qty in holdings.items()
                ]
            }
        except Exception as e:
            log.error(f"Error getting portfolio status: {e}")
            return {
                "balance_usdc": 0,
                "total_value_usdc": 0,
                "holdings": {},
                "prices": {},
                "daily_pnl": 0,
                "total_pnl": 0,
                "positions": []
            }


# Global DEX simulator client instance
dex_simulator_client = DEXSimulatorClient()

