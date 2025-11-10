"""
DEX Trading Toolkit for CAMEL

Provides CAMEL-compatible tools for executing trades via the DEX simulator.
"""
from typing import Dict, Any, List, Optional, Annotated
from pydantic import Field
from core.config import settings
from core.logging import log
from core.dex_simulator_client import DEXSimulatorClient, DEXSimulatorError

try:
    from camel.toolkits import FunctionTool  # type: ignore
    CAMEL_TOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FunctionTool = None  # type: ignore
    CAMEL_TOOLS_AVAILABLE = False
    log.warning("CAMEL function tools not available. Install with: pip install 'camel-ai[tools]'")


class DEXTradingToolkit:
    """Toolkit for DEX simulator trading operations."""
    
    def __init__(self, dex_client: Optional[DEXSimulatorClient] = None):
        """
        Initialize the toolkit.
        
        Args:
            dex_client: Optional DEXSimulatorClient instance
        """
        self.dex_client = dex_client or DEXSimulatorClient()
    
    async def initialize(self):
        """Initialize the DEX client."""
        await self.dex_client.connect()
    
    def buy_asset_tool(self):
        """Get tool for buying assets."""
        if not CAMEL_TOOLS_AVAILABLE:
            raise ImportError("CAMEL tools not installed")
        
        toolkit_instance = self
        
        async def buy_asset(
            ticker: Annotated[str, Field(description="Asset ticker symbol (e.g., BTC)")],
            amount: Annotated[float, Field(description="Amount to buy in USDC")]
        ) -> Dict[str, Any]:
            """
            Buy an asset on the DEX simulator.
            
            Args:
                ticker: Asset ticker symbol
                amount: Amount to spend in USDC
                
            Returns:
                Trade execution result
            """
            try:
                result = await toolkit_instance.dex_client.buy_asset(ticker, amount)
                
                return {
                    "success": result.get("success", False),
                    "ticker": ticker,
                    "amount": amount,
                    "result": result
                }
            except DEXSimulatorError as e:
                log.error(f"DEX simulator error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "amount": amount
                }
            except Exception as e:
                log.error(f"Error buying asset: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "amount": amount
                }
        
        buy_asset.__name__ = "buy_asset"
        buy_asset.__doc__ = "Buy an asset on the DEX simulator using USDC"
        return buy_asset
    
    def sell_asset_tool(self):
        """Get tool for selling assets."""
        if not CAMEL_TOOLS_AVAILABLE:
            raise ImportError("CAMEL tools not installed")
        
        toolkit_instance = self
        
        async def sell_asset(
            ticker: Annotated[str, Field(description="Asset ticker symbol (e.g., BTC)")],
            amount: Annotated[float, Field(description="Amount to sell in asset units")]
        ) -> Dict[str, Any]:
            """
            Sell an asset on the DEX simulator.
            
            Args:
                ticker: Asset ticker symbol
                amount: Amount to sell in asset units
                
            Returns:
                Trade execution result
            """
            try:
                result = await toolkit_instance.dex_client.sell_asset(ticker, amount)
                
                return {
                    "success": result.get("success", False),
                    "ticker": ticker,
                    "amount": amount,
                    "result": result
                }
            except DEXSimulatorError as e:
                log.error(f"DEX simulator error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "amount": amount
                }
            except Exception as e:
                log.error(f"Error selling asset: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "amount": amount
                }
        
        sell_asset.__name__ = "sell_asset"
        sell_asset.__doc__ = "Sell an asset on the DEX simulator"
        return sell_asset
    
    def get_portfolio_status_tool(self):
        """Get tool for retrieving portfolio status."""
        if not CAMEL_TOOLS_AVAILABLE:
            raise ImportError("CAMEL tools not installed")
        
        toolkit_instance = self
        
        async def get_portfolio_status() -> Dict[str, Any]:
            """
            Get current portfolio status including balances and holdings.
            
            Returns:
                Portfolio status information
            """
            try:
                portfolio = await toolkit_instance.dex_client.get_portfolio_status()
                
                return {
                    "success": True,
                    "portfolio": portfolio
                }
            except DEXSimulatorError as e:
                log.error(f"DEX simulator error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "portfolio": {}
                }
            except Exception as e:
                log.error(f"Error getting portfolio status: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "portfolio": {}
                }
        
        get_portfolio_status.__name__ = "get_portfolio_status"
        get_portfolio_status.__doc__ = "Get current portfolio status including balances, holdings, and total value"
        return get_portfolio_status
    
    def get_ticker_price_tool(self):
        """Get tool for retrieving current ticker prices."""
        if not CAMEL_TOOLS_AVAILABLE:
            raise ImportError("CAMEL tools not installed")
        
        toolkit_instance = self
        
        async def get_ticker_price(
            ticker: Annotated[str, Field(description="Asset ticker symbol (e.g., BTC)")]
        ) -> Dict[str, Any]:
            """
            Get current price for a ticker.
            
            Args:
                ticker: Asset ticker symbol
                
            Returns:
                Current price information
            """
            try:
                prices = await toolkit_instance.dex_client.get_ticker_prices()
                price = prices.get(ticker, 0.0)
                
                return {
                    "success": True,
                    "ticker": ticker,
                    "price": price,
                    "currency": "USDC"
                }
            except DEXSimulatorError as e:
                log.error(f"DEX simulator error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "price": 0.0
                }
            except Exception as e:
                log.error(f"Error getting ticker price: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "ticker": ticker,
                    "price": 0.0
                }
        
        get_ticker_price.__name__ = "get_ticker_price"
        get_ticker_price.__doc__ = "Get current price for an asset ticker"
        return get_ticker_price
    
    def get_all_tools(self) -> List:
        """Get all tools in this toolkit."""
        base_tools = [
            self.buy_asset_tool(),
            self.sell_asset_tool(),
            self.get_portfolio_status_tool(),
            self.get_ticker_price_tool(),
        ]
        return [self._wrap_tool(func) for func in base_tools]

    @staticmethod
    def _wrap_tool(func):
        if not CAMEL_TOOLS_AVAILABLE or FunctionTool is None:
            raise ImportError("CAMEL function tools not installed")

        tool = FunctionTool(func)
        try:
            schema = dict(tool.get_openai_tool_schema())
        except Exception:
            schema = {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": func.__doc__ or func.__name__,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }

        function_schema = schema.setdefault("function", {})
        function_schema["name"] = func.__name__
        function_schema.setdefault("description", func.__doc__ or func.__name__)
        tool.openai_tool_schema = schema
        return tool

