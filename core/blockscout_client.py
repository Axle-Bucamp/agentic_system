"""
Blockscout MCP Client for on-chain data queries.

Provides access to blockchain data across 3,000+ EVM-compatible chains via Blockscout MCP.
Reference: https://www.blog.blockscout.com/how-to-set-up-mcp-ai-onchain-data-block-explorer/
"""
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import httpx
from core.logging import log
from core.config import settings


class BlockscoutMCPError(Exception):
    """Base exception for Blockscout MCP operations."""
    pass


class BlockscoutMCPClient:
    """
    Client for interacting with Blockscout MCP for on-chain data.
    
    Supports querying:
    - Transaction data and history
    - Wallet balances and token holdings
    - Smart contract information and source code
    - Network statistics and chain data
    - Cross-chain analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Blockscout MCP client.
        
        Args:
            config: Optional configuration dict with:
                - base_url: Blockscout MCP endpoint (default: http://blockscout-mcp:8080 or https://mcp.blockscout.com/mcp)
                - timeout: Request timeout in seconds (default: 30)
                - retry_attempts: Number of retry attempts (default: 3)
        """
        config = config or {}
        # Use proxy if available, otherwise direct endpoint
        self.base_url = config.get(
            "base_url",
            settings.blockscout_mcp_url or "https://mcp.blockscout.com/mcp"
        )
        self.timeout = config.get("timeout", 30.0)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        
        # Cache for frequently accessed data
        self.cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        self.default_cache_ttl = timedelta(minutes=5)
    
    async def connect(self) -> None:
        """Initialize the HTTP client."""
        try:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "AgenticTradingSystem/1.0.0"
            }
            
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
                follow_redirects=True
            )
            
            log.info(f"Blockscout MCP client connected to {self.base_url}")
            
        except Exception as e:
            log.error(f"Failed to initialize Blockscout MCP client: {e}")
            raise BlockscoutMCPError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
            log.info("Blockscout MCP client disconnected")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to Blockscout MCP.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            json_data: JSON body for POST requests
            
        Returns:
            Response data as dictionary
        """
        if not self.client:
            await self.connect()
        
        cache_key = f"{method}:{endpoint}:{str(params)}"
        
        # Check cache
        if cache_key in self.cache:
            if datetime.now() < self.cache_ttl.get(cache_key, datetime.min):
                log.debug(f"Cache hit for {endpoint}")
                return self.cache[cache_key]
        
        for attempt in range(self.retry_attempts):
            try:
                response = await self.client.request(
                    method=method,
                    url=endpoint,
                    params=params,
                    json=json_data
                )
                response.raise_for_status()
                data = response.json()
                
                # Cache the response
                self.cache[cache_key] = data
                self.cache_ttl[cache_key] = datetime.now() + self.default_cache_ttl
                
                return data
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < self.retry_attempts - 1:
                    log.warning(f"Server error {e.response.status_code}, retrying... ({attempt + 1}/{self.retry_attempts})")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise BlockscoutMCPError(f"HTTP {e.response.status_code}: {e.response.text}")
            except httpx.RequestError as e:
                if attempt < self.retry_attempts - 1:
                    log.warning(f"Request error, retrying... ({attempt + 1}/{self.retry_attempts}): {e}")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise BlockscoutMCPError(f"Request failed: {e}")
            except Exception as e:
                raise BlockscoutMCPError(f"Unexpected error: {e}")
        
        raise BlockscoutMCPError(f"Failed after {self.retry_attempts} attempts")
    
    async def list_chains(self) -> List[Dict[str, Any]]:
        """
        List all supported EVM-compatible chains.
        
        Returns:
            List of chain information dictionaries
        """
        try:
            # MCP call to list chains
            response = await self._make_request(
                "POST",
                "/mcp",
                json_data={
                    "method": "tools/call",
                    "params": {
                        "name": "list_chains",
                        "arguments": {}
                    }
                }
            )
            return response.get("result", {}).get("chains", [])
        except Exception as e:
            log.error(f"Error listing chains: {e}")
            raise BlockscoutMCPError(f"Failed to list chains: {e}")
    
    async def get_wallet_balance(
        self,
        address: str,
        chain: str = "ethereum",
        token_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get wallet balance for an address.
        
        Args:
            address: Wallet address (0x...)
            chain: Chain name (default: ethereum)
            token_address: Optional token contract address for token balance
            
        Returns:
            Balance information
        """
        try:
            response = await self._make_request(
                "POST",
                "/mcp",
                json_data={
                    "method": "tools/call",
                    "params": {
                        "name": "get_balance",
                        "arguments": {
                            "address": address,
                            "chain": chain,
                            "token_address": token_address
                        }
                    }
                }
            )
            return response.get("result", {})
        except Exception as e:
            log.error(f"Error getting wallet balance: {e}")
            raise BlockscoutMCPError(f"Failed to get wallet balance: {e}")
    
    async def get_transaction_history(
        self,
        address: str,
        chain: str = "ethereum",
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history for an address.
        
        Args:
            address: Wallet address
            chain: Chain name
            limit: Number of transactions to return
            offset: Pagination offset
            
        Returns:
            List of transaction dictionaries
        """
        try:
            response = await self._make_request(
                "POST",
                "/mcp",
                json_data={
                    "method": "tools/call",
                    "params": {
                        "name": "get_transactions",
                        "arguments": {
                            "address": address,
                            "chain": chain,
                            "limit": limit,
                            "offset": offset
                        }
                    }
                }
            )
            return response.get("result", {}).get("transactions", [])
        except Exception as e:
            log.error(f"Error getting transaction history: {e}")
            raise BlockscoutMCPError(f"Failed to get transaction history: {e}")
    
    async def get_contract_info(
        self,
        contract_address: str,
        chain: str = "ethereum"
    ) -> Dict[str, Any]:
        """
        Get smart contract information including source code and ABI.
        
        Args:
            contract_address: Contract address
            chain: Chain name
            
        Returns:
            Contract information including source code, ABI, and verification status
        """
        try:
            response = await self._make_request(
                "POST",
                "/mcp",
                json_data={
                    "method": "tools/call",
                    "params": {
                        "name": "get_contract",
                        "arguments": {
                            "address": contract_address,
                            "chain": chain
                        }
                    }
                }
            )
            return response.get("result", {})
        except Exception as e:
            log.error(f"Error getting contract info: {e}")
            raise BlockscoutMCPError(f"Failed to get contract info: {e}")
    
    async def get_token_info(
        self,
        token_address: str,
        chain: str = "ethereum"
    ) -> Dict[str, Any]:
        """
        Get token information (ERC-20/ERC-721).
        
        Args:
            token_address: Token contract address
            chain: Chain name
            
        Returns:
            Token information including name, symbol, decimals, total supply
        """
        try:
            response = await self._make_request(
                "POST",
                "/mcp",
                json_data={
                    "method": "tools/call",
                    "params": {
                        "name": "get_token",
                        "arguments": {
                            "address": token_address,
                            "chain": chain
                        }
                    }
                }
            )
            return response.get("result", {})
        except Exception as e:
            log.error(f"Error getting token info: {e}")
            raise BlockscoutMCPError(f"Failed to get token info: {e}")
    
    async def get_transaction_details(
        self,
        tx_hash: str,
        chain: str = "ethereum"
    ) -> Dict[str, Any]:
        """
        Get detailed transaction information.
        
        Args:
            tx_hash: Transaction hash
            chain: Chain name
            
        Returns:
            Transaction details
        """
        try:
            response = await self._make_request(
                "POST",
                "/mcp",
                json_data={
                    "method": "tools/call",
                    "params": {
                        "name": "get_transaction",
                        "arguments": {
                            "tx_hash": tx_hash,
                            "chain": chain
                        }
                    }
                }
            )
            return response.get("result", {})
        except Exception as e:
            log.error(f"Error getting transaction details: {e}")
            raise BlockscoutMCPError(f"Failed to get transaction details: {e}")
    
    async def compare_wallets(
        self,
        addresses: List[str],
        chain: str = "ethereum"
    ) -> Dict[str, Any]:
        """
        Compare multiple wallets (balances, activity, etc.).
        
        Args:
            addresses: List of wallet addresses
            chain: Chain name
            
        Returns:
            Comparison data
        """
        try:
            # Get balances for all addresses
            balances = await asyncio.gather(*[
                self.get_wallet_balance(addr, chain) for addr in addresses
            ])
            
            return {
                "addresses": addresses,
                "chain": chain,
                "balances": balances,
                "comparison": {
                    "total_addresses": len(addresses),
                    "total_balance": sum(
                        float(b.get("balance", 0)) for b in balances
                    )
                }
            }
        except Exception as e:
            log.error(f"Error comparing wallets: {e}")
            raise BlockscoutMCPError(f"Failed to compare wallets: {e}")

