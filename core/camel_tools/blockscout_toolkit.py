"""
Blockscout MCP Toolkit for CAMEL Agents.

Provides tools for querying on-chain data across 3,000+ EVM-compatible chains.
Reference: https://www.blog.blockscout.com/how-to-set-up-mcp-ai-onchain-data-block-explorer/
"""
from typing import Dict, Any, Annotated
from pydantic import Field
from core.logging import log
from core.config import settings
from core.blockscout_client import BlockscoutMCPClient, BlockscoutMCPError


# Global toolkit instance
_toolkit_instance: "BlockscoutMCPToolkit" = None


def get_blockscout_toolkit() -> "BlockscoutMCPToolkit":
    """Get or create the global Blockscout toolkit instance."""
    global _toolkit_instance
    if _toolkit_instance is None:
        _toolkit_instance = BlockscoutMCPToolkit()
    return _toolkit_instance


class BlockscoutMCPToolkit:
    """Toolkit for Blockscout MCP on-chain data queries."""
    
    def __init__(self):
        """Initialize the Blockscout MCP toolkit."""
        config = {
            "base_url": settings.blockscout_mcp_url,
            "timeout": 30.0,
            "retry_attempts": 3
        }
        self.blockscout_client = BlockscoutMCPClient(config)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the Blockscout client connection."""
        if not self._initialized:
            await self.blockscout_client.connect()
            self._initialized = True
            log.info("Blockscout MCP toolkit initialized")
    
    def get_list_chains_tool(self):
        """Get tool for listing supported EVM chains."""
        toolkit_instance = self
        
        async def list_chains() -> Dict[str, Any]:
            """
            List all EVM-compatible blockchain networks supported by Blockscout.
            
            Returns:
                List of supported chains with metadata
            """
            try:
                await toolkit_instance.initialize()
                chains = await toolkit_instance.blockscout_client.list_chains()
                return {
                    "success": True,
                    "chains": chains,
                    "total_chains": len(chains)
                }
            except BlockscoutMCPError as e:
                log.error(f"Blockscout MCP error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "chains": []
                }
            except Exception as e:
                log.error(f"Error listing chains: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "chains": []
                }
        
        list_chains.__name__ = "list_chains"
        list_chains.__doc__ = "List all EVM-compatible blockchain networks supported by Blockscout"
        return list_chains
    
    def get_wallet_balance_tool(self):
        """Get tool for querying wallet balances."""
        toolkit_instance = self
        
        async def get_wallet_balance(
            address: Annotated[str, Field(description="Wallet address (0x...)")],
            chain: Annotated[str, Field(description="Chain name (e.g., ethereum, polygon, arbitrum)", default="ethereum")] = "ethereum",
            token_address: Annotated[str, Field(description="Optional token contract address for token balance", default=None)] = None
        ) -> Dict[str, Any]:
            """
            Get wallet balance for an address on a specific chain.
            
            Args:
                address: Wallet address (0x...)
                chain: Chain name (default: ethereum)
                token_address: Optional token contract address for token balance
                
            Returns:
                Balance information including native token and token balances
            """
            try:
                await toolkit_instance.initialize()
                result = await toolkit_instance.blockscout_client.get_wallet_balance(
                    address=address,
                    chain=chain,
                    token_address=token_address
                )
                return {
                    "success": True,
                    "address": address,
                    "chain": chain,
                    **result
                }
            except BlockscoutMCPError as e:
                log.error(f"Blockscout MCP error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "address": address,
                    "chain": chain
                }
            except Exception as e:
                log.error(f"Error getting wallet balance: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "address": address,
                    "chain": chain
                }
        
        get_wallet_balance.__name__ = "get_wallet_balance"
        get_wallet_balance.__doc__ = "Get wallet balance for an address on a specific EVM chain"
        return get_wallet_balance
    
    def get_transaction_history_tool(self):
        """Get tool for querying transaction history."""
        toolkit_instance = self
        
        async def get_transaction_history(
            address: Annotated[str, Field(description="Wallet address (0x...)")],
            chain: Annotated[str, Field(description="Chain name (e.g., ethereum, polygon)", default="ethereum")] = "ethereum",
            limit: Annotated[int, Field(description="Number of transactions to return", default=50)] = 50,
            offset: Annotated[int, Field(description="Pagination offset", default=0)] = 0
        ) -> Dict[str, Any]:
            """
            Get transaction history for an address.
            
            Args:
                address: Wallet address
                chain: Chain name
                limit: Number of transactions to return (default: 50)
                offset: Pagination offset (default: 0)
                
            Returns:
                List of transactions with details
            """
            try:
                await toolkit_instance.initialize()
                transactions = await toolkit_instance.blockscout_client.get_transaction_history(
                    address=address,
                    chain=chain,
                    limit=limit,
                    offset=offset
                )
                return {
                    "success": True,
                    "address": address,
                    "chain": chain,
                    "transactions": transactions,
                    "count": len(transactions),
                    "limit": limit,
                    "offset": offset
                }
            except BlockscoutMCPError as e:
                log.error(f"Blockscout MCP error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "address": address,
                    "chain": chain,
                    "transactions": []
                }
            except Exception as e:
                log.error(f"Error getting transaction history: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "address": address,
                    "chain": chain,
                    "transactions": []
                }
        
        get_transaction_history.__name__ = "get_transaction_history"
        get_transaction_history.__doc__ = "Get transaction history for a wallet address on a specific chain"
        return get_transaction_history
    
    def get_contract_info_tool(self):
        """Get tool for querying smart contract information."""
        toolkit_instance = self
        
        async def get_contract_info(
            contract_address: Annotated[str, Field(description="Smart contract address (0x...)")],
            chain: Annotated[str, Field(description="Chain name (e.g., ethereum, arbitrum)", default="ethereum")] = "ethereum"
        ) -> Dict[str, Any]:
            """
            Get smart contract information including source code, ABI, and verification status.
            
            Args:
                contract_address: Contract address
                chain: Chain name
                
            Returns:
                Contract information including source code, ABI, compiler version, and verification status
            """
            try:
                await toolkit_instance.initialize()
                result = await toolkit_instance.blockscout_client.get_contract_info(
                    contract_address=contract_address,
                    chain=chain
                )
                return {
                    "success": True,
                    "contract_address": contract_address,
                    "chain": chain,
                    **result
                }
            except BlockscoutMCPError as e:
                log.error(f"Blockscout MCP error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "contract_address": contract_address,
                    "chain": chain
                }
            except Exception as e:
                log.error(f"Error getting contract info: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "contract_address": contract_address,
                    "chain": chain
                }
        
        get_contract_info.__name__ = "get_contract_info"
        get_contract_info.__doc__ = "Get smart contract information including source code and ABI"
        return get_contract_info
    
    def get_token_info_tool(self):
        """Get tool for querying token information."""
        toolkit_instance = self
        
        async def get_token_info(
            token_address: Annotated[str, Field(description="Token contract address (0x...)")],
            chain: Annotated[str, Field(description="Chain name (e.g., ethereum, polygon)", default="ethereum")] = "ethereum"
        ) -> Dict[str, Any]:
            """
            Get token information (ERC-20/ERC-721).
            
            Args:
                token_address: Token contract address
                chain: Chain name
                
            Returns:
                Token information including name, symbol, decimals, total supply, and type
            """
            try:
                await toolkit_instance.initialize()
                result = await toolkit_instance.blockscout_client.get_token_info(
                    token_address=token_address,
                    chain=chain
                )
                return {
                    "success": True,
                    "token_address": token_address,
                    "chain": chain,
                    **result
                }
            except BlockscoutMCPError as e:
                log.error(f"Blockscout MCP error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "token_address": token_address,
                    "chain": chain
                }
            except Exception as e:
                log.error(f"Error getting token info: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "token_address": token_address,
                    "chain": chain
                }
        
        get_token_info.__name__ = "get_token_info"
        get_token_info.__doc__ = "Get token information (ERC-20/ERC-721) including name, symbol, and supply"
        return get_token_info
    
    def get_transaction_details_tool(self):
        """Get tool for querying transaction details."""
        toolkit_instance = self
        
        async def get_transaction_details(
            tx_hash: Annotated[str, Field(description="Transaction hash")],
            chain: Annotated[str, Field(description="Chain name (e.g., ethereum, polygon)", default="ethereum")] = "ethereum"
        ) -> Dict[str, Any]:
            """
            Get detailed transaction information.
            
            Args:
                tx_hash: Transaction hash
                chain: Chain name
                
            Returns:
                Transaction details including from/to addresses, value, gas, status, and logs
            """
            try:
                await toolkit_instance.initialize()
                result = await toolkit_instance.blockscout_client.get_transaction_details(
                    tx_hash=tx_hash,
                    chain=chain
                )
                return {
                    "success": True,
                    "tx_hash": tx_hash,
                    "chain": chain,
                    **result
                }
            except BlockscoutMCPError as e:
                log.error(f"Blockscout MCP error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "tx_hash": tx_hash,
                    "chain": chain
                }
            except Exception as e:
                log.error(f"Error getting transaction details: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "tx_hash": tx_hash,
                    "chain": chain
                }
        
        get_transaction_details.__name__ = "get_transaction_details"
        get_transaction_details.__doc__ = "Get detailed transaction information by transaction hash"
        return get_transaction_details
    
    def get_compare_wallets_tool(self):
        """Get tool for comparing multiple wallets."""
        toolkit_instance = self
        
        async def compare_wallets(
            addresses: Annotated[str, Field(description="Comma-separated list of wallet addresses (0x...)")],
            chain: Annotated[str, Field(description="Chain name (e.g., ethereum, polygon)", default="ethereum")] = "ethereum"
        ) -> Dict[str, Any]:
            """
            Compare multiple wallets (balances, activity, etc.).
            
            Args:
                addresses: Comma-separated list of wallet addresses
                chain: Chain name
                
            Returns:
                Comparison data including balances and statistics
            """
            try:
                await toolkit_instance.initialize()
                address_list = [addr.strip() for addr in addresses.split(",")]
                result = await toolkit_instance.blockscout_client.compare_wallets(
                    addresses=address_list,
                    chain=chain
                )
                return {
                    "success": True,
                    **result
                }
            except BlockscoutMCPError as e:
                log.error(f"Blockscout MCP error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "addresses": addresses,
                    "chain": chain
                }
            except Exception as e:
                log.error(f"Error comparing wallets: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "addresses": addresses,
                    "chain": chain
                }
        
        compare_wallets.__name__ = "compare_wallets"
        compare_wallets.__doc__ = "Compare multiple wallets across balances and activity"
        return compare_wallets

