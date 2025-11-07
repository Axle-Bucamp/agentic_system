"""
Copy Trade Agent - Monitors on-chain transactions and copies successful traders.
"""
import httpx
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from web3 import Web3
from core.models import (
    AgentType, AgentMessage, MessageType, CopyTradeSignal,
    TradeAction, AgentSignal, SignalType
)
from core.config import settings
from core.logging import log
from agents.base_agent import BaseAgent


class CopyTradeAgent(BaseAgent):
    """Agent responsible for on-chain copy trading."""
    
    def __init__(self, redis_client):
        super().__init__(AgentType.COPYTRADE, redis_client)
        self.web3_clients: Dict[str, Web3] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self.tracked_wallets: List[Dict] = []
        self.wallet_signal_decay_hours = 24
        self.minimum_success_rate = 0.25
        self.max_tracked_wallets = 25
        
    async def initialize(self):
        """Initialize Web3 clients for different blockchains."""
        # Initialize Web3 clients
        try:
            self.web3_clients["ETH"] = Web3(Web3.HTTPProvider(settings.eth_rpc_url))
            self.web3_clients["BSC"] = Web3(Web3.HTTPProvider(settings.bsc_rpc_url))
            # Note: Solana uses different library, simplified here
            
            log.info(f"Copy Trade Agent initialized with {len(self.web3_clients)} blockchain connections")
        except Exception as e:
            log.error(f"Error initializing Web3 clients: {e}")
        
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Load tracked wallets from Redis
        await self._load_tracked_wallets()
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages."""
        # Handle wallet tracking requests
        if message.message_type == MessageType.MARKET_DATA_UPDATE:
            payload = message.payload
            if "add_wallet" in payload:
                await self._add_wallet_to_track(payload["add_wallet"])
        
        return None
    
    async def run_cycle(self):
        """Run periodic on-chain monitoring."""
        log.debug("Copy Trade Agent running cycle...")
        
        try:
            # Monitor tracked wallets
            for wallet_info in self.tracked_wallets:
                await self._monitor_wallet(wallet_info)
            
            # Discover new successful wallets
            await self._discover_successful_wallets()
            await self._prune_wallets()
            await self._publish_wallet_scores()
            
        except Exception as e:
            log.error(f"Copy Trade Agent cycle error: {e}")
    
    def get_cycle_interval(self) -> int:
        return settings.get_agent_cycle_seconds(self.agent_type)
    
    async def _load_tracked_wallets(self):
        """Load list of tracked wallets from Redis."""
        try:
            wallets = await self.redis.get_json("copytrade:tracked_wallets")
            if wallets:
                self.tracked_wallets = wallets
                log.info(f"Loaded {len(self.tracked_wallets)} tracked wallets")
            else:
                # Initialize with empty list
                self.tracked_wallets = []
                await self.redis.set_json("copytrade:tracked_wallets", [])
        except Exception as e:
            log.error(f"Error loading tracked wallets: {e}")
            self.tracked_wallets = []
    
    async def _add_wallet_to_track(self, wallet_data: Dict):
        """Add a new wallet to track."""
        try:
            wallet_address = wallet_data.get("address")
            blockchain = wallet_data.get("blockchain", "ETH")
            
            if not wallet_address:
                return
            
            # Check if already tracked
            if any(w["address"] == wallet_address for w in self.tracked_wallets):
                log.info(f"Wallet {wallet_address} already tracked")
                return
            
            wallet_info = {
                "address": wallet_address,
                "blockchain": blockchain,
                "added_at": datetime.utcnow().isoformat(),
                "performance": 0.0,
                "trade_count": 0,
                "success_rate": 0.4,
                "signal_score": 0.0
            }
            
            self.tracked_wallets.append(wallet_info)
            if len(self.tracked_wallets) > self.max_tracked_wallets:
                await self._prune_wallets(force=True)
            await self.redis.set_json("copytrade:tracked_wallets", self.tracked_wallets)
            
            log.info(f"Added wallet {wallet_address} on {blockchain} to tracking")
            
        except Exception as e:
            log.error(f"Error adding wallet to track: {e}")
    
    async def _monitor_wallet(self, wallet_info: Dict):
        """Monitor a wallet for new transactions."""
        try:
            address = wallet_info["address"]
            blockchain = wallet_info["blockchain"]
            
            # Get recent transactions
            transactions = await self._get_wallet_transactions(address, blockchain)
            
            if not transactions:
                wallet_info["signal_score"] = wallet_info.get("signal_score", 0.0) * 0.9
                return
            
            # Analyze transactions for trading activity
            signal_emitted = False
            for tx in transactions[:5]:  # Check last 5 transactions
                trade_signal = await self._analyze_transaction(tx, wallet_info)
                
                if trade_signal:
                    signal_emitted = True
                    wallet_info["last_signal_at"] = datetime.utcnow().isoformat()
                    wallet_info["trade_count"] = wallet_info.get("trade_count", 0) + 1
                    wallet_info["success_rate"] = min(
                        0.95,
                        wallet_info.get("success_rate", 0.4) * 0.7 + 0.3,
                    )
                    wallet_info["signal_score"] = wallet_info.get("signal_score", 0.0) * 0.7 + 1.0
                    # Send copy trade signal
                    await self._send_copy_trade_signal(trade_signal)
            
            if not signal_emitted:
                wallet_info["success_rate"] = max(
                    0.05, wallet_info.get("success_rate", 0.4) * 0.95
                )
                wallet_info["signal_score"] = wallet_info.get("signal_score", 0.0) * 0.9

            # Update wallet performance
            await self._update_wallet_performance(wallet_info)
            await self.redis.set_json("copytrade:tracked_wallets", self.tracked_wallets)
            
        except Exception as e:
            log.error(f"Error monitoring wallet {wallet_info['address']}: {e}")
    
    async def _get_wallet_transactions(self, address: str, blockchain: str) -> List[Dict]:
        """Get recent transactions for a wallet."""
        try:
            # Check cache first
            cache_key = f"copytrade:txs:{blockchain}:{address}"
            cached_txs = await self.redis.get_json(cache_key)
            
            if cached_txs:
                cache_time = datetime.fromisoformat(cached_txs.get("timestamp", "2000-01-01"))
                if datetime.utcnow() - cache_time < timedelta(minutes=2):
                    return cached_txs.get("transactions", [])
            
            transactions = []
            
            if blockchain == "ETH":
                # Use Etherscan API (free tier)
                url = "https://api.etherscan.io/api"
                params = {
                    "module": "account",
                    "action": "txlist",
                    "address": address,
                    "startblock": 0,
                    "endblock": 99999999,
                    "page": 1,
                    "offset": 10,
                    "sort": "desc"
                }
                
                response = await self.http_client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "1":
                        transactions = data.get("result", [])
            
            elif blockchain == "BSC":
                # Use BSCScan API (similar to Etherscan)
                url = "https://api.bscscan.com/api"
                params = {
                    "module": "account",
                    "action": "txlist",
                    "address": address,
                    "startblock": 0,
                    "endblock": 99999999,
                    "page": 1,
                    "offset": 10,
                    "sort": "desc"
                }
                
                response = await self.http_client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "1":
                        transactions = data.get("result", [])
            
            # Cache transactions
            await self.redis.set_json(
                cache_key,
                {
                    "transactions": transactions,
                    "timestamp": datetime.utcnow().isoformat()
                },
                expire=120  # 2 minutes
            )
            
            return transactions
            
        except Exception as e:
            log.error(f"Error getting wallet transactions: {e}")
            return []
    
    async def _analyze_transaction(self, tx: Dict, wallet_info: Dict) -> Optional[CopyTradeSignal]:
        """Analyze a transaction to determine if it's a tradeable signal."""
        try:
            # This is a simplified analysis
            # In production, decode contract calls to identify DEX swaps
            
            # Check if transaction is recent (last 10 minutes)
            tx_timestamp = int(tx.get("timeStamp", 0))
            tx_time = datetime.fromtimestamp(tx_timestamp)
            
            if datetime.utcnow() - tx_time > timedelta(minutes=10):
                return None  # Too old
            
            # Check if we've already processed this transaction
            tx_hash = tx.get("hash")
            processed_key = f"copytrade:processed:{tx_hash}"
            
            if await self.redis.exists(processed_key):
                return None
            
            # Mark as processed
            await self.redis.set(processed_key, "1", expire=86400)  # 24 hours
            
            # Simplified: Assume transaction is a swap if value > 0
            # In production, decode the input data to identify DEX interactions
            value = int(tx.get("value", 0))
            
            if value == 0:
                return None  # Likely not a simple transfer
            
            # Determine action (simplified)
            # In production, decode contract calls to determine buy/sell
            action = TradeAction.BUY  # Placeholder
            
            # Map to our supported tickers (simplified)
            # In production, decode the swap details
            ticker = "ETH"  # Placeholder
            
            # Calculate confidence based on wallet performance
            success_rate = wallet_info.get("success_rate", 0.4)
            confidence = max(0.1, min(0.95, success_rate))
            
            signal = CopyTradeSignal(
                wallet_address=wallet_info["address"],
                blockchain=wallet_info["blockchain"],
                ticker=ticker,
                action=action,
                amount=float(value) / 1e18,  # Convert from wei
                price=0.0,  # Would need to fetch from DEX
                wallet_performance=wallet_info.get("performance", 0),
                confidence=confidence,
                timestamp=datetime.utcnow()
            )
            
            return signal
            
        except Exception as e:
            log.error(f"Error analyzing transaction: {e}")
            return None
    
    async def _send_copy_trade_signal(self, signal: CopyTradeSignal):
        """Send copy trade signal to orchestrator."""
        try:
            agent_signal = AgentSignal(
                agent_type=self.agent_type,
                signal_type=SignalType.COPY_TRADE,
                ticker=signal.ticker,
                action=signal.action,
                confidence=signal.confidence,
                data={
                    "wallet_address": signal.wallet_address,
                    "blockchain": signal.blockchain,
                    "amount": signal.amount,
                    "wallet_performance": signal.wallet_performance
                },
                reasoning=f"Copy trade from wallet {signal.wallet_address[:8]}... on {signal.blockchain}. "
                         f"Wallet performance: {signal.wallet_performance:.1%}, confidence: {signal.confidence:.2f}"
            )
            
            await self.send_signal(agent_signal.dict())
            log.info(f"Sent copy trade signal: {signal.ticker} {signal.action.value} from {signal.wallet_address[:8]}...")
            await self._cache_last_signal(signal)
            
        except Exception as e:
            log.error(f"Error sending copy trade signal: {e}")

    async def _cache_last_signal(self, signal: CopyTradeSignal):
        payload = {
            "wallet_address": signal.wallet_address,
            "blockchain": signal.blockchain,
            "amount": signal.amount,
            "confidence": signal.confidence,
            "timestamp": signal.timestamp.isoformat() if isinstance(signal.timestamp, datetime) else signal.timestamp,
        }
        await self.redis.set_json(
            f"copytrade:last_signal:{signal.ticker}",
            payload,
            expire=600,
        )
    
    async def _update_wallet_performance(self, wallet_info: Dict):
        """Update performance metrics for a tracked wallet."""
        try:
            # This is simplified - in production, track actual trade outcomes
            address = wallet_info["address"]
            
            # Get historical performance from cache or calculate
            perf_key = f"copytrade:performance:{address}"
            performance = await self.redis.get_json(perf_key)
            
            if not performance:
                performance = {
                    "total_trades": 0,
                    "successful_trades": 0,
                    "total_pnl": 0.0,
                    "success_rate": wallet_info.get("success_rate", 0.4),
                    "signal_score": wallet_info.get("signal_score", 0.0),
                }
            else:
                performance["success_rate"] = wallet_info.get("success_rate", performance.get("success_rate", 0.4))
                performance["signal_score"] = wallet_info.get("signal_score", performance.get("signal_score", 0.0))
            performance["total_trades"] = wallet_info.get("trade_count", performance.get("total_trades", 0))
            performance["total_pnl"] = wallet_info.get("performance", performance.get("total_pnl", 0.0))
            wallet_info["performance"] = performance.get("total_pnl", 0.0)
            await self.redis.set_json(perf_key, performance)
            
        except Exception as e:
            log.error(f"Error updating wallet performance: {e}")

    async def _prune_wallets(self, force: bool = False):
        try:
            now = datetime.utcnow()
            retained: List[Dict] = []
            updated = False
            for wallet in self.tracked_wallets:
                last_signal_at = wallet.get("last_signal_at")
                hours_since_signal = (
                    (now - self._parse_timestamp(last_signal_at)).total_seconds() / 3600
                    if last_signal_at
                    else float("inf")
                )
                success_rate = wallet.get("success_rate", 0.0)
                should_remove = force or (
                    success_rate < self.minimum_success_rate
                    and wallet.get("trade_count", 0) >= 3
                )
                if hours_since_signal > self.wallet_signal_decay_hours * 3:
                    should_remove = True
                if should_remove:
                    log.info(
                        "Pruning wallet %s (success_rate %.2f, inactivity %.1fh)",
                        wallet["address"][:8] + "...",
                        success_rate,
                        hours_since_signal,
                    )
                    updated = True
                    continue
                wallet["signal_score"] = wallet.get("signal_score", 0.0) * 0.9
                retained.append(wallet)

            if updated or force:
                self.tracked_wallets = retained[: self.max_tracked_wallets]
                await self.redis.set_json("copytrade:tracked_wallets", self.tracked_wallets)
        except Exception as e:
            log.error(f"Error pruning wallets: {e}")

    async def _publish_wallet_scores(self):
        try:
            summary = {
                wallet["address"]: {
                    "success_rate": wallet.get("success_rate", 0.0),
                    "trade_count": wallet.get("trade_count", 0),
                    "signal_score": round(wallet.get("signal_score", 0.0), 3),
                    "last_signal_at": wallet.get("last_signal_at"),
                }
                for wallet in self.tracked_wallets
            }
            await self.redis.set_json("copytrade:wallet_scores", summary, expire=600)
        except Exception as e:
            log.error(f"Error publishing wallet scores: {e}")

    def _parse_timestamp(self, value: Optional[str]) -> datetime:
        if not value:
            return datetime.utcnow() - timedelta(hours=self.wallet_signal_decay_hours * 4)
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.utcnow() - timedelta(hours=self.wallet_signal_decay_hours * 4)
    
    async def _discover_successful_wallets(self):
        """Discover new successful wallets to track."""
        try:
            # This is a placeholder for wallet discovery logic
            # In production, use on-chain analytics platforms or APIs
            # to find wallets with high success rates
            
            log.debug("Wallet discovery not yet implemented")
            
        except Exception as e:
            log.error(f"Error discovering wallets: {e}")

