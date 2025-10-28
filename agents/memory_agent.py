"""
Memory Agent - Maintains optimized history of trades, predictions, and performance.
"""
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import json
from core.models import (
    AgentType, AgentMessage, MessageType, MemoryRecord,
    PerformanceMetrics, AgentSignal, SignalType
)
from core.config import settings
from core.logging import log
from agents.base_agent import BaseAgent


class MemoryAgent(BaseAgent):
    """Agent responsible for maintaining trading history and performance metrics."""
    
    def __init__(self, redis_client):
        super().__init__(AgentType.MEMORY, redis_client)
        
    async def initialize(self):
        """Initialize memory agent."""
        log.info("Memory Agent initialized")
        
        # Initialize performance tracking
        await self._initialize_performance_tracking()
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages and store relevant data."""
        try:
            if message.message_type == MessageType.TRADE_EXECUTED:
                await self._record_trade(message.payload)
            elif message.message_type == MessageType.SIGNAL_GENERATED:
                await self._record_signal(message.payload)
            
        except Exception as e:
            log.error(f"Memory Agent error processing message: {e}")
        
        return None
    
    async def run_cycle(self):
        """Run periodic memory maintenance and analysis."""
        log.debug("Memory Agent running cycle...")
        
        try:
            # Update performance metrics
            await self._update_performance_metrics()
            
            # Analyze recent patterns
            await self._analyze_patterns()
            
            # Clean old data
            await self._cleanup_old_data()
            
        except Exception as e:
            log.error(f"Memory Agent cycle error: {e}")
    
    def get_cycle_interval(self) -> int:
        """Run every 10 minutes."""
        return 600
    
    async def _initialize_performance_tracking(self):
        """Initialize performance tracking if not exists."""
        metrics = await self.redis.get_json("memory:performance")
        
        if not metrics:
            initial_metrics = PerformanceMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                sharpe_ratio=None,
                max_drawdown=0.0,
                current_drawdown=0.0,
                roi=0.0
            )
            await self.redis.set_json("memory:performance", initial_metrics.dict())
    
    async def _record_trade(self, trade_data: Dict):
        """Record a trade execution in memory."""
        try:
            # Create memory record
            record = MemoryRecord(
                record_type="trade",
                ticker=trade_data.get("ticker"),
                data=trade_data,
                timestamp=datetime.utcnow()
            )
            
            # Store in Redis list (keep last 1000 trades)
            await self.redis.lpush(
                "memory:trades",
                json.dumps(record.dict(), default=str)
            )
            await self.redis.ltrim("memory:trades", 0, 999)
            
            # Store in hash for quick ticker lookup
            ticker = trade_data.get("ticker")
            if ticker:
                await self.redis.lpush(
                    f"memory:trades:{ticker}",
                    json.dumps(record.dict(), default=str)
                )
                await self.redis.ltrim(f"memory:trades:{ticker}", 0, 99)
            
            log.info(f"Recorded trade: {ticker} {trade_data.get('action')}")
            
        except Exception as e:
            log.error(f"Error recording trade: {e}")
    
    async def _record_signal(self, signal_data: Dict):
        """Record an agent signal in memory."""
        try:
            record = MemoryRecord(
                record_type="signal",
                ticker=signal_data.get("ticker"),
                data=signal_data,
                timestamp=datetime.utcnow()
            )
            
            # Store signals (keep last 5000)
            await self.redis.lpush(
                "memory:signals",
                json.dumps(record.dict(), default=str)
            )
            await self.redis.ltrim("memory:signals", 0, 4999)
            
        except Exception as e:
            log.error(f"Error recording signal: {e}")
    
    async def _update_performance_metrics(self):
        """Update overall performance metrics."""
        try:
            # Get recent trades
            trades_raw = await self.redis.lrange("memory:trades", 0, -1)
            if not trades_raw:
                return
            
            trades = [json.loads(t) for t in trades_raw]
            
            # Calculate metrics
            total_trades = len(trades)
            winning_trades = 0
            losing_trades = 0
            total_pnl = 0.0
            
            for trade in trades:
                trade_data = trade.get("data", {})
                pnl = trade_data.get("pnl", 0)
                
                total_pnl += pnl
                if pnl > 0:
                    winning_trades += 1
                elif pnl < 0:
                    losing_trades += 1
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Get portfolio for ROI calculation
            portfolio = await self.get_portfolio()
            total_value = portfolio.get("total_value_usdc", settings.initial_capital) if portfolio else settings.initial_capital
            roi = (total_value - settings.initial_capital) / settings.initial_capital
            
            # Get drawdown from risk agent
            risk_data = await self.redis.get_json("risk:portfolio")
            current_drawdown = risk_data.get("current_drawdown", 0) if risk_data else 0
            max_drawdown = risk_data.get("max_drawdown", 0) if risk_data else 0
            
            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = self._calculate_sharpe_ratio(trades)
            
            metrics = PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                roi=roi
            )
            
            await self.redis.set_json("memory:performance", metrics.dict())
            log.info(f"Performance updated: {total_trades} trades, {win_rate:.1%} win rate, {roi:.1%} ROI")
            
        except Exception as e:
            log.error(f"Error updating performance metrics: {e}")
    
    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> Optional[float]:
        """Calculate Sharpe ratio from trade history."""
        if len(trades) < 10:
            return None
        
        try:
            returns = [t.get("data", {}).get("pnl", 0) for t in trades]
            
            if not returns:
                return None
            
            mean_return = sum(returns) / len(returns)
            
            # Calculate standard deviation
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5
            
            if std_dev == 0:
                return None
            
            # Annualized Sharpe ratio (assuming daily trades)
            sharpe = (mean_return / std_dev) * (365 ** 0.5)
            
            return sharpe
            
        except Exception as e:
            log.error(f"Error calculating Sharpe ratio: {e}")
            return None
    
    async def _analyze_patterns(self):
        """Analyze recent trading patterns and provide insights."""
        try:
            # Get recent trades
            trades_raw = await self.redis.lrange("memory:trades", 0, 99)
            if not trades_raw:
                return
            
            trades = [json.loads(t) for t in trades_raw]
            
            # Analyze per-ticker performance
            ticker_performance = {}
            
            for trade in trades:
                trade_data = trade.get("data", {})
                ticker = trade_data.get("ticker")
                pnl = trade_data.get("pnl", 0)
                
                if ticker:
                    if ticker not in ticker_performance:
                        ticker_performance[ticker] = {
                            "trades": 0,
                            "wins": 0,
                            "losses": 0,
                            "total_pnl": 0.0
                        }
                    
                    ticker_performance[ticker]["trades"] += 1
                    ticker_performance[ticker]["total_pnl"] += pnl
                    
                    if pnl > 0:
                        ticker_performance[ticker]["wins"] += 1
                    elif pnl < 0:
                        ticker_performance[ticker]["losses"] += 1
            
            # Store ticker performance
            await self.redis.set_json("memory:ticker_performance", ticker_performance, expire=3600)
            
            # Identify best and worst performers
            if ticker_performance:
                best_ticker = max(ticker_performance.items(), key=lambda x: x[1]["total_pnl"])
                worst_ticker = min(ticker_performance.items(), key=lambda x: x[1]["total_pnl"])
                
                log.info(f"Best performer: {best_ticker[0]} (PnL: {best_ticker[1]['total_pnl']:.2f})")
                log.info(f"Worst performer: {worst_ticker[0]} (PnL: {worst_ticker[1]['total_pnl']:.2f})")
            
        except Exception as e:
            log.error(f"Error analyzing patterns: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old data to optimize memory usage."""
        try:
            # Keep only last 30 days of detailed data
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            
            # This is a simplified cleanup - in production, use a proper database
            log.debug("Memory cleanup completed")
            
        except Exception as e:
            log.error(f"Error cleaning up old data: {e}")
    
    async def get_ticker_history(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Get trade history for a specific ticker."""
        try:
            trades_raw = await self.redis.lrange(f"memory:trades:{ticker}", 0, limit - 1)
            return [json.loads(t) for t in trades_raw]
        except Exception as e:
            log.error(f"Error getting ticker history: {e}")
            return []
    
    async def get_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics."""
        try:
            metrics_data = await self.redis.get_json("memory:performance")
            if metrics_data:
                return PerformanceMetrics(**metrics_data)
        except Exception as e:
            log.error(f"Error getting performance metrics: {e}")
        return None
    
    async def get_insights_for_ticker(self, ticker: str) -> Dict:
        """Get historical insights for a ticker to inform future decisions."""
        try:
            # Get ticker performance
            ticker_perf = await self.redis.get_json("memory:ticker_performance")
            
            if not ticker_perf or ticker not in ticker_perf:
                return {
                    "has_history": False,
                    "message": f"No historical data for {ticker}"
                }
            
            perf = ticker_perf[ticker]
            win_rate = perf["wins"] / perf["trades"] if perf["trades"] > 0 else 0
            
            insight = {
                "has_history": True,
                "total_trades": perf["trades"],
                "win_rate": win_rate,
                "total_pnl": perf["total_pnl"],
                "recommendation": "FAVORABLE" if win_rate > 0.6 and perf["total_pnl"] > 0 else "NEUTRAL" if win_rate > 0.4 else "UNFAVORABLE"
            }
            
            return insight
            
        except Exception as e:
            log.error(f"Error getting insights for {ticker}: {e}")
            return {"has_history": False, "error": str(e)}

