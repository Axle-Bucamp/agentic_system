"""Memory Agent - Maintains optimized history of trades, news, and performance."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agents.base_agent import BaseAgent
from core.config import settings
from core.logging import log
from core.memory.graph_memory import GraphMemoryManager
from core.models import (
    AgentMessage,
    AgentType,
    GraphMemoryNode,
    MessageType,
    NewsMemoryEntry,
    PerformanceMetrics,
    TradeAction,
    TradeMemoryEntry,
)


class MemoryAgent(BaseAgent):
    """Agent responsible for maintaining trading history and performance metrics."""

    def __init__(self, redis_client):
        super().__init__(AgentType.MEMORY, redis_client)
        self.trade_tape_limit = 1000
        self.ticker_trade_limit = 200
        self.signal_limit = 5000
        self.news_memory_limit = 500
        self.news_half_life_hours = 12
        self.graph_memory = GraphMemoryManager(redis_client)

    async def initialize(self):
        """Initialize memory subsystems."""
        log.info("Memory Agent initialized")
        await self._initialize_performance_tracking()

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages and store relevant data."""
        try:
            if message.message_type == MessageType.TRADE_EXECUTED:
                await self._record_trade(message.payload)
            elif message.message_type == MessageType.SIGNAL_GENERATED:
                await self._record_signal(message.payload)
            elif message.message_type == MessageType.NEWS_EVENT:
                await self._record_news_event(message.payload)
            elif message.message_type == MessageType.HUMAN_VALIDATION_RESPONSE:
                await self._record_user_feedback(message.payload)
        except Exception as exc:
            log.error("Memory Agent error processing %s: %s", message.message_type, exc)

        return None

    async def run_cycle(self):
        """Run periodic memory maintenance and analysis."""
        log.debug("Memory Agent running cycle...")

        try:
            await self._update_performance_metrics()
            await self._analyze_patterns()
            await self._refresh_news_weights()
            await self._cleanup_old_data()
        except Exception as exc:
            log.error("Memory Agent cycle error: %s", exc)

    async def _initialize_performance_tracking(self):
        metrics = await self.redis.get_json("memory:performance")
        if metrics:
            return

        initial_metrics = PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
            current_drawdown=0.0,
            roi=0.0,
        )
        await self.redis.set_json("memory:performance", initial_metrics.dict())

    async def _record_trade(self, trade_data: Dict[str, Any]):
        ticker = trade_data.get("ticker", "UNKNOWN")
        action_value = trade_data.get("action", "HOLD")
        action = self._normalize_action(action_value)
        quantity = self._safe_float(
            trade_data.get("quantity")
            or trade_data.get("amount")
            or trade_data.get("size")
        )
        price = self._safe_float(
            trade_data.get("executed_price")
            or trade_data.get("price")
            or trade_data.get("avg_price")
        )
        pnl = self._extract_pnl(trade_data)
        status = self._determine_trade_status(pnl)
        trade_timestamp = self._parse_timestamp(trade_data.get("timestamp"))
        trade_id = (
            trade_data.get("decision_id")
            or trade_data.get("trade_id")
            or trade_data.get("id")
            or str(uuid4())
        )

        entry = TradeMemoryEntry(
            trade_id=trade_id,
            ticker=ticker,
            action=action,
            quantity=quantity,
            price=price,
            pnl=pnl,
            status=status,
            metadata={"raw": trade_data},
            timestamp=trade_timestamp,
        )

        await self.redis.rpush("memory:trades", entry.json())
        await self.redis.ltrim("memory:trades", -self.trade_tape_limit, -1)

        await self.redis.rpush(f"memory:trades:{ticker}", entry.json())
        await self.redis.ltrim(f"memory:trades:{ticker}", -self.ticker_trade_limit, -1)

        await self._record_trade_in_graph(entry)
        log.info("Recorded trade: %s %s @ %.4f", ticker, action.value, price)

    async def _record_signal(self, signal_data: Dict[str, Any]):
        record = {
            "record_type": "signal",
            "ticker": signal_data.get("ticker"),
            "signal_type": signal_data.get("signal_type"),
            "data": signal_data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.redis.rpush("memory:signals", json.dumps(record))
        await self.redis.ltrim("memory:signals", -self.signal_limit, -1)

    async def _record_news_event(self, news_data: Dict[str, Any]):
        sentiment_score = self._safe_float(news_data.get("sentiment_score"))
        confidence = self._safe_float(news_data.get("confidence"), default=0.5)
        news_timestamp = self._parse_timestamp(news_data.get("timestamp"))
        ticker = news_data.get("ticker")
        summary = news_data.get("summary", news_data.get("title", ""))
        sources = news_data.get("sources") or [news_data.get("source", "unknown")]
        news_id = news_data.get("news_id") or hashlib.sha1(
            f"{ticker}:{summary}:{news_timestamp.isoformat()}".encode("utf-8")
        ).hexdigest()

        weight = self._compute_news_weight(news_timestamp, confidence)
        entry = NewsMemoryEntry(
            news_id=news_id,
            ticker=ticker,
            sentiment_score=sentiment_score,
            confidence=confidence,
            summary=summary,
            sources=sources,
            weight=weight,
            metadata={"raw": news_data},
            timestamp=news_timestamp,
        )

        await self.redis.rpush("memory:news", entry.json())
        await self.redis.ltrim("memory:news", -self.news_memory_limit, -1)

        if ticker:
            await self.redis.rpush(f"memory:news:{ticker}", entry.json())
            await self.redis.ltrim(f"memory:news:{ticker}", -self.news_memory_limit, -1)

        await self._record_news_in_graph(entry)
        await self._recalculate_news_sentiment()

    async def _record_user_feedback(self, feedback_data: Dict[str, Any]):
        user_id = feedback_data.get("user_id", "human")
        approved = feedback_data.get("approved", False)
        decision = feedback_data.get("decision") or {}
        ticker = decision.get("ticker")
        action = decision.get("action")
        content = feedback_data.get("feedback") or (
            f"{'Approved' if approved else 'Rejected'} trade {action} on {ticker}"
        )
        tags = [tag for tag in [ticker, action, "approval" if approved else "rejection"] if tag]

        await self.graph_memory.record_user_input(
            user_id=user_id,
            content=content,
            tags=tags,
            weight=1.2 if approved else 0.8,
            metadata={"decision": decision},
        )

    async def _update_performance_metrics(self):
        trades_raw = await self.redis.lrange("memory:trades", -self.trade_tape_limit, -1)
        if not trades_raw:
            return

        trades: List[TradeMemoryEntry] = []
        for raw in trades_raw:
            entry = self._parse_trade_entry(raw)
            if entry:
                trades.append(entry)

        if not trades:
            return

        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if (trade.pnl or 0) > 0)
        losing_trades = sum(1 for trade in trades if (trade.pnl or 0) < 0)
        total_pnl = sum(trade.pnl or 0 for trade in trades)
        win_rate = winning_trades / total_trades if total_trades else 0.0

        portfolio = await self.get_portfolio()
        total_value = (
            portfolio.get("total_value_usdc", settings.initial_capital)
            if portfolio
            else settings.initial_capital
        )
        roi = (total_value - settings.initial_capital) / settings.initial_capital

        risk_data = await self.redis.get_json("risk:portfolio") or {}
        current_drawdown = risk_data.get("current_drawdown", 0.0)
        max_drawdown = risk_data.get("max_drawdown", 0.0)

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
            roi=roi,
        )

        await self.redis.set_json("memory:performance", metrics.dict())
        log.info(
            "Performance updated: %s trades, win rate %.1f%%, ROI %.2f%%",
            total_trades,
            win_rate * 100,
            roi * 100,
        )

    def _calculate_sharpe_ratio(self, trades: List[TradeMemoryEntry]) -> Optional[float]:
        if len(trades) < 10:
            return None

        returns = [trade.pnl or 0.0 for trade in trades]
        if not any(returns):
            return None

        mean_return = sum(returns) / len(returns)
        variance = sum((ret - mean_return) ** 2 for ret in returns) / len(returns)
        std_dev = math.sqrt(variance)
        if std_dev == 0:
            return None

        return (mean_return / std_dev) * math.sqrt(365)

    async def _analyze_patterns(self):
        trades_raw = await self.redis.lrange("memory:trades", -200, -1)
        if not trades_raw:
            return

        ticker_performance: Dict[str, Dict[str, float]] = {}
        for raw in trades_raw:
            entry = self._parse_trade_entry(raw)
            if not entry:
                continue
            pnl = entry.pnl or 0.0
            perf = ticker_performance.setdefault(
                entry.ticker,
                {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0},
            )
            perf["trades"] += 1
            perf["total_pnl"] += pnl
            if pnl > 0:
                perf["wins"] += 1
            elif pnl < 0:
                perf["losses"] += 1

        await self.redis.set_json("memory:ticker_performance", ticker_performance, expire=3600)

        if ticker_performance:
            best_ticker = max(ticker_performance.items(), key=lambda item: item[1]["total_pnl"])
            worst_ticker = min(
                ticker_performance.items(), key=lambda item: item[1]["total_pnl"]
            )
            log.info(
                "Best performer: %s (PnL %.2f) | Worst performer: %s (PnL %.2f)",
                best_ticker[0],
                best_ticker[1]["total_pnl"],
                worst_ticker[0],
                worst_ticker[1]["total_pnl"],
            )

    async def _refresh_news_weights(self):
        await self._recalculate_news_sentiment()

    async def _cleanup_old_data(self):
        # FIFO retention keeps the tape compact; nothing extra required today.
        log.debug("Memory cleanup pass completed")

    async def get_ticker_history(self, ticker: str, limit: int = 50) -> List[Dict[str, Any]]:
        history_raw = await self.redis.lrange(f"memory:trades:{ticker}", -limit, -1)
        entries: List[Dict[str, Any]] = []
        for raw in history_raw:
            entry = self._parse_trade_entry(raw)
            if entry:
                entries.append(json.loads(entry.json()))
        return entries

    async def get_recency_weighted_news(
        self, ticker: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        key = f"memory:news:{ticker}" if ticker else "memory:news"
        news_raw = await self.redis.lrange(key, -limit, -1)
        entries: List[Dict[str, Any]] = []
        for raw in news_raw:
            entry = self._parse_news_entry(raw)
            if entry:
                entries.append(json.loads(entry.json()))
        return entries

    async def get_performance_metrics(self) -> Optional[PerformanceMetrics]:
        try:
            metrics_data = await self.redis.get_json("memory:performance")
            if metrics_data:
                return PerformanceMetrics(**metrics_data)
        except Exception as exc:
            log.error("Error getting performance metrics: %s", exc)
        return None

    async def get_insights_for_ticker(self, ticker: str) -> Dict[str, Any]:
        try:
            ticker_perf = await self.redis.get_json("memory:ticker_performance") or {}
            perf = ticker_perf.get(ticker)
            if not perf:
                return {"has_history": False, "message": f"No historical data for {ticker}"}

            win_rate = perf["wins"] / perf["trades"] if perf["trades"] else 0.0
            recommendation = "FAVORABLE"
            if win_rate <= 0.4:
                recommendation = "UNFAVORABLE"
            elif perf["total_pnl"] <= 0 or win_rate <= 0.6:
                recommendation = "NEUTRAL"

            return {
                "has_history": True,
                "total_trades": perf["trades"],
                "win_rate": win_rate,
                "total_pnl": perf["total_pnl"],
                "recommendation": recommendation,
            }
        except Exception as exc:
            log.error("Error getting insights for %s: %s", ticker, exc)
            return {"has_history": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _parse_trade_entry(self, raw: str) -> Optional[TradeMemoryEntry]:
        try:
            return TradeMemoryEntry.parse_raw(raw)
        except Exception:
            try:
                data = json.loads(raw)
            except Exception:
                return None
            if "data" not in data:
                return None
            legacy = data["data"]
            return TradeMemoryEntry(
                trade_id=legacy.get("decision_id") or legacy.get("id") or str(uuid4()),
                ticker=legacy.get("ticker", "UNKNOWN"),
                action=self._normalize_action(legacy.get("action", "HOLD")),
                quantity=self._safe_float(legacy.get("quantity")),
                price=self._safe_float(legacy.get("executed_price") or legacy.get("price")),
                pnl=self._extract_pnl(legacy),
                status=self._determine_trade_status(self._extract_pnl(legacy)),
                metadata={"raw": legacy},
                timestamp=self._parse_timestamp(legacy.get("timestamp")),
            )

    def _parse_news_entry(self, raw: str) -> Optional[NewsMemoryEntry]:
        try:
            return NewsMemoryEntry.parse_raw(raw)
        except Exception:
            try:
                data = json.loads(raw)
            except Exception:
                return None
            if "sentiment_score" not in data:
                return None
            return NewsMemoryEntry(
                news_id=data.get("news_id", str(uuid4())),
                ticker=data.get("ticker"),
                sentiment_score=self._safe_float(data.get("sentiment_score")),
                confidence=self._safe_float(data.get("confidence"), default=0.5),
                summary=data.get("summary", ""),
                sources=data.get("sources", []),
                weight=self._safe_float(data.get("weight"), default=0.5),
                metadata=data.get("metadata", {}),
                timestamp=self._parse_timestamp(data.get("timestamp")),
            )

    def _normalize_action(self, action_value: Any) -> TradeAction:
        if isinstance(action_value, TradeAction):
            return action_value
        try:
            return TradeAction(str(action_value).upper())
        except Exception:
            return TradeAction.HOLD

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _extract_pnl(self, trade_data: Dict[str, Any]) -> Optional[float]:
        for key in ("pnl", "profit", "realized_pnl", "return"):
            val = trade_data.get(key)
            if val is not None:
                return self._safe_float(val)
        return None

    def _determine_trade_status(self, pnl: Optional[float]) -> str:
        if pnl is None:
            return "UNKNOWN"
        if pnl > 0:
            return "WIN"
        if pnl < 0:
            return "LOSS"
        return "BREAKEVEN"

    def _parse_timestamp(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if not value:
            return datetime.utcnow()
        text = str(value)
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return datetime.utcnow()

    def _compute_news_weight(self, timestamp: datetime, confidence: float) -> float:
        age_seconds = max((datetime.utcnow() - timestamp).total_seconds(), 0.0)
        half_life_seconds = max(self.news_half_life_hours, 1) * 3600
        recency_decay = 0.5 ** (age_seconds / half_life_seconds)
        return max(0.05, recency_decay * max(confidence, 0.1))

    async def _recalculate_news_sentiment(self):
        news_raw = await self.redis.lrange("memory:news", -self.news_memory_limit, -1)
        aggregates: Dict[str, Dict[str, float]] = {}
        for raw in news_raw:
            entry = self._parse_news_entry(raw)
            if not entry:
                continue
            key = entry.ticker or "__market__"
            aggregate = aggregates.setdefault(key, {"score": 0.0, "weight": 0.0})
            aggregate["score"] += entry.sentiment_score * entry.weight
            aggregate["weight"] += entry.weight

        for key, values in aggregates.items():
            weighted_score = (
                values["score"] / values["weight"] if values["weight"] else 0.0
            )
            await self.redis.set_json(
                f"memory:news:weighted:{key}",
                {
                    "ticker": None if key == "__market__" else key,
                    "weighted_score": weighted_score,
                    "total_weight": values["weight"],
                    "last_updated": datetime.utcnow().isoformat(),
                },
                expire=3600,
            )

    async def _record_trade_in_graph(self, entry: TradeMemoryEntry):
        trade_node_weight = max(abs(entry.pnl or 0), 1.0)
        trade_node = GraphMemoryNode(
            node_id=f"trade:{entry.trade_id}",
            label=f"{entry.action.value} {entry.ticker}",
            node_type="trade",
            weight=trade_node_weight,
            metadata={
                "ticker": entry.ticker,
                "action": entry.action.value,
                "quantity": entry.quantity,
                "price": entry.price,
                "pnl": entry.pnl,
                "status": entry.status,
            },
        )
        asset_node = GraphMemoryNode(
            node_id=f"asset:{entry.ticker}",
            label=entry.ticker,
            node_type="asset",
            weight=1.0,
        )
        outcome_node = GraphMemoryNode(
            node_id=f"trade_outcome:{entry.status.lower()}",
            label=entry.status,
            node_type="outcome",
            weight=1.0,
        )

        await self.graph_memory.upsert_node(trade_node)
        await self.graph_memory.upsert_node(asset_node)
        await self.graph_memory.upsert_node(outcome_node)

        await self.graph_memory.connect(trade_node.node_id, asset_node.node_id, "INVOLVES")
        await self.graph_memory.connect(trade_node.node_id, outcome_node.node_id, "RESULT")

    async def _record_news_in_graph(self, entry: NewsMemoryEntry):
        news_node = GraphMemoryNode(
            node_id=f"news:{entry.news_id}",
            label=entry.summary[:100],
            node_type="news",
            weight=entry.weight,
            metadata={
                "summary": entry.summary,
                "sentiment": entry.sentiment_score,
                "sources": entry.sources,
            },
        )
        await self.graph_memory.upsert_node(news_node)

        if entry.ticker:
            asset_node = GraphMemoryNode(
                node_id=f"asset:{entry.ticker}",
                label=entry.ticker,
                node_type="asset",
                weight=1.0,
            )
            await self.graph_memory.upsert_node(asset_node)
            await self.graph_memory.connect(
                news_node.node_id,
                asset_node.node_id,
                "MENTIONS",
                weight=entry.weight,
            )

