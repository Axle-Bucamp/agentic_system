"""Memory Agent - Maintains optimized history of trades, news, and performance."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agents.base_agent import BaseAgent
from core import asset_registry
from core.config import settings
from core.logging import log
from core.memory.graph_memory import GraphMemoryManager
from core.memory.neo4j_client import get_neo4j_client
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
from core.pipelines import MemoryPruningPipeline, WeightReviewPipeline

PENDING_REWARD_KEY = "memory:pending_trade_rewards"


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
        self.neo4j_client = None
        self.auto_enhance_ttl_seconds = 900
        self.camel_memory = None
        self.reward_history_limit = 500
        self.pruning_pipeline: Optional[MemoryPruningPipeline] = None
        self.weight_review_pipeline: Optional[WeightReviewPipeline] = None
        self._last_review_run: datetime = datetime.min

    async def initialize(self):
        """Initialize memory subsystems."""
        log.info("Memory Agent initialized")
        self.neo4j_client = await get_neo4j_client()
        if self.neo4j_client:
            self.graph_memory.neo4j_client = self.neo4j_client
            log.info("Neo4j graph mirror enabled")
        try:
            from core.memory.camel_memory_manager import CamelMemoryManager  # pylint: disable=import-error

            self.camel_memory = CamelMemoryManager(agent_id="memory_agent_auto")
            log.info("Qdrant vector memory enabled for auto-enhancement")
        except Exception as exc:  # pragma: no cover - optional dependency
            log.debug("Camel/Qdrant memory unavailable: %s", exc)
            self.camel_memory = None
        self.pruning_pipeline = MemoryPruningPipeline(self.redis, camel_memory=self.camel_memory)
        self.weight_review_pipeline = WeightReviewPipeline(self.redis, camel_memory=self.camel_memory)
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
            await self._evaluate_trade_rewards()
            await self._update_performance_metrics()
            await self._analyze_patterns()
            await self._refresh_news_weights()
            await self._cleanup_old_data()
            await self._auto_enhance_long_term_memory()
            await self._prune_memories_if_needed()
            await self._run_weight_review_if_due()
        except Exception as exc:
            log.error("Memory Agent cycle error: %s", exc)

    async def stop(self):
        if self.neo4j_client:
            await self.neo4j_client.close()
            self.neo4j_client = None
        await super().stop()

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

        snapshot = self._build_trade_snapshot(entry, trade_data)

        await self.redis.set_json(f"memory:trade:{trade_id}", snapshot)

        await self.redis.rpush("memory:trades", json.dumps({"trade_id": trade_id}))
        await self.redis.ltrim("memory:trades", -self.trade_tape_limit, -1)

        await self.redis.rpush(f"memory:trades:{ticker}", json.dumps({"trade_id": trade_id}))
        await self.redis.ltrim(f"memory:trades:{ticker}", -self.ticker_trade_limit, -1)

        await self._record_trade_in_graph(entry)
        log.info("Recorded trade: %s %s @ %.4f", ticker, action.value, price)

    def _build_trade_snapshot(self, entry: TradeMemoryEntry, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        decision = trade_data.get("decision")
        snapshot = {
            "trade_id": entry.trade_id,
            "ticker": entry.ticker,
            "action": entry.action.value,
            "quantity": entry.quantity,
            "executed_price": entry.price,
            "entry_price": entry.price,
            "pnl": entry.pnl,
            "status": entry.status,
            "confidence": trade_data.get("confidence")
            or (decision or {}).get("confidence"),
            "timestamp": entry.timestamp.isoformat(),
            "decision": decision,
            "fusion": trade_data.get("fusion"),
            "trend": trade_data.get("trend"),
            "fact": trade_data.get("fact"),
            "reward_due_at": trade_data.get("reward_due_at"),
            "reward_window_seconds": trade_data.get("reward_window_seconds"),
            "portfolio_snapshot": trade_data.get("portfolio_snapshot"),
            "metadata": entry.metadata,
        }
        return snapshot

    async def _evaluate_trade_rewards(self):
        pending = await self.redis.hgetall(PENDING_REWARD_KEY)
        if not pending:
            return

        if len(pending) > settings.trade_reward_max_pending:
            overflow = len(pending) - settings.trade_reward_max_pending
            for trade_id in list(pending.keys())[:overflow]:
                await self.redis.hdel(PENDING_REWARD_KEY, trade_id)
                log.bind(agent="MEMORY").warning(
                    "Pruned stale pending reward %s to respect cap", trade_id
                )
                pending.pop(trade_id, None)

        now = datetime.utcnow()
        evaluated = 0

        for trade_id, raw in pending.items():
            try:
                payload = json.loads(raw)
            except Exception:
                await self.redis.hdel(PENDING_REWARD_KEY, trade_id)
                continue

            confidence = self._safe_float(payload.get("confidence"), 0.0) or 0.0
            if confidence < settings.trade_reward_min_confidence:
                await self.redis.hdel(PENDING_REWARD_KEY, trade_id)
                continue

            due_at = self._parse_timestamp(payload.get("reward_due_at"))
            if due_at > now:
                continue

            ticker = payload.get("ticker")
            if not ticker:
                await self.redis.hdel(PENDING_REWARD_KEY, trade_id)
                continue

            evaluation_price = await self._resolve_current_price(ticker)
            if evaluation_price is None:
                # Retry later
                continue

            entry_price = self._safe_float(
                payload.get("entry_price") or payload.get("executed_price")
            )
            quantity = self._safe_float(payload.get("quantity"))
            action = self._normalize_action(payload.get("action"))

            if entry_price <= 0 or quantity <= 0:
                pnl = 0.0
                reward = 0.0
            else:
                direction = 1.0 if action == TradeAction.BUY else -1.0
                price_delta = (evaluation_price - entry_price) * direction
                pnl = price_delta * quantity
                reward = (price_delta / entry_price) if entry_price else 0.0

            payload.setdefault("metadata", {})
            payload.update(
                {
                    "evaluation_price": evaluation_price,
                    "pnl": pnl,
                    "reward": reward,
                    "status": self._determine_trade_status(pnl),
                    "reward_evaluated_at": now.isoformat(),
                }
            )

            await self._persist_trade_reward(trade_id, payload)
            await self.redis.hdel(PENDING_REWARD_KEY, trade_id)
            evaluated += 1

        if evaluated:
            log.info("Evaluated %d pending trade rewards", evaluated)

    async def _resolve_current_price(self, ticker: str) -> Optional[float]:
        if settings.trade_reward_price_source.lower() in {"chart", "auto"}:
            chart_payload = await self.redis.get_json(f"chart:signal:{ticker}")
            if chart_payload:
                data = chart_payload.get("data") or {}
                price = self._safe_float(data.get("current_price"))
                if price > 0:
                    return price

        portfolio = await self.redis.get_json("state:portfolio")
        if portfolio:
            price = self._safe_float(
                (portfolio.get("prices") or {}).get(ticker)
            )
            if price > 0:
                return price

        if settings.trade_reward_price_source.lower() in {"dqn", "auto"}:
            dqn_payload = await self.redis.get_json(f"dqn:prediction:{ticker}")
            if dqn_payload:
                data = dqn_payload.get("data") or {}
                price = self._safe_float(data.get("forecast_price"))
                if price > 0:
                    return price

        return None

    async def _persist_trade_reward(self, trade_id: str, payload: Dict[str, Any]):
        await self.redis.set_json(f"memory:trade:{trade_id}", payload)
        record = json.dumps(
            {
                "trade_id": trade_id,
                "ticker": payload.get("ticker"),
                "pnl": payload.get("pnl"),
                "reward": payload.get("reward"),
                "status": payload.get("status"),
                "timestamp": payload.get("reward_evaluated_at"),
            }
        )
        await self.redis.lpush("memory:trade_rewards", record)
        await self.redis.ltrim("memory:trade_rewards", 0, self.reward_history_limit - 1)

        entry = TradeMemoryEntry(
            trade_id=payload.get("trade_id", trade_id),
            ticker=payload.get("ticker", "UNKNOWN"),
            action=self._normalize_action(payload.get("action", "HOLD")),
            quantity=self._safe_float(payload.get("quantity")),
            price=self._safe_float(
                payload.get("executed_price")
                or payload.get("entry_price")
                or payload.get("price")
            ),
            pnl=self._extract_pnl(payload),
            status=payload.get("status", "UNKNOWN"),
            metadata=payload.get("metadata", {}),
            timestamp=self._parse_timestamp(payload.get("timestamp")),
        )
        await self._record_trade_in_graph(entry)
        await self._update_agent_rewards(payload, payload.get("reward", 0.0))

    async def _update_agent_rewards(self, payload: Dict[str, Any], reward: float):
        decision = payload.get("decision") or {}
        signals = decision.get("contributing_signals") or []
        if not signals:
            return

        rewards = await self.redis.get_json("memory:agent_rewards") or {}

        confidences = [
            self._safe_float(signal.get("confidence"), 0.0) for signal in signals
        ]
        total_conf = sum(confidences) if any(confidences) else float(len(signals) or 1)

        for signal, confidence in zip(signals, confidences or [1.0] * len(signals)):
            agent_key = self._normalize_agent_key(signal.get("agent_type"))
            if not agent_key:
                continue
            weight = confidence / total_conf if total_conf else 0.0
            agent_reward = reward * weight
            stats = rewards.get(agent_key, {"total_reward": 0.0, "trades": 0})
            stats["total_reward"] += agent_reward
            stats["trades"] += 1
            stats["average_reward"] = (
                stats["total_reward"] / stats["trades"] if stats["trades"] else 0.0
            )
            rewards[agent_key] = stats

        await self.redis.set_json("memory:agent_rewards", rewards)

    def _normalize_agent_key(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            candidate = value.get("value") or value.get("name")
        else:
            candidate = str(value)
        if not candidate:
            return ""
        return candidate.upper()

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
            entry = await self._parse_trade_entry(raw)
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
            entry = await self._parse_trade_entry(raw)
        for raw in trades_raw:
            entry = await self._parse_trade_entry(raw)
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

    async def _auto_enhance_long_term_memory(self):
        """Promote recent fusion recommendations into long-term graph memory."""
        try:
            for ticker in asset_registry.get_assets():
                fusion = await self.redis.get_json(f"pipeline:fusion:{ticker}")
                if not fusion:
                    continue

                fusion_id = fusion.get("generated_at")
                if not fusion_id:
                    continue

                cache_key = f"memory:auto_enhanced:{ticker}:{fusion_id}"
                if await self.redis.exists(cache_key):
                    continue

                action = fusion.get("action", "HOLD")
                confidence = float(fusion.get("confidence", 0.0))
                allocation = float(fusion.get("percent_allocation", 0.0))
                rationale = fusion.get("rationale", "")
                summary_text = (
                    f"{ticker} fusion decision: {action} with confidence {confidence:.2f} "
                    f"and allocation {allocation:.2%}. {rationale}"
                )

                node = GraphMemoryNode(
                    node_id=f"fusion:{ticker}:{fusion_id}",
                    label=f"{ticker} fusion insight",
                    node_type="fusion_insight",
                    weight=max(confidence, 0.1),
                    metadata={
                        "ticker": ticker,
                        "action": action,
                        "confidence": confidence,
                        "allocation": allocation,
                        "rationale": rationale,
                        "components": fusion.get("components", {}),
                    },
                )
                await self.graph_memory.upsert_node(node)

                asset_node = GraphMemoryNode(
                    node_id=f"asset:{ticker}",
                    label=ticker,
                    node_type="asset",
                    weight=1.0,
                )
                await self.graph_memory.upsert_node(asset_node)
                await self.graph_memory.connect(node.node_id, asset_node.node_id, "RELATES", weight=node.weight)

                if self.camel_memory:
                    try:
                        from camel.messages import BaseMessage  # pylint: disable=import-error
                        from camel.types import OpenAIBackendRole  # pylint: disable=import-error

                        message = BaseMessage.make_assistant_message(
                            role_name="Fusion Insight",
                            content=summary_text,
                        )
                        self.camel_memory.write_record(
                            message,
                            role=OpenAIBackendRole.ASSISTANT,
                            extra_info={
                                "ticker": ticker,
                                "confidence": confidence,
                                "allocation": allocation,
                            },
                        )
                    except Exception as embed_exc:  # pragma: no cover - optional dependency
                        log.debug("Unable to persist fusion insight to Qdrant: %s", embed_exc)

                await self.redis.set(cache_key, "1", expire=self.auto_enhance_ttl_seconds)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.debug("Auto enhancement skipped due to error: %s", exc)

    async def _prune_memories_if_needed(self) -> None:
        if not self.pruning_pipeline:
            return
        try:
            await self.pruning_pipeline.prune_all()
        except Exception as exc:  # pragma: no cover
            log.debug("Memory pruning skipped: %s", exc)

    async def _run_weight_review_if_due(self) -> None:
        if not self.weight_review_pipeline:
            return
        manual_trigger = await self.redis.get("review:trigger")
        if manual_trigger:
            await self.redis.delete("review:trigger")
            await self.weight_review_pipeline.run(trigger="manual")
            self._last_review_run = datetime.utcnow()
            return

        dashboard = await self.redis.get_json("dashboard:settings") or {}
        review_interval_hours = float(dashboard.get("review_interval_hours", settings.review_interval_hours))
        interval = timedelta(hours=review_interval_hours)
        if datetime.utcnow() - self._last_review_run < interval:
            return
        await self.weight_review_pipeline.run(trigger="schedule")
        self._last_review_run = datetime.utcnow()

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

    async def _parse_trade_entry(self, raw: str) -> Optional[TradeMemoryEntry]:
        trade_id: Optional[str] = None
        candidate: Optional[Dict[str, Any]] = None

        try:
            data = json.loads(raw)
        except Exception:
            try:
                parsed = TradeMemoryEntry.parse_raw(raw)
            except Exception:
                return None
            trade_id = parsed.trade_id
            candidate = parsed.dict()
        else:
            trade_id = (
                data.get("trade_id")
                or data.get("decision_id")
                or data.get("id")
            )
            if "data" in data and not trade_id:
                legacy = data["data"]
                trade_id = legacy.get("decision_id") or legacy.get("id")
                candidate = legacy
            elif "ticker" in data and "action" in data:
                candidate = data
            else:
                candidate = data.get("data")

        if trade_id:
            stored = await self.redis.get_json(f"memory:trade:{trade_id}")
            if stored:
                candidate = stored

        if not candidate:
            return None

        snapshot = candidate.copy()
        snapshot.setdefault("trade_id", trade_id or str(uuid4()))
        snapshot.setdefault("timestamp", snapshot.get("timestamp") or datetime.utcnow().isoformat())

        pnl_value = self._extract_pnl(snapshot)
        return TradeMemoryEntry(
            trade_id=snapshot.get("trade_id", str(uuid4())),
            ticker=snapshot.get("ticker", "UNKNOWN"),
            action=self._normalize_action(snapshot.get("action", "HOLD")),
            quantity=self._safe_float(
                snapshot.get("quantity")
                or snapshot.get("amount")
                or snapshot.get("size")
            ),
            price=self._safe_float(
                snapshot.get("executed_price")
                or snapshot.get("entry_price")
                or snapshot.get("price")
            ),
            pnl=pnl_value,
            status=snapshot.get("status") or self._determine_trade_status(pnl_value),
            metadata=snapshot.get("metadata", {}),
            timestamp=self._parse_timestamp(snapshot.get("timestamp")),
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

    def _safe_float(self, value: Any, default: Optional[float] = 0.0) -> Optional[float]:
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
        log.bind(agent="MEMORY").debug(
            "Graph memory recorded trade %s action=%s pnl=%s",
            entry.trade_id,
            entry.action.value,
            entry.pnl,
        )

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
        log.bind(agent="MEMORY").debug(
            "Graph memory recorded news %s ticker=%s sentiment=%.2f",
            entry.news_id,
            entry.ticker,
            entry.sentiment_score,
        )

