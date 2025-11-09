"""Memory pruning pipeline to enforce sparsity and deduplicate content."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from core.config import settings
from core.logging import log
from core.models import NewsMemoryEntry, TradeAction, TradeMemoryEntry


def _tokenise(text: str) -> set[str]:
    return {token.lower() for token in text.split() if token}


def _similarity(a: str, b: str) -> float:
    tokens_a = _tokenise(a)
    tokens_b = _tokenise(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


class MemoryPruningPipeline:
    """Pipeline in charge of pruning FIFO memories and consolidating similar news entries."""

    SETTINGS_KEY = "dashboard:settings"

    def __init__(self, redis_client, camel_memory=None):
        self.redis = redis_client
        self.camel_memory = camel_memory
        self.limit = settings.memory_prune_limit
        self.similarity_threshold = settings.memory_prune_similarity_threshold

    async def prune_all(self) -> None:
        await self._refresh_settings()
        await self.prune_news()
        await self.prune_trades()

    async def prune_news(self) -> None:
        raw_entries = await self.redis.lrange("memory:news", 0, -1)
        entries: List[NewsMemoryEntry] = []
        for raw in raw_entries:
            try:
                parsed = NewsMemoryEntry.parse_raw(raw)
            except Exception:
                try:
                    data = json.loads(raw)
                    parsed = NewsMemoryEntry(**data)
                except Exception:
                    continue
            entries.append(parsed)

        if not entries:
            return

        entries.sort(key=lambda entry: entry.timestamp)
        groups: List[List[NewsMemoryEntry]] = []
        for entry in entries:
            matched = False
            for group in groups:
                if _similarity(group[-1].summary, entry.summary) >= self.similarity_threshold:
                    group.append(entry)
                    matched = True
                    break
            if not matched:
                groups.append([entry])

        summarised: List[NewsMemoryEntry] = []
        for group in groups:
            if len(group) == 1:
                summarised.append(group[0])
                continue
            latest = group[-1]
            total_weight = sum(item.weight for item in group)
            aggregated_sentiment = sum(item.sentiment_score * item.weight for item in group) / total_weight
            latest.sentiment_score = aggregated_sentiment
            latest.confidence = max(item.confidence for item in group)
            latest.weight = total_weight
            latest.summary = latest.summary[:240]
            if not isinstance(latest.metadata, dict):
                latest.metadata = {}
            source_weights = [
                (item.metadata or {}).get("source_weight", 0.1) for item in group
            ]
            avg_source_weight = sum(source_weights) / len(source_weights)
            latest.metadata["source_weight"] = avg_source_weight
            summarised.append(latest)

        summarised.sort(key=lambda entry: entry.timestamp, reverse=True)
        trimmed = summarised[: self.limit]

        await self.redis.delete("memory:news")
        for entry in reversed(trimmed):
            await self.redis.rpush("memory:news", entry.json())

        for ticker, grouped in self._group_by_ticker(trimmed).items():
            key = f"memory:news:{ticker}" if ticker else "memory:news"
            await self.redis.delete(key)
            for entry in reversed(grouped[: self.limit]):
                await self.redis.rpush(key, entry.json())

        log.info("Memory pruning pipeline applied to news entries (kept %d)" % len(trimmed))

    async def prune_trades(self) -> None:
        raw_entries = await self.redis.lrange("memory:trades", 0, -1)
        if len(raw_entries) <= self.limit:
            return
        entries: List[TradeMemoryEntry] = []
        for raw in raw_entries:
            trade: Optional[TradeMemoryEntry] = None
            try:
                data = json.loads(raw)
            except Exception:
                data = None

            trade_id = None
            if data:
                trade_id = (
                    data.get("trade_id")
                    or data.get("decision_id")
                    or data.get("id")
                )
            if trade_id:
                stored = await self.redis.get_json(f"memory:trade:{trade_id}")
                if stored:
                    data = stored
            if data:
                try:
                    trade = TradeMemoryEntry(
                        trade_id=data.get("trade_id") or trade_id or str(uuid4()),
                        ticker=data.get("ticker", "UNKNOWN"),
                        action=self._normalize_action(data.get("action", "HOLD")),
                        quantity=self._safe_float(data.get("quantity")),
                        price=self._safe_float(
                            data.get("executed_price")
                            or data.get("entry_price")
                            or data.get("price")
                        ),
                        pnl=self._safe_float(data.get("pnl"), default=None),
                        status=data.get("status"),
                        metadata=data.get("metadata", {}),
                        timestamp=self._parse_timestamp(data.get("timestamp")),
                    )
                except Exception:
                    trade = None
            if trade is None:
                try:
                    trade = TradeMemoryEntry.parse_raw(raw)
                except Exception:
                    continue
            entries.append(trade)
        if not entries:
            return
        entries.sort(key=lambda entry: entry.timestamp, reverse=True)
        trimmed = entries[: self.limit]
        await self.redis.delete("memory:trades")
        for entry in reversed(trimmed):
            await self.redis.rpush("memory:trades", entry.json())
        log.info("Memory pruning pipeline trimmed trades to %d entries", len(trimmed))

    def _group_by_ticker(self, entries: Iterable[NewsMemoryEntry]) -> Dict[Optional[str], List[NewsMemoryEntry]]:
        grouped: Dict[Optional[str], List[NewsMemoryEntry]] = defaultdict(list)
        for entry in entries:
            grouped[entry.ticker].append(entry)
        return grouped

    async def _refresh_settings(self) -> None:
        try:
            dashboard = await self.redis.get_json(self.SETTINGS_KEY) or {}
            self.limit = int(dashboard.get("memory_prune_limit", settings.memory_prune_limit))
            self.similarity_threshold = float(
                dashboard.get("memory_prune_similarity_threshold", settings.memory_prune_similarity_threshold)
            )
        except Exception:
            self.limit = settings.memory_prune_limit
            self.similarity_threshold = settings.memory_prune_similarity_threshold

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

    def _parse_timestamp(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if not value:
            return datetime.utcnow()
        text = str(value)
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return datetime.utcnow()
