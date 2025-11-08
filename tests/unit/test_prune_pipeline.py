import json
from datetime import datetime, timedelta

import pytest

from core.models import NewsMemoryEntry
from core.pipelines.prune_pipeline import MemoryPruningPipeline
from tests.conftest import FakeRedis


@pytest.mark.asyncio
async def test_prune_pipeline_deduplicates_news():
    redis = FakeRedis()
    now = datetime.utcnow()
    entries = [
        NewsMemoryEntry(
            news_id=f"id{i}",
            ticker="ETH",
            sentiment_score=0.2,
            confidence=0.9,
            summary="Ethereum upgrade optimism spreading",
            sources=["Test"],
            weight=1.0,
            metadata={},
            timestamp=now - timedelta(minutes=i),
        )
        for i in range(3)
    ]
    for entry in entries:
        await redis.rpush("memory:news", entry.json())
    pipeline = MemoryPruningPipeline(redis)
    pipeline.limit = 2
    await pipeline.prune_news()
    remaining = await redis.lrange("memory:news", 0, -1)
    assert len(remaining) == 2


@pytest.mark.asyncio
async def test_prune_pipeline_trims_trades():
    redis = FakeRedis()
    now = datetime.utcnow()
    for i in range(105):
        trade = {
            "trade_id": str(i),
            "ticker": "BTC",
            "action": "BUY",
            "quantity": 1.0,
            "price": 30000 + i,
            "pnl": 0.0,
            "status": "WIN",
            "metadata": {},
            "timestamp": (now - timedelta(minutes=i)).isoformat(),
        }
        await redis.rpush("memory:trades", json.dumps(trade))
    pipeline = MemoryPruningPipeline(redis)
    pipeline.limit = 100
    await pipeline.prune_trades()
    remaining = await redis.lrange("memory:trades", 0, -1)
    assert len(remaining) == 100
