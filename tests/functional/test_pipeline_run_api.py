import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.models import NewsMemoryEntry
from core.config import settings

client = TestClient(app)


@pytest.mark.asyncio
async def test_trend_pipeline_endpoint(fake_redis):
    now = datetime.utcnow().isoformat()
    await fake_redis.set_json(
        "dqn:prediction:BTC",
        {
            "action": "BUY",
            "confidence": 0.8,
            "data": {"forecast_price": 32000},
            "generated_at": now,
        },
    )
    await fake_redis.set_json(
        "chart:signal:BTC",
        {
            "action": "BUY",
            "confidence": 0.6,
            "data": {
                "rsi": 45,
                "bollinger_upper": 1.1,
                "bollinger_lower": 0.9,
                "current_price": 30000,
            },
            "generated_at": now,
        },
    )
    response = client.post("/api/pipelines/trend/run", json={"ticker": "BTC"})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["assessment"]["ticker"] == "BTC"
    assert body["assessment"]["agentic"] is True
    assert isinstance(body["assessment"]["ai_explanation"], str) and body["assessment"]["ai_explanation"]
    assert "blended_score" in body["assessment"]["supporting_signals"]


@pytest.mark.asyncio
async def test_trend_pipeline_endpoint_with_partial_inputs(fake_redis):
    now = datetime.utcnow().isoformat()
    await fake_redis.set_json(
        "dqn:prediction:ETH",
        {
            "action": "BUY",
            "confidence": 0.7,
            "data": {"forecast_price": 2100},
            "generated_at": now,
        },
    )
    response = client.post("/api/pipelines/trend/run", json={"ticker": "ETH"})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["assessment"]["ticker"] == "ETH"
    assert body["assessment"]["supporting_signals"]["chart"]["source_active"] is False


@pytest.mark.asyncio
async def test_fact_pipeline_endpoint(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "cmc_api_key", None, raising=False)
    now = datetime.utcnow()
    entry = NewsMemoryEntry(
        news_id="id1",
        ticker="BTC",
        sentiment_score=0.4,
        confidence=0.8,
        summary="Bitcoin adoption is rising rapidly among institutions",
        sources=["Test"],
        weight=1.0,
        metadata={},
        timestamp=now,
    )
    await fake_redis.set_json(
        "memory:news:weighted:BTC",
        {
            "weighted_score": 0.4,
            "total_weight": 1.2,
            "last_updated": now.isoformat(),
        },
    )
    await fake_redis.rpush("memory:news:BTC", entry.json())
    response = client.post("/api/pipelines/fact/run", json={"ticker": "BTC"})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    insight = body["insight"]
    assert insight["agentic"] is True
    assert isinstance(insight["ai_explanation"], str) and insight["ai_explanation"]
    assert insight["sentiment_breakdown"]["aggregate_score"] is not None
    assert "news_sentiment" in insight["sentiment_breakdown"]["components"]


@pytest.mark.asyncio
async def test_fusion_pipeline_endpoint(fake_redis):
    now = datetime.utcnow().isoformat()
    await fake_redis.set_json(
        "pipeline:trend:BTC",
        {
            "ticker": "BTC",
            "trend_score": 0.65,
            "momentum": 0.7,
            "volatility": 0.05,
            "recommended_action": "BUY",
            "confidence": 0.8,
            "supporting_signals": {},
            "generated_at": now,
            "agentic": True,
            "ai_explanation": "trend explanation",
        },
    )
    await fake_redis.set_json(
        "pipeline:fact:BTC",
        {
            "ticker": "BTC",
            "sentiment_score": 0.3,
            "confidence": 0.6,
            "thesis": "Bullish outlook",
            "references": [],
            "anomalies": [],
            "generated_at": now,
            "agentic": True,
            "ai_explanation": "fact explanation",
            "sentiment_breakdown": {
                "aggregate_score": 0.3,
                "total_weight": 1.0,
                "components": {
                    "news_sentiment": {"score": 0.3, "weight": 1.0, "last_updated": now},
                },
            },
            "market_indicators": {},
        },
    )
    await fake_redis.set_json(
        "risk:asset:BTC",
        {
            "ticker": "BTC",
            "risk_level": "LOW",
            "risk_score": 0.1,
            "warnings": [],
            "timestamp": now,
            "agentic": True,
        },
    )
    response = client.post("/api/pipelines/fusion/run", json={"ticker": "BTC"})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["recommendation"]["ticker"] == "BTC"
    assert body["recommendation"]["agentic"] is True
    assert isinstance(body["recommendation"]["ai_explanation"], str) and body["recommendation"]["ai_explanation"]
    assert "market_regime_bias" in body["recommendation"]["components"]


@pytest.mark.asyncio
async def test_prune_pipeline_endpoint(fake_redis):
    for i in range(5):
        await fake_redis.rpush("memory:trades", json.dumps({"id": i}))
    response = client.post("/api/pipelines/prune/run")
    assert response.status_code == 200
    assert response.json()["success"] is True
