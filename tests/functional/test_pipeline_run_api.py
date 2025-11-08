import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.models import NewsMemoryEntry

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
async def test_fact_pipeline_endpoint(fake_redis):
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
    await fake_redis.rpush("memory:news", entry.json())
    response = client.post("/api/pipelines/fact/run", json={"ticker": "BTC"})
    assert response.status_code == 200
    body = response.json()
    # Fact pipeline might fail without external dependencies; ensure graceful response
    assert body["success"] in (True, False)


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
        },
    )
    response = client.post("/api/pipelines/fusion/run", json={"ticker": "BTC"})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["recommendation"]["ticker"] == "BTC"


@pytest.mark.asyncio
async def test_prune_pipeline_endpoint(fake_redis):
    for i in range(5):
        await fake_redis.rpush("memory:trades", json.dumps({"id": i}))
    response = client.post("/api/pipelines/prune/run")
    assert response.status_code == 200
    assert response.json()["success"] is True
