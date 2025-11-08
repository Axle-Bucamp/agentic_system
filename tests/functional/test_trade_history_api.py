import json

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_portfolio_trades_returns_enriched_records(fake_redis):
    trade_id = "trade-123"
    snapshot = {
        "trade_id": trade_id,
        "ticker": "BTC",
        "action": "BUY",
        "quantity": 0.1,
        "executed_price": 30000.0,
        "entry_price": 30000.0,
        "pnl": 250.0,
        "status": "WIN",
        "timestamp": "2025-11-08T12:30:00Z",
    }
    await fake_redis.set_json(f"memory:trade:{trade_id}", snapshot)
    await fake_redis.rpush("memory:trades", json.dumps({"trade_id": trade_id}))

    response = client.get("/api/portfolios/main/trades")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    record = payload["trades"][0]
    assert record["trade_id"] == trade_id
    assert record["pnl"] == 250.0


@pytest.mark.asyncio
async def test_agent_rewards_endpoint(fake_redis):
    await fake_redis.set_json(
        "memory:agent_rewards",
        {"TREND": {"total_reward": 1.5, "trades": 3, "average_reward": 0.5}},
    )

    response = client.get("/api/agents/rewards")
    assert response.status_code == 200
    payload = response.json()
    assert "rewards" in payload
    assert payload["rewards"]["TREND"]["average_reward"] == 0.5

