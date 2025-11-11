import pytest

from core.pipelines.review_pipeline import WeightReviewPipeline, get_cached_weights
from tests.conftest import FakeRedis


@pytest.mark.asyncio
async def test_weight_review_pipeline_generates_weights():
    redis = FakeRedis()
    pipeline = WeightReviewPipeline(redis)
    weights = await pipeline.run(trigger="test")
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    cached = await get_cached_weights(redis)
    assert cached is not None
    assert set(weights.keys()) == set(cached.keys())
    snapshot = await redis.get_json("orchestrator:agent_weights")
    assert snapshot is not None
    assert snapshot["agentic"] is False
    assert isinstance(snapshot.get("ai_explanation"), str)


@pytest.mark.asyncio
async def test_weight_review_pipeline_uses_custom_prompt():
    redis = FakeRedis()
    await redis.set_json("dashboard:settings", {"review_prompt": "custom prompt"})
    pipeline = WeightReviewPipeline(redis)
    await pipeline.run(trigger="schedule")
    snapshot = await redis.get_json("orchestrator:agent_weights")
    assert snapshot["prompt"] == "custom prompt"
    assert snapshot["agentic"] is False
    settings_payload = await redis.get_json("dashboard:settings")
    assert settings_payload["review_prompt"] == "custom prompt"
    assert "latest_review_snapshot" in settings_payload
    latest_snapshot = settings_payload["latest_review_snapshot"]
    assert latest_snapshot["prompt"] == "custom prompt"
    assert "agent_prompts" not in settings_payload
    assert "mcp_overrides" not in settings_payload
