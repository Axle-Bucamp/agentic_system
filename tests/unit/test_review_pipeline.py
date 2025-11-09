import pytest

from core.models import AgentType
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


@pytest.mark.asyncio
async def test_weight_review_pipeline_overrides_prompt(monkeypatch):
    redis = FakeRedis()
    await redis.set_json("dashboard:settings", {"review_prompt": "custom prompt"})
    pipeline = WeightReviewPipeline(redis)
    await pipeline.run(trigger="schedule")
    snapshot = await redis.get_json("orchestrator:agent_weights")
    assert snapshot["prompt"] == "custom prompt"
    settings_payload = await redis.get_json("dashboard:settings")
    assert "agent_prompts" in settings_payload
    assert "mcp_overrides" in settings_payload
    assert "news" in settings_payload["mcp_overrides"]
    assert "news_source_weights" in settings_payload
    assert "toolkits" in settings_payload["mcp_overrides"]["news"]
    news_key = AgentType.NEWS.value
    news_prompt = settings_payload["agent_prompts"].get(news_key)
    assert news_prompt is not None
    assert "source weights" in news_prompt.lower()
