"""Shared storage helpers for pipeline outputs."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from core.logging import log
from core.models import FactInsight, FusionRecommendation, TrendAssessment
from core.redis_client import RedisClient


def _trend_key(ticker: str) -> str:
    return f"pipeline:trend:{ticker.upper()}"


def _fact_key(ticker: Optional[str]) -> str:
    target = "__market__" if ticker is None else ticker.upper()
    return f"pipeline:fact:{target}"


def _fusion_key(ticker: str) -> str:
    return f"pipeline:fusion:{ticker.upper()}"


async def get_trend_assessment(redis: RedisClient, ticker: str) -> Optional[TrendAssessment]:
    """Load a trend assessment for a ticker from Redis."""
    payload = await redis.get_json(_trend_key(ticker))
    if not payload:
        return None
    try:
        return TrendAssessment(**payload)
    except Exception as exc:  # pragma: no cover - defensive conversion
        log.warning("Unable to parse trend assessment for %s: %s", ticker, exc)
        return None


async def set_trend_assessment(
    redis: RedisClient,
    assessment: TrendAssessment,
    ttl_seconds: int = 300,
) -> None:
    """Persist a trend assessment to Redis."""
    await redis.set_json(_trend_key(assessment.ticker), assessment.dict(), expire=ttl_seconds)


async def get_fact_insight(redis: RedisClient, ticker: Optional[str]) -> Optional[FactInsight]:
    """Load fact insight (ticker-specific or market wide) from Redis."""
    payload = await redis.get_json(_fact_key(ticker))
    if not payload:
        return None
    try:
        return FactInsight(**payload)
    except Exception as exc:  # pragma: no cover - defensive conversion
        log.warning("Unable to parse fact insight for %s: %s", ticker or '__market__', exc)
        return None


async def set_fact_insight(
    redis: RedisClient,
    insight: FactInsight,
    ttl_seconds: int = 3600,
) -> None:
    """Persist a fact insight to Redis."""
    await redis.set_json(_fact_key(insight.ticker), insight.dict(), expire=ttl_seconds)


async def get_fusion_recommendation(redis: RedisClient, ticker: str) -> Optional[FusionRecommendation]:
    """Load the fusion recommendation for a ticker."""
    payload = await redis.get_json(_fusion_key(ticker))
    if not payload:
        return None
    try:
        return FusionRecommendation(**payload)
    except Exception as exc:  # pragma: no cover - defensive conversion
        log.warning("Unable to parse fusion recommendation for %s: %s", ticker, exc)
        return None


async def set_fusion_recommendation(
    redis: RedisClient,
    recommendation: FusionRecommendation,
    ttl_seconds: int = 300,
) -> None:
    """Persist a fusion recommendation to Redis."""
    await redis.set_json(_fusion_key(recommendation.ticker), recommendation.dict(), expire=ttl_seconds)


def is_stale(generated_at: datetime, ttl_seconds: int) -> bool:
    """Return True when the generated_at timestamp is older than ttl window."""
    return datetime.utcnow() - generated_at >= timedelta(seconds=max(ttl_seconds, 1))

