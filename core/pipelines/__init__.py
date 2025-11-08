"""Shared pipeline utilities for agentic orchestration."""

from .storage import (
    get_trend_assessment,
    set_trend_assessment,
    get_fact_insight,
    set_fact_insight,
    get_fusion_recommendation,
    set_fusion_recommendation,
)
from .trend_pipeline import TrendPipeline
from .fact_pipeline import FactPipeline
from .fusion_pipeline import FusionEngine, FusionInputs
from .review_pipeline import WeightReviewPipeline, get_cached_weights
from .prune_pipeline import MemoryPruningPipeline

__all__ = [
    "TrendPipeline",
    "FactPipeline",
    "FusionEngine",
    "FusionInputs",
    "WeightReviewPipeline",
    "MemoryPruningPipeline",
    "get_cached_weights",
    "get_trend_assessment",
    "set_trend_assessment",
    "get_fact_insight",
    "set_fact_insight",
    "get_fusion_recommendation",
    "set_fusion_recommendation",
]

