"""Fusion engine that collapses pipeline outputs into deterministic portfolio guidance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from core.models import (
    FactInsight,
    FusionRecommendation,
    RiskMetrics,
    TradeAction,
    TrendAssessment,
)


@dataclass
class FusionInputs:
    trend: Optional[TrendAssessment]
    fact: Optional[FactInsight]
    risk: Optional[RiskMetrics]
    copy_confidence: float


class FusionEngine:
    """Compute fusion recommendations from trend/fact/risk/copy signals."""

    def __init__(
        self,
        min_action_threshold: float = 0.15,
        max_allocation: float = 0.25,
    ):
        self.min_action_threshold = min_action_threshold
        self.max_allocation = max_allocation

    def combine(self, ticker: str, inputs: FusionInputs) -> FusionRecommendation:
        trend_score = inputs.trend.trend_score if inputs.trend else 0.5
        trend_weight = 0.4 if inputs.trend else 0.0

        fact_score = (inputs.fact.sentiment_score + 1) / 2 if inputs.fact else 0.5
        fact_weight = 0.35 if inputs.fact else 0.0

        copy_bonus = min(max(inputs.copy_confidence, 0.0), 1.0) * 0.15

        risk_score = 0.3
        risk_level = "UNKNOWN"
        stop_loss_upper = 0.015
        stop_loss_lower = -0.03

        if inputs.risk:
            risk_score = min(max(inputs.risk.risk_score or 0.3, 0.05), 0.95)
            risk_level = inputs.risk.risk_level
            if inputs.risk.stop_loss_upper is not None:
                stop_loss_upper = inputs.risk.stop_loss_upper
            if inputs.risk.stop_loss_lower is not None:
                stop_loss_lower = inputs.risk.stop_loss_lower

        # Weighted blended score, centre 0.5 baseline to avoid null data bias
        combined_weight = max(trend_weight + fact_weight + 0.15, 1e-6)
        baseline_score = 0.5 * (trend_weight + fact_weight)
        blended = (trend_score * trend_weight + fact_score * fact_weight) - baseline_score
        blended /= combined_weight
        blended += copy_bonus

        # Risk adjustment
        blended *= (1.0 - risk_score)

        # Determine action and confidence
        if blended > self.min_action_threshold:
            action = TradeAction.BUY
        elif blended < -self.min_action_threshold:
            action = TradeAction.SELL
        else:
            action = TradeAction.HOLD

        confidence = min(abs(blended) * 1.5, 1.0)

        percent_allocation = 0.0
        if action == TradeAction.BUY:
            percent_allocation = max(0.0, min(blended, 1.0)) * self.max_allocation
        elif action == TradeAction.SELL:
            percent_allocation = max(0.0, min(abs(blended), 1.0)) * self.max_allocation

        components: Dict[str, object] = {
            "trend": inputs.trend.dict() if inputs.trend else None,
            "fact": inputs.fact.dict() if inputs.fact else None,
            "risk": inputs.risk.dict() if inputs.risk else None,
            "copy_confidence": inputs.copy_confidence,
            "blended_score": blended,
        }

        rationale_parts = []
        if inputs.trend:
            rationale_parts.append(
                f"Trend suggests {inputs.trend.recommended_action.value} with {inputs.trend.confidence:.2f} confidence."
            )
        if inputs.fact:
            tone = "positive" if inputs.fact.sentiment_score > 0 else "negative" if inputs.fact.sentiment_score < 0 else "neutral"
            rationale_parts.append(f"Fundamental tone is {tone} ({inputs.fact.sentiment_score:+.2f}).")
        if inputs.copy_confidence > 0.2:
            rationale_parts.append(f"Copy-trade wallets consensus adds {inputs.copy_confidence:.2f} momentum.")
        rationale_parts.append(f"Risk level {risk_level} moderates exposure.")
        rationale = " ".join(rationale_parts)

        return FusionRecommendation(
            ticker=ticker,
            action=action,
            confidence=confidence,
            percent_allocation=round(percent_allocation, 4),
            stop_loss_upper=round(stop_loss_upper, 4),
            stop_loss_lower=round(stop_loss_lower, 4),
            risk_level=risk_level,
            rationale=rationale,
            components=components,
        )

