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
from core.memory.camel_memory_manager import CamelMemoryManager

try:
    from camel.messages import BaseMessage
except ImportError:  # pragma: no cover
    BaseMessage = None


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
        try:
            self._memory_manager = CamelMemoryManager(agent_id="pipeline_fusion")
        except Exception:
            self._memory_manager = None

    def combine(self, ticker: str, inputs: FusionInputs) -> FusionRecommendation:
        trend_score = inputs.trend.trend_score if inputs.trend else 0.5
        trend_weight = 0.4 if inputs.trend else 0.0

        if inputs.fact:
            breakdown_score = None
            if isinstance(inputs.fact.sentiment_breakdown, dict):
                breakdown_score = inputs.fact.sentiment_breakdown.get("aggregate_score")
                if isinstance(breakdown_score, (int, float)):
                    fact_score = (breakdown_score + 1) / 2
                else:
                    fact_score = (inputs.fact.sentiment_score + 1) / 2
            else:
                fact_score = (inputs.fact.sentiment_score + 1) / 2
            fact_weight = 0.35
        else:
            fact_score = 0.5
            fact_weight = 0.0

        copy_bonus = min(max(inputs.copy_confidence, 0.0), 1.0) * 0.15

        risk_score = 0.3
        risk_level = "UNKNOWN"
        stop_loss_upper = 0.015
        stop_loss_lower = -0.03

        market_regime_bias = 0.0
        if inputs.fact and isinstance(inputs.fact.sentiment_breakdown, dict):
            components = inputs.fact.sentiment_breakdown.get("components", {})
            regime = components.get("market_regime", {}) if isinstance(components, dict) else {}
            regime_score = regime.get("score") if isinstance(regime, dict) else None
            if isinstance(regime_score, (int, float)):
                market_regime_bias = regime_score * 0.1

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
        blended += market_regime_bias

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
            "trend": inputs.trend.model_dump(mode="json") if inputs.trend else None,
            "fact": inputs.fact.model_dump(mode="json") if inputs.fact else None,
            "risk": inputs.risk.model_dump(mode="json") if inputs.risk else None,
            "copy_confidence": inputs.copy_confidence,
            "blended_score": blended,
            "market_regime_bias": market_regime_bias,
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

        recommendation = FusionRecommendation(
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
        recommendation.agentic = True
        recommendation.ai_explanation = (
            f"Fusion engine recommends {action.value} with confidence {confidence:.2f} "
            f"after blending technical, fundamental, copy-trade, and risk signals."
        )
        self._record_memory(recommendation)
        return recommendation

    def _record_memory(self, recommendation: FusionRecommendation) -> None:
        if not self._memory_manager or BaseMessage is None:
            return
        try:
            summary = (
                f"Fusion recommendation for {recommendation.ticker}: {recommendation.action.value} "
                f"with {recommendation.confidence:.2f} confidence, allocation {recommendation.percent_allocation:.2%}."
            )
            message = BaseMessage.make_assistant_message(
                role_name="FusionPipeline",
                content=summary,
            )
            self._memory_manager.write_record(message)
        except Exception as exc:
            if BaseMessage is not None:
                # best-effort logging; keep pipeline resilient
                from core.logging import log as fusion_log

                fusion_log.debug("Unable to record fusion pipeline memory: %s", exc)

