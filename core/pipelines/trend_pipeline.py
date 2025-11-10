"""Trend pipeline that fuses DQN and technical analysis outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from core.logging import log
from core.models import TradeAction, TrendAssessment
from core.pipelines.storage import set_trend_assessment
from core.redis_client import RedisClient
from core.memory.camel_memory_manager import CamelMemoryManager

try:
    from camel.messages import BaseMessage
except ImportError:  # pragma: no cover - optional dependency
    BaseMessage = None

ACTION_TO_SCORE = {
    TradeAction.BUY.value: 1.0,
    TradeAction.SELL.value: -1.0,
    TradeAction.HOLD.value: 0.0,
}


def _default_chart_stats(current_price: float = 0.0) -> Dict[str, float]:
    return {
        "rsi": 50.0,
        "bollinger_upper": 0.0,
        "bollinger_lower": 0.0,
        "current_price": current_price,
    }


@dataclass
class TrendInputs:
    dqn_action: TradeAction = TradeAction.HOLD
    dqn_confidence: float = 0.0
    dqn_forecast: Optional[float] = None
    chart_action: TradeAction = TradeAction.HOLD
    chart_confidence: float = 0.0
    chart_stats: Dict[str, float] = field(default_factory=_default_chart_stats)
    sources: Dict[str, bool] = field(default_factory=dict)


class TrendPipeline:
    """Generate deterministic trend assessments using cached agent outputs."""

    def __init__(self, redis: RedisClient, ttl_seconds: int = 300):
        self.redis = redis
        self.ttl_seconds = ttl_seconds
        try:
            self._memory_manager = CamelMemoryManager(agent_id="pipeline_trend")
        except Exception:
            self._memory_manager = None

    async def run_for_ticker(self, ticker: str) -> Optional[TrendAssessment]:
        """Produce and persist a trend assessment for a given ticker."""
        inputs = await self._load_inputs(ticker)
        if not inputs:
            log.debug("Trend pipeline skipped for %s due to missing inputs", ticker)
            return None

        assessment = self._build_assessment(ticker, inputs)
        await set_trend_assessment(self.redis, assessment, ttl_seconds=self.ttl_seconds)
        self._record_memory(assessment)
        return assessment

    async def _load_inputs(self, ticker: str) -> Optional[TrendInputs]:
        dqn_payload = await self.redis.get_json(f"dqn:prediction:{ticker}")
        chart_payload = await self.redis.get_json(f"chart:signal:{ticker}")

        sources = {"dqn": bool(dqn_payload), "chart": bool(chart_payload)}

        if not sources["dqn"] and not sources["chart"]:
            return None

        dqn_action = TradeAction.HOLD
        dqn_confidence = 0.0
        forecast_price: Optional[float] = None

        if dqn_payload:
            try:
                dqn_action = TradeAction(dqn_payload.get("action"))
            except Exception:
                dqn_action = TradeAction.HOLD
            try:
                dqn_confidence = float(dqn_payload.get("confidence", 0.0))
            except (TypeError, ValueError):
                dqn_confidence = 0.0

            data = dqn_payload.get("data") or {}
            forecast_price = data.get("forecast_price")
            if forecast_price is not None:
                try:
                    forecast_price = float(forecast_price)
                except (TypeError, ValueError):
                    forecast_price = None

        chart_stats = _default_chart_stats(current_price=forecast_price or 0.0)
        chart_action = TradeAction.HOLD
        chart_confidence = 0.0

        if chart_payload:
            try:
                chart_action = TradeAction(chart_payload.get("action"))
            except Exception:
                chart_action = TradeAction.HOLD
            try:
                chart_confidence = float(chart_payload.get("confidence", 0.0))
            except (TypeError, ValueError):
                chart_confidence = 0.0

            data = chart_payload.get("data") or {}
            chart_stats = {
                "rsi": float(data.get("rsi", 50.0) or 50.0),
                "bollinger_upper": float(data.get("bollinger_upper", 0.0) or 0.0),
                "bollinger_lower": float(data.get("bollinger_lower", 0.0) or 0.0),
                "current_price": float(data.get("current_price", forecast_price or 0.0) or 0.0),
            }

        # Ensure current price has some sensible fallback
        if chart_stats.get("current_price", 0.0) == 0.0 and forecast_price:
            chart_stats["current_price"] = forecast_price

        return TrendInputs(
            dqn_action=dqn_action,
            dqn_confidence=dqn_confidence,
            dqn_forecast=forecast_price,
            chart_action=chart_action,
            chart_confidence=chart_confidence,
            chart_stats=chart_stats,
            sources=sources,
        )

    def _build_assessment(self, ticker: str, inputs: TrendInputs) -> TrendAssessment:
        # Weighted action score normalised between -1 and 1
        dqn_score = ACTION_TO_SCORE.get(inputs.dqn_action.value, 0.0) * inputs.dqn_confidence
        chart_score = ACTION_TO_SCORE.get(inputs.chart_action.value, 0.0) * inputs.chart_confidence

        blended_score = (dqn_score * 0.6) + (chart_score * 0.4)
        trend_score = max(0.0, min((blended_score + 1.0) / 2.0, 1.0))

        # Momentum emphasises confidence from both sources
        momentum = max(
            0.0,
            min(
                (inputs.dqn_confidence * 0.7) + (inputs.chart_confidence * 0.3),
                1.0,
            ),
        )

        volatility = None
        price = inputs.chart_stats.get("current_price") or 0.0
        upper = inputs.chart_stats.get("bollinger_upper") or 0.0
        lower = inputs.chart_stats.get("bollinger_lower") or 0.0
        if price > 0 and upper > 0 and lower > 0:
            volatility = max((upper - lower) / price, 0.0)

        # Determine recommendation
        if blended_score > 0.15:
            action = TradeAction.BUY
        elif blended_score < -0.15:
            action = TradeAction.SELL
        else:
            action = TradeAction.HOLD

        confidence = max(0.0, min(abs(blended_score), 1.0))

        supporting_signals = {
            "dqn": {
                "action": inputs.dqn_action.value,
                "confidence": inputs.dqn_confidence,
                "forecast_price": inputs.dqn_forecast,
                "source_active": inputs.sources.get("dqn", False),
            },
            "chart": {
                "action": inputs.chart_action.value,
                "confidence": inputs.chart_confidence,
                "source_active": inputs.sources.get("chart", False),
                **inputs.chart_stats,
            },
            "blended_score": blended_score,
        }

        assessment = TrendAssessment(
            ticker=ticker,
            trend_score=trend_score,
            momentum=momentum,
            volatility=volatility,
            recommended_action=action,
            confidence=confidence,
            supporting_signals=supporting_signals,
        )
        assessment.agentic = True
        assessment.ai_explanation = (
            f"Trend analysis recommends {action.value} with confidence {confidence:.2f} "
            f"based on blended score {blended_score:.2f}."
        )
        return assessment

    def _record_memory(self, assessment: TrendAssessment) -> None:
        if not self._memory_manager or BaseMessage is None:
            return
        try:
            summary = (
                f"Trend assessment for {assessment.ticker}: "
                f"{assessment.recommended_action.value} @ confidence {assessment.confidence:.2f}; "
                f"score={assessment.trend_score:.2f}, momentum={assessment.momentum:.2f}."
            )
            message = BaseMessage.make_assistant_message(
                role_name="TrendPipeline",
                content=summary,
            )
            self._memory_manager.write_record(message)
        except Exception as exc:  # pragma: no cover
            log.debug("Unable to write trend pipeline memory: %s", exc)

