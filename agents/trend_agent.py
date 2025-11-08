"""
Trend Agent - Aggregates DQN and technical analysis into a consolidated trend view.
"""
from datetime import datetime
from typing import Optional

from agents.base_agent import BaseAgent
from core import asset_registry
from core.logging import log
from core.models import AgentSignal, AgentType, SignalType
from core.pipelines import TrendPipeline, get_trend_assessment
from core.pipelines.storage import is_stale


class TrendAgent(BaseAgent):
    """Agent responsible for producing fused trend assessments."""

    def __init__(self, redis_client):
        super().__init__(AgentType.TREND, redis_client)
        self.pipeline = TrendPipeline(redis_client)

    async def initialize(self):
        log.info("Trend Agent initialized (fusing DQN + chart signals)")

    async def process_message(self, message):
        # Trend agent currently runs on its own cadence
        return None

    async def run_cycle(self):
        log.debug("Trend Agent running cycle...")

        for ticker in asset_registry.get_assets():
            try:
                existing = await get_trend_assessment(self.redis, ticker)
                if existing and not is_stale(existing.generated_at, self.pipeline.ttl_seconds):
                    continue

                assessment = await self.pipeline.run_for_ticker(ticker)
                if not assessment:
                    continue

                signal = AgentSignal(
                    agent_type=self.agent_type,
                    signal_type=SignalType.TREND_ASSESSMENT,
                    ticker=ticker,
                    action=assessment.recommended_action,
                    confidence=assessment.confidence,
                    data=assessment.dict(),
                    reasoning=self._build_reasoning(assessment.dict()),
                )
                await self.send_signal(signal.dict())
            except Exception as exc:  # pragma: no cover - defensive logging only
                log.error("Trend Agent error for %s: %s", ticker, exc)

    def get_cycle_interval(self) -> int:
        return self.pipeline.ttl_seconds

    def _build_reasoning(self, data: dict) -> str:
        trend_score = data.get("trend_score")
        momentum = data.get("momentum")
        support = data.get("supporting_signals", {})
        dqn = support.get("dqn", {})
        chart = support.get("chart", {})
        return (
            f"Composite trend score {trend_score:.2f} with momentum {momentum:.2f}. "
            f"DQN {dqn.get('action')} ({dqn.get('confidence', 0):.2f}); "
            f"technical {chart.get('action')} ({chart.get('confidence', 0):.2f})."
        )

