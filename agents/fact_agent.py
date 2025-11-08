"""
Fact Agent - Produces fundamental insights via news, sentiment, and research aggregation.
"""
from typing import Optional

from agents.base_agent import BaseAgent
from core import asset_registry
from core.logging import log
from core.models import AgentSignal, AgentType, SignalType, TradeAction
from core.pipelines import FactPipeline, get_fact_insight
from core.pipelines.storage import is_stale


class FactAgent(BaseAgent):
    """Agent responsible for deep fact gathering and summarisation."""

    def __init__(self, redis_client):
        super().__init__(AgentType.FACT, redis_client)
        self.pipeline = FactPipeline(redis_client)

    async def initialize(self):
        log.info("Fact Agent initialized (news + deep research fusion)")

    async def stop(self):
        await self.pipeline.close()
        await super().stop()

    async def process_message(self, message):
        # Fact agent is primarily schedule-driven today
        return None

    async def run_cycle(self):
        log.debug("Fact Agent running cycle...")

        for ticker in asset_registry.get_assets():
            try:
                existing = await get_fact_insight(self.redis, ticker)
                if existing and not is_stale(existing.generated_at, self.pipeline.ttl_seconds):
                    continue

                insight = await self.pipeline.run_for_ticker(ticker)
                if not insight:
                    continue

                action = self._derive_action(insight.sentiment_score)
                confidence = max(min(abs(insight.sentiment_score), 1.0), 0.0) * insight.confidence

                signal = AgentSignal(
                    agent_type=self.agent_type,
                    signal_type=SignalType.FACT_SUMMARY,
                    ticker=ticker,
                    action=action,
                    confidence=round(confidence, 3),
                    data=insight.dict(),
                    reasoning=insight.thesis,
                )
                await self.send_signal(signal.dict())
            except Exception as exc:  # pragma: no cover - defensive logging only
                log.error("Fact Agent error for %s: %s", ticker, exc)

    def get_cycle_interval(self) -> int:
        return max(self.pipeline.ttl_seconds, 900)

    def _derive_action(self, sentiment: float) -> TradeAction:
        if sentiment > 0.2:
            return TradeAction.BUY
        if sentiment < -0.2:
            return TradeAction.SELL
        return TradeAction.HOLD

