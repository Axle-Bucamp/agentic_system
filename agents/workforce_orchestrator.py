"""
Light-weight CAMEL workforce orchestrator.

Routes every signal through `CamelTradingRuntime`, which manages the CAMEL
society, toolkit registry, and the graceful fallback path when camel-ai is
missing or misconfigured.
"""

from __future__ import annotations

from typing import Optional

from agents.base_agent import BaseAgent
from core.camel_runtime.runtime import CamelTradingRuntime
from core.logging import log
from core.models import AgentMessage, AgentType, MessageType
from core.redis_client import RedisClient


class WorkforceOrchestratorAgent(BaseAgent):
    """Thin proxy between Redis events and the shared CAMEL trading runtime."""

    def __init__(self, redis_client: RedisClient) -> None:
        super().__init__(AgentType.ORCHESTRATOR, redis_client)
        self._runtime: Optional[CamelTradingRuntime] = None

    async def initialize(self) -> None:
        log.info("Initialising CamelTradingRuntime for workforce orchestrator")
        self._runtime = await CamelTradingRuntime.instance()

    async def process_message(self, message: AgentMessage):
        if not self._runtime:
            log.warning("CamelTradingRuntime not ready; message dropped")
            return None

        if message.message_type == MessageType.SIGNAL_GENERATED and message.payload:
            decision = await self._runtime.process_signal(message.payload)
            log.bind(agent="WORKFORCE", ticker=decision.get("ticker")).info(
                "Workforce decision captured (agentic=%s): %s",
                decision.get("agentic", True),
                decision.get("result"),
            )
        elif message.message_type == MessageType.RISK_ALERT:
            await self.redis.set("orchestrator:trading_paused", "1", expire=3600)
            log.warning("Risk alert received; trading paused for 1 hour")
        return None

    async def run_cycle(self) -> None:
        if not self._runtime:
            return

        ready = await self._runtime.ensure_ready()
        log.debug("Workforce orchestrator heartbeat (agentic=%s)", ready)

    def get_cycle_interval(self) -> int:
        # Keep the loop relaxed: the runtime reacts to inbound events.
        return 600
