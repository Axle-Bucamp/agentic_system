"""
CamelTradingRuntime orchestrates society, memory, and task execution.

This runtime is the single integration point used by the orchestrator,
pipelines, and API entrypoints.  It exposes convenience methods for
processing workforce tasks, running ad-hoc prompts, and marshalling
decision traces.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Optional

from camel.messages import BaseMessage

from core.logging import log
from core.redis_client import redis_client
from core.camel_runtime.societies import society_factory
from core.models import AgentSignal


class CamelTradingRuntime:
    """Singleton-style runtime wrapper."""

    _instance: Optional["CamelTradingRuntime"] = None
    _initialise_lock = asyncio.Lock()

    def __init__(self) -> None:
        self._workforce = None
        self._fallback_mode = False
        self._fallback_history: list[Dict[str, Any]] = []
        self._runtime_lock = asyncio.Lock()

    @classmethod
    async def instance(cls) -> "CamelTradingRuntime":
        if cls._instance is None:
            async with cls._initialise_lock:
                if cls._instance is None:
                    cls._instance = CamelTradingRuntime()
                    await cls._instance._initialise()
        return cls._instance

    async def _initialise(self) -> None:
        """Initialise the workforce society and ensure Redis connection."""
        try:
            self._workforce = await society_factory.build()
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._fallback_mode = True
            self._workforce = None
            log.warning(
                "Failed to initialise CAMEL workforce (%s); entering heuristic fallback mode.",
                exc,
            )
        if not redis_client.redis:
            await redis_client.connect()
        log.info(
            "CamelTradingRuntime initialised%s",
            " (fallback mode)" if self._fallback_mode else "",
        )

    async def ensure_ready(self) -> bool:
        """Attempt to recover the workforce if we previously entered fallback mode."""

        if not self._fallback_mode:
            return self._workforce is not None

        async with self._runtime_lock:
            if not self._fallback_mode:
                return self._workforce is not None

            try:
                workforce = await society_factory.build()
            except Exception as exc:  # pragma: no cover - recovery attempts
                log.debug("CAMEL workforce still unavailable: %s", exc)
                return False

            self._workforce = workforce
            self._fallback_mode = False
            log.info("CamelTradingRuntime recovered from fallback mode")
            return True

    # ------------------------------------------------------------------ #
    # Workforce interaction                                              #
    # ------------------------------------------------------------------ #

    async def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a trading signal as a workforce task and return the decision.

        The workforce coordinates specialised workers (forecasting, risk,
        execution) to produce an aggregated recommendation.  The results are
        also persisted to Redis so the UI can display decision traces.
        """
        if not self._workforce or self._fallback_mode:
            ready = await self.ensure_ready()
            if not ready:
                return await self._fallback_process_signal(signal)

        decision_id = str(uuid.uuid4())
        task_message = BaseMessage.make_user_message(
            role_name="Trading Signal Intake",
            content=(
                "You are the Camel-AI trading workforce. Execute the canonical workflow for every signal:\n"
                "1. Validate context (ticker availability, wallet exposure) and call list_supported_assets if uncertain.\n"
                "2. Forecast phase: request the DQN worker to analyse policy actions and price targets.\n"
                "3. Technical phase: obtain chart diagnostics (RSI, MACD, Bollinger, momentum, volatility).\n"
                "4. Research phase: gather news/sentiment catalysts and macro narratives with citations.\n"
                "5. Risk phase: evaluate portfolio risk, sizing limits, and stop levels based on current exposure.\n"
                "6. Execution phase: simulate viable trades, accounting for fees, slippage, and wallet balances.\n"
                "7. Synthesis: deliver a final recommendation summarising action, confidence, risk guardrails, "
                "and outstanding concerns. Reference tool outputs explicitly.\n"
                "Inspect guidry-cloud health via get_guidry_cloud_api_stats whenever forecasting reliability is relevant "
                "and carry forward any warnings.\n"
                f"Signal payload: {signal}"
            ),
        )

        log.bind(decision_id=decision_id).info("Dispatching workforce task for signal %s", signal.get("ticker"))
        response = await self._workforce.process_task(task_message)

        decision_payload = self._format_decision(decision_id, signal, response)
        await redis_client.set_json(f"ai_decision:{decision_id}", decision_payload)
        return decision_payload

    async def run_task(self, instructions: str) -> Dict[str, Any]:
        """Run an ad-hoc workforce task."""
        if not self._workforce or self._fallback_mode:
            ready = await self.ensure_ready()
            if not ready:
                return await self._fallback_run_task(instructions)

        task_message = BaseMessage.make_user_message(role_name="Task", content=instructions)
        response = await self._workforce.process_task(task_message)
        return {"success": True, "response": response, "agentic": True}

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _format_decision(self, decision_id: str, signal: Dict[str, Any], response: Any) -> Dict[str, Any]:
        """Serialise the workforce response into the UI-friendly schema."""
        agent_signal = AgentSignal.parse_obj(signal)
        result_text = ""
        messages = []

        if isinstance(response, dict):
            result_text = response.get("text") or response.get("result") or ""
            messages = response.get("messages", [])
        elif hasattr(response, "msgs"):
            messages = [msg.content for msg in response.msgs]
            if messages:
                result_text = messages[-1]
        else:
            result_text = str(response)

        payload = {
            "decision_id": decision_id,
            "status": "completed",
            "ticker": agent_signal.ticker,
            "action": agent_signal.action.name if agent_signal.action else None,
            "confidence": agent_signal.confidence,
            "result": result_text or "[no agent response]",
            "messages": messages,
            "signal": signal,
        }
        payload["agentic"] = not self._fallback_mode
        payload["ai_explanation"] = (
            result_text.strip()
            if result_text
            else f"Aggregated workforce decision for {agent_signal.ticker or 'N/A'}."
        )

        return payload

    async def _fallback_process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Store the signal and emulate a deterministic response when CAMEL is unavailable."""

        decision_id = str(uuid.uuid4())
        agent_signal = AgentSignal.parse_obj(signal)
        fallback_reasoning = (
            "CAMEL workforce unavailable; recorded signal for audit only."
        )
        payload = {
            "decision_id": decision_id,
            "status": "degraded",
            "ticker": agent_signal.ticker,
            "action": agent_signal.action.name if agent_signal.action else None,
            "confidence": agent_signal.confidence,
            "result": fallback_reasoning,
            "messages": [],
            "signal": signal,
        }
        payload["agentic"] = False
        payload["ai_explanation"] = fallback_reasoning

        self._fallback_history.append(payload)
        await redis_client.set_json(f"ai_decision:{decision_id}", payload)
        log.warning("Fallback runtime stored signal for %s (action=%s)", agent_signal.ticker, payload["action"])
        return payload

    async def _fallback_run_task(self, instructions: str) -> Dict[str, Any]:
        """Return a deterministic response for ad-hoc tasks when fallback mode is active."""

        message = {
            "success": False,
            "response": {
                "text": (
                    "CAMEL workforce runtime is not available in this environment. "
                    "The request has been captured for later processing."
                ),
                "instructions": instructions,
            },
            "agentic": False,
            "ai_explanation": "Workforce runtime unavailable; task captured for audit only.",
        }
        log.warning("Fallback runtime received task request: %s", instructions[:120])
        return message


# exported factory helpers
async def process_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    runtime = await CamelTradingRuntime.instance()
    return await runtime.process_signal(signal)


async def run_task(instructions: str) -> Dict[str, Any]:
    runtime = await CamelTradingRuntime.instance()
    return await runtime.run_task(instructions)

