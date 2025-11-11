"""Agent weight review pipeline with optional CAMEL decision making."""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from statistics import fmean
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from core.config import settings
from core.logging import log
from core.models import AgentType
from core.models.camel_models import CamelModelFactory

try:  # Optional CAMEL dependency
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from camel.types import OpenAIBackendRole
except ImportError:  # pragma: no cover - optional dependency
    ChatAgent = None  # type: ignore
    BaseMessage = None  # type: ignore
    OpenAIBackendRole = None  # type: ignore

REDIS_REVIEW_KEY = "orchestrator:agent_weights"
REDIS_REVIEW_HISTORY = "orchestrator:agent_weights_history"
REDIS_REVIEW_METADATA = "orchestrator:agent_weights_meta"
MAX_AGENT_ATTEMPTS = 2


DEFAULT_WEIGHTS: Dict[str, float] = {
    AgentType.TREND.value: 0.35,
    AgentType.FACT.value: 0.20,
    AgentType.DQN.value: 0.15,
    AgentType.CHART.value: 0.10,
    AgentType.COPYTRADE.value: 0.10,
    AgentType.NEWS.value: 0.05,
    AgentType.RISK.value: 0.05,
}


class ReviewAgentResponse(BaseModel):
    """Structured schema expected from the CAMEL review agent."""

    weights: Dict[str, float] = Field(description="Updated agent weight mapping.")
    explanation: str = Field(description="Short explanation of the weight adjustments.")
    notes: Optional[str] = Field(
        default=None, description="Optional warnings or residual observations."
    )


@dataclass
class ReviewDecision:
    """Outcome of an attempted CAMEL review."""

    weights: Dict[str, float]
    explanation: str
    agentic: bool
    raw_response: str = ""
    failure_reason: Optional[str] = None


class WeightReviewPipeline:
    """Review pipeline that recalculates orchestrator agent weights daily or on demand."""

    SETTINGS_KEY = "dashboard:settings"

    def __init__(self, redis_client, camel_memory=None):
        self.redis = redis_client
        self.camel_memory = camel_memory
        self.prompt = settings.review_prompt_default
        self._review_agent: Optional["ChatAgent"] = None

    async def run(self, trigger: str = "schedule") -> Dict[str, float]:
        """Compute new agent weights and persist them with provenance metadata."""

        await self._refresh_prompt()
        metrics = await self._collect_metrics()
        previous = await get_cached_weights(self.redis) or DEFAULT_WEIGHTS.copy()
        decision = await self._agent_decision(trigger, metrics, previous)

        if decision and decision.agentic and decision.weights:
            weights = decision.weights
            explanation = decision.explanation
            agentic = True
            raw_response = decision.raw_response
            failure_reason = decision.failure_reason
        else:
            weights = self._compute_weights(metrics)
            agentic = False
            failure_reason = None
            raw_response = decision.raw_response if decision else ""
            explanation = (
                decision.explanation
                if decision and decision.explanation
                else "Applied heuristic weighting because the CAMEL reviewer was unavailable."
            )

        snapshot = {
            "weights": weights,
            "previous_weights": previous,
            "generated_at": datetime.utcnow().isoformat(),
            "trigger": trigger,
            "metrics": metrics,
            "prompt": self.prompt,
            "agentic": agentic,
            "ai_explanation": explanation,
            "failure_reason": failure_reason,
            "raw_response": raw_response,
        }

        await self._persist_snapshot(snapshot)
        return weights

    async def _refresh_prompt(self) -> None:
        try:
            dashboard = await self.redis.get_json(self.SETTINGS_KEY) or {}
            self.prompt = dashboard.get("review_prompt", settings.review_prompt_default)
        except Exception:
            self.prompt = settings.review_prompt_default

    async def _collect_metrics(self) -> Dict[str, float]:
        """Gather performance heuristics for each agent from Redis caches."""

        metrics: Dict[str, float] = {}
        signal_raw = await self.redis.lrange("memory:signals", -500, -1)
        counts: Dict[str, int] = {}
        latest_ts: Dict[str, datetime] = {}

        for raw in signal_raw:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            agent_type = payload.get("data", {}).get("agent_type") or payload.get("agent_type")
            if not agent_type:
                continue
            counts[agent_type] = counts.get(agent_type, 0) + 1
            timestamp = payload.get("timestamp") or payload.get("data", {}).get("timestamp")
            if not timestamp:
                continue
            try:
                ts = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            except ValueError:
                ts = None
            if ts and (agent_type not in latest_ts or ts > latest_ts[agent_type]):
                latest_ts[agent_type] = ts

        ticker_perf = await self.redis.get_json("memory:ticker_performance") or {}
        avg_pnl = fmean([perf.get("total_pnl", 0.0) for perf in ticker_perf.values()]) if ticker_perf else 0.0

        for agent_key in DEFAULT_WEIGHTS:
            base_count = counts.get(agent_key, 0)
            staleness_hours = 0.0
            if agent_key in latest_ts:
                age = (datetime.utcnow() - latest_ts[agent_key]).total_seconds() / 3600
                staleness_hours = max(age, 0.0)
            metrics[agent_key] = base_count - (staleness_hours * 0.1) + (avg_pnl * 0.01)

        reward_snapshot = await self.redis.get_json("memory:agent_rewards") or {}
        for agent_key, stats in reward_snapshot.items():
            reward_score = float(stats.get("average_reward", 0.0))
            metrics[agent_key] = metrics.get(agent_key, 0.0) + reward_score * 100.0

        return metrics

    def _compute_weights(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Normalise metrics into weight vector with smoothing toward defaults."""

        blended: Dict[str, float] = {}
        for agent_key, default_weight in DEFAULT_WEIGHTS.items():
            score = metrics.get(agent_key, 0.0)
            adjusted = max(score, 0.0) + default_weight * 10
            blended[agent_key] = adjusted

        total = sum(blended.values())
        if total <= 0:
            return DEFAULT_WEIGHTS.copy()

        normalised = {agent_key: round(value / total, 4) for agent_key, value in blended.items()}
        remainder = 1.0 - sum(normalised.values())
        if abs(remainder) > 1e-6:
            first_key = next(iter(normalised))
            normalised[first_key] = round(normalised[first_key] + remainder, 4)
        return normalised

    async def _agent_decision(
        self,
        trigger: str,
        metrics: Dict[str, float],
        previous: Dict[str, float],
    ) -> ReviewDecision:
        """Attempt to delegate weight selection to a CAMEL ChatAgent."""

        if not self._ensure_agent():
            return ReviewDecision(
                weights={},
                explanation="CAMEL review agent unavailable, falling back to heuristic weights.",
                agentic=False,
                failure_reason="agent_unavailable",
            )

        payload = {
            "trigger": trigger,
            "current_weights": previous,
            "default_weights": DEFAULT_WEIGHTS,
            "metrics": metrics,
            "instructions": self.prompt,
        }

        user_content = (
            f"{self.prompt.strip()}\n\n"
            "Carefully review the supplied agent performance metrics and determine updated weights. "
            "Always respond with JSON only in the following schema:\n"
            '{ "weights": {"TREND": 0.32, "FACT": 0.18, ...}, "explanation": "short rationale", "notes": "optional" }\n'
            "Weights must be non-negative, include every agent key listed, and sum to 1.0.\n"
            "Metrics snapshot:\n```json\n"
            f"{json.dumps(payload, indent=2)}\n```"
        )

        user_message = BaseMessage.make_user_message(role_name="Memory Agent", content=user_content)  # type: ignore[arg-type]

        try:
            response = await self._call_review_agent(user_message)
        except Exception as exc:
            log.warning("Review agent call failed: %s", exc)
            return ReviewDecision(
                weights={},
                explanation="Review agent error; reverting to heuristics.",
                agentic=False,
                failure_reason=str(exc),
            )

        if response is None:
            return ReviewDecision(
                weights={},
                explanation="Review agent unavailable; heuristic weights retained.",
                agentic=False,
                failure_reason="no_response",
            )

        reply_text = ""
        parsed_payload: Optional[Dict[str, Any]] = None

        for msg in response.msgs or []:
            parsed_obj = getattr(msg, "parsed", None)
            if isinstance(parsed_obj, ReviewAgentResponse):
                parsed_payload = parsed_obj.model_dump()
                reply_text = getattr(msg, "content", "") or json.dumps(parsed_payload)
                break
            if isinstance(parsed_obj, dict):
                parsed_payload = parsed_obj
                reply_text = getattr(msg, "content", "") or json.dumps(parsed_payload)
                break
            content = getattr(msg, "content", None)
            if content:
                reply_text = content
                parsed_payload = self._extract_json(content)
                if parsed_payload:
                    break

        if not reply_text and response.msgs:
            reply_text = getattr(response.msgs[-1], "content", "") or ""

        log.bind(pipeline="REVIEW", agentic=True).info(
            "Review agent conversation",
            prompt=user_content,
            reply=reply_text,
            parsed=parsed_payload,
        )

        if not isinstance(parsed_payload, dict):
            return ReviewDecision(
                weights={},
                explanation="Review agent did not provide parseable JSON; heuristic weights retained.",
                agentic=False,
                raw_response=reply_text,
                failure_reason="invalid_json",
            )

        normalised = self._normalise_weights(parsed_payload.get("weights"))
        if not normalised:
            return ReviewDecision(
                weights={},
                explanation="Review agent produced invalid weights; heuristic weights retained.",
                agentic=False,
                raw_response=reply_text,
                failure_reason="invalid_weights",
            )

        explanation = str(
            parsed_payload.get("explanation")
            or parsed_payload.get("summary")
            or "Weights updated by CAMEL reviewer."
        )
        notes = parsed_payload.get("notes")
        failure_reason = str(notes) if isinstance(notes, str) else None
        return ReviewDecision(
            weights=normalised,
            explanation=explanation,
            agentic=True,
            raw_response=reply_text,
            failure_reason=failure_reason,
        )

    def _normalise_weights(self, candidate: Any) -> Dict[str, float]:
        if not isinstance(candidate, dict):
            return {}
        values: Dict[str, float] = {}
        for key in DEFAULT_WEIGHTS.keys():
            try:
                value = float(candidate.get(key, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            values[key] = max(value, 0.0)

        total = sum(values.values())
        if total <= 0:
            return {}
        return {key: round(value / total, 4) for key, value in values.items()}

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        stripped = text.strip()
        if not stripped:
            return None

        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            newline_index = stripped.find("\n")
            if newline_index != -1:
                stripped = stripped[newline_index + 1 :]

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    async def _call_review_agent(self, message: "BaseMessage"):
        if not self._review_agent:
            return None

        loop = asyncio.get_running_loop()
        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_AGENT_ATTEMPTS + 1):
            try:
                def _invoke():
                    return self._review_agent.step(message, response_format=ReviewAgentResponse)

                return await loop.run_in_executor(None, _invoke)
            except TypeError:
                # response_format not supported in this runtime; fall back to plain call.
                return await loop.run_in_executor(None, self._review_agent.step, message)
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.35 * attempt)

        if last_error:
            raise last_error
        return None

    def _ensure_agent(self) -> bool:
        if ChatAgent is None or BaseMessage is None:
            return False
        if self._review_agent is not None:
            return True
        try:
            model = CamelModelFactory.create_task_model()
        except Exception as exc:  # pragma: no cover - optional failure path
            log.warning("Unable to create CAMEL model for review pipeline: %s", exc)
            return False

        system_message = BaseMessage.make_assistant_message(  # type: ignore[arg-type]
            role_name="Agent Weight Reviewer",
            content=(
                "You are the orchestrator weight reviewer for a trading workforce. "
                "You receive aggregated metrics for each specialist (trend, fact, DQN, chart, copytrade, news, risk). "
                "Return JSON only. Always preserve all agent keys and ensure weights sum to one."
            ),
        )

        self._review_agent = ChatAgent(system_message=system_message, model=model)  # type: ignore[arg-type]
        return True

    async def _persist_snapshot(self, snapshot: Dict[str, Any]) -> None:
        await self.redis.set_json(REDIS_REVIEW_KEY, snapshot, expire=settings.review_interval_hours * 3600)
        await self.redis.lpush(REDIS_REVIEW_HISTORY, json.dumps(snapshot))
        await self.redis.ltrim(REDIS_REVIEW_HISTORY, 0, 30)
        await self.redis.set_json(
            REDIS_REVIEW_METADATA,
            {"last_run": snapshot["generated_at"], "trigger": snapshot["trigger"], "weights": snapshot["weights"]},
        )

        try:
            dashboard = await self.redis.get_json(self.SETTINGS_KEY) or {}
            dashboard["latest_review_snapshot"] = snapshot
            dashboard["last_review_update"] = snapshot["generated_at"]
            dashboard.setdefault("review_prompt", self.prompt)
            dashboard.pop("agent_prompts", None)
            dashboard.pop("mcp_overrides", None)
            await self.redis.set_json(self.SETTINGS_KEY, dashboard)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.debug("Unable to persist review snapshot to dashboard settings: %s", exc)

        self._record_memory(snapshot)

    def _record_memory(self, snapshot: Dict[str, Any]) -> None:
        if not self.camel_memory or BaseMessage is None:
            return
        try:
            message = BaseMessage.make_assistant_message(  # type: ignore[arg-type]
                role_name="Agent Weight Reviewer",
                content=(
                    f"Review ({snapshot['trigger']}): agentic={snapshot['agentic']} "
                    f"weights={snapshot['weights']}"
                ),
            )
            role = OpenAIBackendRole.ASSISTANT if OpenAIBackendRole else None
            extra_info = {
                "weights": snapshot["weights"],
                "generated_at": snapshot["generated_at"],
                "agentic": snapshot["agentic"],
                "explanation": snapshot.get("ai_explanation"),
            }
            self.camel_memory.write_record(message, role=role, extra_info=extra_info)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - optional dependency
            log.debug("Unable to persist review outcome to CAMEL memory: %s", exc)


async def get_cached_weights(redis_client) -> Optional[Dict[str, float]]:
    snapshot = await redis_client.get_json(REDIS_REVIEW_KEY)
    if not snapshot:
        return None
    weights = snapshot.get("weights")
    if not weights:
        return None
    return {key: float(value) for key, value in weights.items()}
