"""Agent weight review pipeline leveraging auto-enhancement heuristics."""
from __future__ import annotations

import json
from datetime import datetime
from statistics import fmean
from typing import Dict, Optional

from core.config import settings
from core.logging import log
from core.models import AgentType

REDIS_REVIEW_KEY = "orchestrator:agent_weights"
REDIS_REVIEW_HISTORY = "orchestrator:agent_weights_history"
REDIS_REVIEW_METADATA = "orchestrator:agent_weights_meta"


DEFAULT_WEIGHTS: Dict[str, float] = {
    AgentType.TREND.value: 0.35,
    AgentType.FACT.value: 0.20,
    AgentType.DQN.value: 0.15,
    AgentType.CHART.value: 0.10,
    AgentType.COPYTRADE.value: 0.10,
    AgentType.NEWS.value: 0.05,
    AgentType.RISK.value: 0.05,
}


class WeightReviewPipeline:
    """Review pipeline that recalculates orchestrator agent weights daily or on demand."""

    SETTINGS_KEY = "dashboard:settings"

    def __init__(self, redis_client, camel_memory=None):
        self.redis = redis_client
        self.camel_memory = camel_memory
        self.prompt = settings.review_prompt_default

    async def run(self, trigger: str = "schedule") -> Dict[str, float]:
        """Compute new agent weights and persist them with provenance metadata."""
        await self._refresh_prompt()
        metrics = await self._collect_metrics()
        weights = self._compute_weights(metrics)
        snapshot = {
            "weights": weights,
            "generated_at": datetime.utcnow().isoformat(),
            "trigger": trigger,
            "metrics": metrics,
            "prompt": self.prompt,
        }
        await self.redis.set_json(REDIS_REVIEW_KEY, snapshot, expire=settings.review_interval_hours * 3600)
        await self.redis.lpush(REDIS_REVIEW_HISTORY, json.dumps(snapshot))
        await self.redis.ltrim(REDIS_REVIEW_HISTORY, 0, 30)
        await self.redis.set_json(
            REDIS_REVIEW_METADATA,
            {"last_run": snapshot["generated_at"], "trigger": trigger, "weights": weights},
        )

        log.info("Review pipeline produced agent weights: %s", weights)

        if self.camel_memory:
            try:
                from camel.messages import BaseMessage  # pylint: disable=import-error
                from camel.types import OpenAIBackendRole  # pylint: disable=import-error

                message = BaseMessage.make_assistant_message(
                    role_name="Agent Weight Reviewer",
                    content=f"Trigger: {trigger}. Updated weights: {weights}",
                )
                self.camel_memory.write_record(
                    message,
                    role=OpenAIBackendRole.ASSISTANT,
                    extra_info={"weights": weights, "trigger": trigger},
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                log.debug("Unable to persist review outcome to CAMEL memory: %s", exc)

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
            if timestamp:
                try:
                    ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    ts = None
                if ts:
                    if agent_type not in latest_ts or ts > latest_ts[agent_type]:
                        latest_ts[agent_type] = ts

        ticker_perf = await self.redis.get_json("memory:ticker_performance") or {}
        avg_pnl = fmean([perf.get("total_pnl", 0.0) for perf in ticker_perf.values()]) if ticker_perf else 0.0

        for agent_key in DEFAULT_WEIGHTS:
            base_count = counts.get(agent_key, 0)
            staleness = 0.0
            if agent_key in latest_ts:
                age = (datetime.utcnow() - latest_ts[agent_key]).total_seconds() / 3600
                staleness = max(age, 0.0)
            metrics[agent_key] = base_count - (staleness * 0.1) + (avg_pnl * 0.01)

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
        # Ensure stable ordering and sum to 1
        remainder = 1.0 - sum(normalised.values())
        if abs(remainder) > 1e-6:
            first_key = next(iter(normalised))
            normalised[first_key] = round(normalised[first_key] + remainder, 4)
        return normalised


async def get_cached_weights(redis_client) -> Optional[Dict[str, float]]:
    snapshot = await redis_client.get_json(REDIS_REVIEW_KEY)
    if not snapshot:
        return None
    weights = snapshot.get("weights")
    if not weights:
        return None
    return {key: float(value) for key, value in weights.items()}
