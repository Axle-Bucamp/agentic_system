"""Agent weight review pipeline leveraging auto-enhancement heuristics."""
from __future__ import annotations

import json
from datetime import datetime
from statistics import fmean
from typing import Dict, Optional, Tuple

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

DEFAULT_AGENT_PROMPTS: Dict[str, str] = {
    AgentType.TREND.value: "Blend DQN distributions with technical indicators to surface trend direction, cite confidence bands, and call out volatility spikes.",
    AgentType.FACT.value: "Summarize macro and micro catalysts with sources, include sentiment scores, and highlight disagreements in coverage.",
    AgentType.DQN.value: "Report forecast distribution (sell/hold/buy) with expected move and note when model confidence is low.",
    AgentType.CHART.value: "Explain key indicators (RSI, MACD, Bollinger) in two sentences and align with risk tolerances.",
    AgentType.COPYTRADE.value: "Validate copy candidates with on-chain behavior, recent success rate, and risk tier reminders.",
    AgentType.NEWS.value: "Prioritize regulatory, protocol, and market-moving headlines; attach deep-search references and quick sentiment tags.",
    AgentType.RISK.value: "Return updated stop-loss, drawdown, and exposure metrics; insist on mitigation if limits breached.",
}

DEFAULT_MCP_SETTINGS: Dict[str, Dict[str, object]] = {
    "news": {
        "deep_search_enabled": True,
        "sentiment_enabled": True,
        "sources": settings.deep_search_sources,
        "refresh_minutes": 60,
    }
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
        prompts, mcp_overrides = await self._update_prompts_and_mcp(metrics)
        snapshot = {
            "weights": weights,
            "generated_at": datetime.utcnow().isoformat(),
            "trigger": trigger,
            "metrics": metrics,
            "prompt": self.prompt,
            "agent_prompts": prompts,
            "mcp_overrides": mcp_overrides,
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

    async def _update_prompts_and_mcp(self, metrics: Dict[str, float]) -> Tuple[Dict[str, str], Dict[str, Dict[str, object]]]:
        """Refresh agent prompt guidance and MCP overrides based on latest metrics."""
        dashboard = await self.redis.get_json(self.SETTINGS_KEY) or {}
        agent_prompts: Dict[str, str] = dashboard.get("agent_prompts", {}).copy()

        for agent_key, base_prompt in DEFAULT_AGENT_PROMPTS.items():
            score = metrics.get(agent_key, 0.0)
            agent_prompts[agent_key] = self._compose_agent_prompt(base_prompt, score)

        mcp_overrides: Dict[str, Dict[str, object]] = dashboard.get("mcp_overrides", {}).copy()
        news_settings = mcp_overrides.get("news", {}).copy()
        news_score = metrics.get(AgentType.NEWS.value, 0.0)
        news_settings["deep_search_enabled"] = news_score < 2 or news_settings.get("deep_search_enabled", True)
        news_settings["sentiment_enabled"] = True
        news_settings["sources"] = news_settings.get("sources") or settings.deep_search_sources
        news_settings["refresh_minutes"] = self._derive_refresh_window(news_score)
        mcp_overrides["news"] = news_settings

        dashboard.update(
            {
                "agent_prompts": agent_prompts,
                "mcp_overrides": mcp_overrides,
                "last_review_update": datetime.utcnow().isoformat(),
            }
        )
        dashboard.setdefault("review_prompt", self.prompt)
        await self.redis.set_json(self.SETTINGS_KEY, dashboard)
        return agent_prompts, mcp_overrides

    @staticmethod
    def _compose_agent_prompt(base_prompt: str, score: float) -> str:
        if score < 1:
            modifier = "Performance trending soft; tighten reasoning, demand extra validation, and call out data gaps."
        elif score > 8:
            modifier = "Performance strong; maintain current strategy but flag complacency risks."
        else:
            modifier = "Keep balanced perspective; reinforce collaboration with fusion agent."
        return f"{base_prompt} {modifier}"

    @staticmethod
    def _derive_refresh_window(score: float) -> int:
        if score < 1:
            return 30
        if score > 6:
            return 120
        return 60


async def get_cached_weights(redis_client) -> Optional[Dict[str, float]]:
    snapshot = await redis_client.get_json(REDIS_REVIEW_KEY)
    if not snapshot:
        return None
    weights = snapshot.get("weights")
    if not weights:
        return None
    return {key: float(value) for key, value in weights.items()}
