"""
Trading Workforce Society configuration.

Defines a CAMEL Workforce tailored for the trading system.  The society
instantiates the coordinator, task agent, and worker agent blueprints so
that we can reuse them both in the long running orchestrator and inside
pipeline executions.
"""

from __future__ import annotations

from typing import Any, List, Optional

from camel.agents import ChatAgent
from camel.logger import get_logger  # type: ignore[import]
from camel.messages import BaseMessage

try:  # pragma: no cover - optional dependency
    from camel.societies.workforce import Workforce
except ImportError:  # pragma: no cover
    Workforce = None  # type: ignore

from core.models.camel_models import CamelModelFactory
from core.camel_runtime.registries import build_default_tools
from agents.workforce_workers import (
    ChartAnalysisWorker,
    DQNWorker,
    MarketResearchWorker,
    RiskAssessmentWorker,
    TradeExecutionWorker,
)

logger = get_logger(__name__)


class TradingWorkforceSociety:
    """
    Factory for CAMEL Workforces specialised for trading tasks.

    The workforce wires the coordinator, task, and worker chat agents with
    the shared toolkit registry, ensuring every worker has access to the
    FunctionTool set aligned to the official CAMEL tooling API.
    """

    def __init__(self) -> None:
        self._coordinator: Optional[ChatAgent] = None
        self._task_agent: Optional[ChatAgent] = None
        self._workforce: Optional[Workforce] = None
        self._workers: List[Any] = []

    async def build(self) -> Workforce:
        """Initialise (or return) a fully configured workforce instance."""
        if self._workforce:
            return self._workforce

        if Workforce is None:
            raise RuntimeError(
                "CAMEL workforce module is unavailable. Ensure `camel-ai[workforce]` is installed "
                "and compatible with this runtime."
            )

        tools = await build_default_tools()

        coord_model = CamelModelFactory.create_coordinator_model()
        task_model = CamelModelFactory.create_task_model()

        workforce = Workforce(
            description="Trading System Workforce",
            coordinator_agent_kwargs={"model": coord_model, "tools": tools},
            task_agent_kwargs={"model": task_model, "tools": tools},
        )

        coordinator_msg = BaseMessage.make_assistant_message(
            role_name="Coordinator",
            content=(
                "You coordinate a multi-agent trading desk tasked with producing disciplined trading plans.\n"
                "Always follow the mission pipeline:\n"
                "1. Validate the incoming signal (ticker, interval, wallet exposure).\n"
                "2. Sequence specialised workers: DQN forecast, Technical analysis, Market research, "
                "Risk review, Trade execution simulation.\n"
                "3. Ensure each worker calls their tools and returns structured evidence with citations.\n"
                "4. Inspect guidry-cloud telemetry via browse_url/get_guidry_cloud_api_stats whenever forecasting reliability "
                "is in question and propagate warnings to the final decision.\n"
                "5. Synthesize the final action with clear justification, confidence, and execution instructions."
            ),
        )
        workforce.coordinator_agent.orig_sys_message = coordinator_msg
        workforce.coordinator_agent.system_message = coordinator_msg

        task_msg = BaseMessage.make_assistant_message(
            role_name="Task Decomposer",
            content=(
                "You break trading objectives into ordered subtasks for the workforce.\n"
                "For every request, generate a numbered plan with the following phases: "
                "(1) Forecast & policy, (2) Technical analysis, (3) Market research, "
                "(4) Risk evaluation, (5) Trade execution preview, (6) Final coordinator synthesis.\n"
                "Provide each worker with the signal context, required tool calls, expected outputs, "
                "and success criteria. Highlight when guidry-cloud statistics should be examined before proceeding."
            ),
        )
        workforce.task_agent.orig_sys_message = task_msg
        workforce.task_agent.system_message = task_msg

        self._workers = [
            DQNWorker(agent_id="dqn_worker_primary"),
            ChartAnalysisWorker(agent_id="chart_worker_primary"),
            RiskAssessmentWorker(agent_id="risk_worker_primary"),
            MarketResearchWorker(agent_id="research_worker_primary"),
            TradeExecutionWorker(agent_id="execution_worker_primary"),
        ]

        for worker in self._workers:
            await worker.initialize()
            if not getattr(worker, "agent", None):
                logger.warning(
                    "Worker %s has no agent after initialisation; skipping",
                    worker.__class__.__name__,
                )
                continue
            workforce.add_single_agent_worker(
                description=worker.get_description(),
                worker=worker.agent,
                enable_workflow_memory=True,
            )

        try:
            loaded = getattr(workforce, "load_workflow_memories", None)
            if callable(loaded):
                summary = loaded()
                logger.debug(
                    "Loaded workforce workflow memories: %s",
                    summary,
                )
        except Exception as exc:
            logger.debug("Unable to load workflow memories: %s", exc)

        self._workforce = workforce
        logger.info(
            "Trading workforce society initialised with %d workers",
            len(self._workers),
        )
        return workforce


society_factory = TradingWorkforceSociety()

