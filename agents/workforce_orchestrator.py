"""
CAMEL Workforce Orchestrator

Replaces the traditional OrchestratorAgent with CAMEL Workforce for advanced
multi-agent task orchestration, decomposition, and coordination.
"""
import asyncio
import contextlib
import time
import uuid
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.config import settings
from core.logging import log
from core.models import AgentType, AgentMessage, MessageType
from core.redis_client import RedisClient
from agents.base_agent import BaseAgent
from core import asset_registry
from agents.workforce_workers import (
    DQNWorker,
    ChartAnalysisWorker,
    RiskAssessmentWorker,
    MarketResearchWorker,
    TradeExecutionWorker,
)
from core.models.camel_models import CamelModelFactory
from core.memory.camel_memory_manager import CamelMemoryManager

try:
    from camel.societies.workforce import Workforce
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from camel.tasks import Task
    CAMEL_WORKFORCE_AVAILABLE = True
except ImportError:
    CAMEL_WORKFORCE_AVAILABLE = False
    log.warning("CAMEL Workforce not available. Install with: pip install camel-ai")


class WorkforceOrchestratorAgent(BaseAgent):
    """
    Workforce-based orchestrator agent using CAMEL Workforce.
    
    This replaces the traditional OrchestratorAgent with a more sophisticated
    multi-agent system that can decompose tasks, assign them to specialized workers,
    and coordinate their execution.
    """
    
    def __init__(self, redis_client: RedisClient):
        super().__init__(AgentType.ORCHESTRATOR, redis_client)
        self.workforce: Optional[Workforce] = None
        self.workers: Dict[str, Any] = {}
        self.memory_manager: Optional[CamelMemoryManager] = None
        self.running = False
        self._last_wallet_plan_timestamp: Optional[str] = None
        self._workforce_available: bool = True
        self._workforce_queue: "asyncio.Queue[Tuple[Task, asyncio.Future]]" = asyncio.Queue()
        self._workforce_worker_task: Optional[asyncio.Task] = None
        self._workforce_task_timeout: float = float(getattr(settings, "workforce_task_timeout_seconds", 120))
        
    async def initialize(self):
        """Initialize the Workforce orchestrator."""
        if not CAMEL_WORKFORCE_AVAILABLE:
            raise ImportError("CAMEL Workforce not installed. Install with: pip install camel-ai")
        
        try:
            log.info("Initializing Workforce Orchestrator...")
            # Initialize memory
            log.debug("Initializing memory manager...")
            self.memory_manager = CamelMemoryManager(
                agent_id="workforce_orchestrator",
                collection_name="workforce_orchestrator_memory"
            )
            log.debug("Memory manager initialized successfully")
            
            # Create coordinator agent
            log.debug("Creating coordinator agent...")
            coordinator_model = CamelModelFactory.create_coordinator_model()
            coordinator_agent = ChatAgent(
                system_message=BaseMessage.make_assistant_message(
                    role_name="Task Coordinator",
                    content=(
                        "You are a task coordinator for a trading system. "
                        "You assign tasks to specialized workers based on their capabilities. "
                        "Available workers: DQN worker (forecasts, recommendations), "
                        "Chart Analysis worker (technical analysis), "
                        "Risk Assessment worker (risk evaluation), "
                        "Market Research worker (news, sentiment), "
                        "Trade Execution worker (executing trades)."
                    )
                ),
                model=coordinator_model,
            )
            
            # Create task decomposition agent
            log.debug("Creating task decomposition agent...")
            task_model = CamelModelFactory.create_task_model()
            task_agent = ChatAgent(
                system_message=BaseMessage.make_assistant_message(
                    role_name="Task Decomposer",
                    content=(
                        "You decompose high-level trading tasks into smaller, "
                        "manageable subtasks that can be assigned to specialized workers. "
                        "Each subtask should be self-contained and clear."
                    )
                ),
                model=task_model,
            )
            
            # Initialize workers
            log.debug("Initializing workers...")
            await self._initialize_workers()
            log.info(f"Initialized {len(self.workers)} workers successfully")
            
            # Create workforce
            log.debug("Creating Workforce instance...")
            # Note: CAMEL Workforce API - try minimal initialization first
            try:
                # Try with just description (minimal parameters)
                self.workforce = Workforce(
                    description="Trading System Workforce - Coordinates specialized workers for trading tasks"
                )
                log.info("Workforce created with minimal initialization")
                
                # Try to set coordinator and task agent if attributes exist
                if hasattr(self.workforce, 'coordinator'):
                    self.workforce.coordinator = coordinator_agent
                    log.debug("Set coordinator agent on workforce")
                elif hasattr(self.workforce, 'coordinator_agent'):
                    self.workforce.coordinator_agent = coordinator_agent
                    log.debug("Set coordinator_agent on workforce")
                
                if hasattr(self.workforce, 'task_agent'):
                    self.workforce.task_agent = task_agent
                    log.debug("Set task_agent on workforce")
                elif hasattr(self.workforce, 'task_decomposition_agent'):
                    self.workforce.task_decomposition_agent = task_agent
                    log.debug("Set task_decomposition_agent on workforce")
                    
            except TypeError as e:
                log.warning(f"Workforce initialization failed: {e}, trying with coordinator_agent parameter")
                try:
                    # Try with coordinator_agent parameter name
                    self.workforce = Workforce(
                        description="Trading System Workforce - Coordinates specialized workers for trading tasks",
                        coordinator_agent=coordinator_agent,
                        task_agent=task_agent,
                    )
                    log.info("Workforce created with coordinator_agent parameter")
                except Exception as e2:
                    log.error(f"All Workforce initialization attempts failed: {e2}")
                    log.error(f"Error details: {type(e2).__name__}: {e2}")
                    raise
            
            # Add workers to workforce
            await self._add_workers_to_workforce()
            
            log.info("Workforce Orchestrator initialized successfully")
            self._ensure_workforce_worker_running()
            
        except Exception as e:
            self._workforce_available = False
            log.error(f"Failed to initialize Workforce Orchestrator: {e}")
            log.warning("Workforce orchestration disabled; continuing without CAMEL workforce support.")
    
    async def _initialize_workers(self):
        """Initialize all worker agents."""
        try:
            # Initialize DQN worker
            dqn_worker_instance = DQNWorker(agent_id="dqn_worker_1")
            await dqn_worker_instance.initialize()
            self.workers["dqn"] = dqn_worker_instance
            
            # Initialize Chart Analysis worker
            chart_worker = ChartAnalysisWorker(agent_id="chart_worker_1")
            await chart_worker.initialize()
            self.workers["chart"] = chart_worker
            
            # Initialize Risk Assessment worker
            risk_worker = RiskAssessmentWorker(agent_id="risk_worker_1")
            await risk_worker.initialize()
            self.workers["risk"] = risk_worker
            
            # Initialize Market Research worker
            research_worker = MarketResearchWorker(agent_id="research_worker_1")
            await research_worker.initialize()
            self.workers["research"] = research_worker
            
            # Initialize Trade Execution worker
            execution_worker = TradeExecutionWorker(agent_id="execution_worker_1")
            await execution_worker.initialize()
            self.workers["execution"] = execution_worker
            
            log.info(f"Initialized {len(self.workers)} workers")
            
        except Exception as e:
            log.error(f"Failed to initialize workers: {e}")
            raise
    
    async def _add_workers_to_workforce(self):
        """Add workers to the workforce."""
        if not self.workforce:
            return
        
        try:
            added_count = 0
            # Add DQN worker
            dqn_worker = self.workers["dqn"]
            worker_agent = dqn_worker.agent
            if worker_agent:
                self.workforce.add_single_agent_worker(
                    description=dqn_worker.get_description(),
                    worker=worker_agent,
                )
                added_count += 1
                log.debug(f"Added DQN worker to workforce: {dqn_worker.get_description()}")
            else:
                log.warning("DQN worker agent is None, skipping")
            
            # Add Chart Analysis worker
            chart_worker = self.workers["chart"]
            if chart_worker.agent:
                self.workforce.add_single_agent_worker(
                    description=chart_worker.get_description(),
                    worker=chart_worker.agent,
                )
                added_count += 1
                log.debug(f"Added Chart Analysis worker to workforce: {chart_worker.get_description()}")
            else:
                log.warning("Chart Analysis worker agent is None, skipping")
            
            # Add Risk Assessment worker
            risk_worker = self.workers["risk"]
            if risk_worker.agent:
                self.workforce.add_single_agent_worker(
                    description=risk_worker.get_description(),
                    worker=risk_worker.agent,
                )
                added_count += 1
                log.debug(f"Added Risk Assessment worker to workforce: {risk_worker.get_description()}")
            else:
                log.warning("Risk Assessment worker agent is None, skipping")
            
            # Add Market Research worker
            research_worker = self.workers["research"]
            if research_worker.agent:
                self.workforce.add_single_agent_worker(
                    description=research_worker.get_description(),
                    worker=research_worker.agent,
                )
                added_count += 1
                log.debug(f"Added Market Research worker to workforce: {research_worker.get_description()}")
            else:
                log.warning("Market Research worker agent is None, skipping")
            
            # Add Trade Execution worker
            execution_worker = self.workers["execution"]
            if execution_worker.agent:
                self.workforce.add_single_agent_worker(
                    description=execution_worker.get_description(),
                    worker=execution_worker.agent,
                )
                added_count += 1
                log.debug(f"Added Trade Execution worker to workforce: {execution_worker.get_description()}")
            else:
                log.warning("Trade Execution worker agent is None, skipping")
            
            log.info(f"Successfully added {added_count}/{len(self.workers)} workers to workforce")
            
        except Exception as e:
            log.error(f"Failed to add workers to workforce: {e}")
            raise
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages."""
        try:
            if message.message_type == MessageType.SIGNAL_GENERATED:
                # Handle signal by creating a task for the workforce
                await self._handle_signal_with_workforce(message)
            elif message.message_type == MessageType.RISK_ALERT:
                await self._handle_risk_alert(message)
            
        except Exception as e:
            log.error(f"Workforce Orchestrator error processing message: {e}")
        
        return None
    
    async def _handle_signal_with_workforce(self, message: AgentMessage):
        """Handle signal by processing it through the workforce."""
        if not self._workforce_available:
            log.debug("Workforce unavailable; skipping workforce signal processing.")
            return

        try:
            signal_data = message.payload
            ticker = signal_data.get("ticker")
            action = signal_data.get("action")
            
            if not ticker:
                return
            
            # Log AI decision start
            decision_id = str(uuid.uuid4())
            log.info(f"[AI_DECISION] Starting decision {decision_id} for {ticker} with action {action}")
            
            # Create a task for the workforce
            task_description = (
                f"Analyze trading signal for {ticker} with action {action}. "
                f"Get DQN forecast, perform technical analysis, assess risk, "
                f"and if appropriate, execute the trade."
            )
            
            # Store decision metadata
            decision_metadata = {
                "decision_id": decision_id,
                "ticker": ticker,
                "action": action,
                "task_description": task_description,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "processing",
                "steps": []
            }
            
            # Store in Redis for UI access
            await self.redis.set_json(f"ai_decision:{decision_id}", decision_metadata)
            
            task = Task(content=task_description)

            log.info(f"[AI_DECISION] Task created: {task_description}")
            decision_metadata["steps"].append({
                "step": "task_created",
                "description": task_description,
                "timestamp": datetime.utcnow().isoformat()
            })

            await self._summarize_wallet_plan()

            result = await self._process_task_with_workforce(task)
            if result is None:
                log.warning("[AI_DECISION] Workforce unavailable; skipping CAMEL processing for {}", ticker)
                decision_metadata["steps"].append({
                    "step": "workforce_disabled",
                    "description": "Workforce processing skipped; using baseline heuristic outputs.",
                    "timestamp": datetime.utcnow().isoformat()
                })
                decision_metadata["status"] = "degraded"
                await self.redis.set_json(f"ai_decision:{decision_id}", decision_metadata)
                return

            if isinstance(result, dict):
                result_text = result.get("text") or ""
                if not result_text and result.get("messages"):
                    result_text = "\n".join(result.get("messages", []))
                if not result_text:
                    result_text = str(result.get("raw", "")) or "[no agent response]"
                result_messages = result.get("messages", [])
                success_flag = result.get("success", True)
                if not success_flag:
                    decision_metadata["status"] = "degraded"
                    if result.get("error"):
                        decision_metadata["error"] = result["error"]
                else:
                    decision_metadata["status"] = "completed"
                if "raw" in result:
                    decision_metadata["result_raw"] = str(result.get("raw"))
            else:
                result_text = str(result)
                result_messages = []
                decision_metadata["status"] = "completed"

            log.info(f"[AI_DECISION] Workforce processed task for {ticker}: {result_text}")
            decision_metadata["steps"].append({
                "step": "task_processed",
                "result": result_text,
                "messages": result_messages,
                "timestamp": datetime.utcnow().isoformat()
            })

            decision_metadata["result"] = result_text
            decision_metadata["result_text"] = result_text
            decision_metadata["completed_at"] = datetime.utcnow().isoformat()
            await self.redis.set_json(f"ai_decision:{decision_id}", decision_metadata)

            if self.memory_manager:
                result_message = BaseMessage.make_assistant_message(
                    role_name="Workforce",
                    content=result_text
                )
                self.memory_manager.write_record(result_message)
                log.info(f"[AI_DECISION] Decision {decision_id} stored in memory")
            
        except Exception as e:
            log.error(f"[AI_DECISION] Error handling signal with workforce: {e}")
            if 'decision_id' in locals():
                decision_metadata["status"] = "error"
                decision_metadata["error"] = str(e)
                decision_metadata["error_at"] = datetime.utcnow().isoformat()
                await self.redis.set_json(f"ai_decision:{decision_id}", decision_metadata)
    
    async def _handle_risk_alert(self, message: AgentMessage):
        """Handle risk alert from risk agent."""
        try:
            payload = message.payload
            risk_level = payload.get("risk_level")
            warnings = payload.get("warnings", [])
            
            log.warning(f"Risk alert received: {risk_level} - {warnings}")
            
            # If critical, pause trading
            if risk_level == "CRITICAL":
                await self.redis.set("orchestrator:trading_paused", "1", expire=3600)
                log.error("CRITICAL risk level - Trading paused for 1 hour")
                
        except Exception as e:
            log.error(f"Error handling risk alert: {e}")
    
    async def run_cycle(self):
        """Run periodic decision-making cycle."""
        log.debug("Workforce Orchestrator running cycle...")

        if not self._workforce_available:
            log.debug("Workforce orchestrator disabled; skipping cycle execution.")
            return
        
        try:
            await self._summarize_wallet_plan()

            # Check if trading is paused
            if await self.redis.exists("orchestrator:trading_paused"):
                log.debug("Trading is paused due to risk alert")
                return
            
            # Process each supported asset
            for ticker in asset_registry.get_assets():
                try:
                    # Create a comprehensive trading task
                    task_description = (
                        f"Evaluate trading opportunities for {ticker}. "
                        f"1. Get DQN forecast and action recommendation. "
                        f"2. Perform technical analysis. "
                        f"3. Assess risk and position sizing. "
                        f"4. If conditions are favorable, execute appropriate trade."
                    )
                    
                    task = Task(content=task_description)
                    result = await self._process_task_with_workforce(task)
                    if result is None:
                        log.debug("Workforce disabled; skipping automated analysis for {}", ticker)
                        continue

                    log.bind(agent="WORKFORCE", ticker=ticker).info("Cycle result: %s", result)
                        
                except Exception as e:
                    log.error(f"Error processing cycle for {ticker}: {e}")
            
        except Exception as e:
            log.error(f"Workforce Orchestrator cycle error: {e}")
    
    def get_cycle_interval(self) -> int:
        """Run every 5 minutes."""
        return 300

    async def _summarize_wallet_plan(self, force: bool = False):
        """Create and persist an LLM-driven summary of the latest wallet plan."""
        try:
            plan = await self.redis.get_json("orchestrator:wallet_plan")
        except Exception as exc:  # pragma: no cover - defensive logging
            log.debug("Unable to fetch wallet plan for workforce summary: {}", exc)
            return

        if not plan:
            return

        plan_id = plan.get("generated_at")
        if not force and plan_id and plan_id == self._last_wallet_plan_timestamp:
            return

        allocations = plan.get("allocations", {})
        total_value = plan.get("total_value_usdc", 0)
        high_risk_assets = [
            ticker
            for ticker, entry in allocations.items()
            if isinstance(entry, dict) and str(entry.get("risk_level", "")).upper() in {"HIGH", "CRITICAL"}
        ]

        plan_brief = json.dumps(allocations, indent=2)
        instructions = (
            "You are the strategic portfolio analyst for a multi-agent trading desk. \n"
            "Summarize the wallet balancing plan using bullet points. \n"
            "Highlight: 1) Key allocations with rationale, 2) High-risk assets, 3) Recommended stop-loss windows, 4) Gas fee impact. \n"
            "Close with a one sentence executive summary." 
        )

        summary_text: Optional[str] = None
        if self.workforce and self._workforce_available:
            task_body = (
                f"Plan timestamp: {plan_id}\n"
                f"Total value (USDC): {total_value}\n"
                f"High risk tickers: {', '.join(high_risk_assets) if high_risk_assets else 'none'}\n"
                f"Allocations JSON:\n{plan_brief}\n\n{instructions}"
            )
            task = Task(content=task_body)
            result = await self._process_task_with_workforce(task)
            if result is not None:
                summary_text = str(result)
                log.info("[WORKFORCE] Wallet plan summary generated via workforce")

        if not summary_text:
            summary_text = (
                "Automated summary unavailable. Review portfolio allocations, risk levels, and stop-loss guidance manually."
            )

        summary_payload = {
            "generated_at": plan_id,
            "summary": summary_text,
            "high_risk_assets": high_risk_assets,
            "total_value_usdc": total_value,
        }

        try:
            await self.redis.set_json("orchestrator:wallet_plan_summary", summary_payload)
            log.bind(PORTFOLIO_PLAN=True).info("Wallet plan summary refreshed: {}", summary_payload)
            self._last_wallet_plan_timestamp = plan_id
        except Exception as exc:  # pragma: no cover - persistence failure
            log.error("Failed to persist wallet plan summary: {}", exc)
    
    async def stop(self):
        """Stop the orchestrator and cleanup."""
        # Disconnect workers
        for worker_name, worker in self.workers.items():
            try:
                if hasattr(worker, 'forecasting_toolkit') and worker.forecasting_toolkit:
                    if hasattr(worker.forecasting_toolkit, 'forecasting_client'):
                        await worker.forecasting_toolkit.forecasting_client.disconnect()
                if hasattr(worker, 'dex_toolkit') and worker.dex_toolkit:
                    if hasattr(worker.dex_toolkit, 'dex_client'):
                        await worker.dex_toolkit.dex_client.disconnect()
            except Exception as e:
                log.warning(f"Error disconnecting {worker_name} worker: {e}")
        
        # Call parent stop method
        await super().stop()

        if self._workforce_worker_task:
            self._workforce_worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._workforce_worker_task
            self._workforce_worker_task = None

    def _ensure_workforce_worker_running(self) -> None:
        if not self.workforce or not self._workforce_available:
            return

        if self._workforce_worker_task and not self._workforce_worker_task.done():
            return

        if self._workforce_worker_task and self._workforce_worker_task.done():
            self._workforce_worker_task = None

        try:
            self._workforce_worker_task = asyncio.create_task(
                self._run_workforce_queue(),
                name="workforce-queue-worker",
            )
            log.debug("Started workforce queue worker task")
        except RuntimeError as exc:
            log.warning("Unable to start workforce queue worker: %s", exc)

    @staticmethod
    def _is_workforce_busy_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "workforce is running" in message or "already running" in message

    def _should_disable_workforce(self, exc: Exception) -> bool:
        message = str(exc).lower()
        markers = (
            "models/gemini",
            "not found",
            "api key",
            "unauthorized",
            "quota",
        )
        if self._is_workforce_busy_error(exc):
            return False
        return any(marker in message for marker in markers)

    async def _process_task_with_workforce(self, task: Task) -> Optional[Dict[str, Any]]:
        """Queue the task for workforce processing with robust retries and fallbacks."""
        if not self.workforce or not self._workforce_available:
            return self._format_workforce_result(
                {"success": False, "error": "workforce unavailable"}
            )

        self._ensure_workforce_worker_running()

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._workforce_queue.put((task, future))

        try:
            result = await asyncio.wait_for(future, timeout=self._workforce_task_timeout)
        except asyncio.TimeoutError:
            future.cancel()
            log.bind(agent="WORKFORCE").warning(
                "Workforce task timed out after %.1fs; returning degraded result",
                self._workforce_task_timeout,
            )
            return self._format_workforce_result(
                {
                    "success": False,
                    "error": f"workforce timeout after {self._workforce_task_timeout:.0f}s",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.bind(agent="WORKFORCE").exception("Unexpected error awaiting workforce result")
            return self._format_workforce_result({"success": False, "error": str(exc)})

        return result if isinstance(result, dict) else self._format_workforce_result(result)

    async def _run_workforce_queue(self) -> None:
        log.debug("Workforce queue worker loop started")
        while True:
            task, future = await self._workforce_queue.get()

            if future.cancelled():
                self._workforce_queue.task_done()
                continue

            try:
                result = await self._execute_workforce_task(task)
            except Exception as exc:  # pragma: no cover - defensive logging
                log.bind(agent="WORKFORCE").exception("Unhandled error in queue worker: {}", exc)
                result = self._format_workforce_result({"success": False, "error": str(exc)})

            if not future.cancelled():
                future.set_result(result)

            self._workforce_queue.task_done()

    async def _execute_workforce_task(self, task: Task) -> Dict[str, Any]:
        if not self.workforce or not self._workforce_available:
            return self._format_workforce_result(
                {"success": False, "error": "workforce unavailable"}
            )

        log.bind(agent="WORKFORCE").debug("Processing task payload: %s", task.content[:200])

        busy_deadline = time.monotonic() + max(self._workforce_task_timeout, 10.0)
        delay = 0.5
        last_exception: Optional[Exception] = None

        attempt = 0
        while time.monotonic() <= busy_deadline:
            attempt += 1
            try:
                raw_result = await self._invoke_workforce(task)
                return self._format_workforce_result(raw_result)
            except Exception as exc:
                last_exception = exc
                log.bind(agent="WORKFORCE", attempt=attempt).warning(
                    "Workforce task execution failed: {}", exc
                )

                if self._is_workforce_busy_error(exc):
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.5, 5.0)
                    continue

                if self._should_disable_workforce(exc):
                    self._workforce_available = False
                    log.bind(agent="WORKFORCE").warning(
                        "Disabling workforce after repeated model failures. Falling back to rule-based orchestration."
                    )
                break
        else:
            log.bind(agent="WORKFORCE").warning(
                "Workforce busy window exceeded %.1fs; degrading result", self._workforce_task_timeout
            )

        error_payload = {
            "success": False,
            "error": str(last_exception) if last_exception else "Unknown workforce error",
        }
        return self._format_workforce_result(error_payload)

    async def _invoke_workforce(self, task: Task) -> Any:
        if not self.workforce:
            raise RuntimeError("workforce not initialised")

        async_processor = getattr(self.workforce, "process_task_async", None)
        if callable(async_processor):
            return await async_processor(task)

        processor = getattr(self.workforce, "process_task", None)
        if callable(processor):
            if asyncio.iscoroutinefunction(processor):
                return await processor(task)
            return await asyncio.to_thread(processor, task)

        raise RuntimeError("workforce handler unavailable")

    @staticmethod
    def _format_workforce_result(raw: Any) -> Dict[str, Any]:
        """Normalise CAMEL workforce responses into a consistent dictionary with textual output."""
        success = True
        messages: List[str] = []
        text: str = ""
        error_text: Optional[str] = None

        if isinstance(raw, dict):
            success = raw.get("success", True)
            messages = [str(msg) for msg in raw.get("messages", []) if msg is not None]
            text = raw.get("text") or raw.get("response") or raw.get("result") or raw.get("content", "")
            if not text and messages:
                text = "\n".join(messages)
            error_text = raw.get("error")
        elif isinstance(raw, list):
            messages = [str(item) for item in raw]
            text = "\n".join(messages)
        elif raw is None:
            success = False
            text = ""
        else:
            text = str(raw)

        if not text:
            if error_text:
                text = str(error_text)
            elif messages:
                text = "\n".join(messages)
            else:
                text = "[no agent response]"

        if not success:
            context_message = error_text or text
            log.bind(agent="WORKFORCE").warning("Workforce worker reported failure: {}", context_message)
            if isinstance(context_message, str) and "tool" in context_message.lower():
                log.bind(agent="WORKFORCE").warning("Workforce worker reported missing tool: {}", context_message)

        formatted: Dict[str, Any] = {
            "success": success,
            "text": text,
            "messages": messages,
            "raw": raw,
        }
        if error_text:
            formatted["error"] = error_text
        return formatted

