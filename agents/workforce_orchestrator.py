"""
CAMEL Workforce Orchestrator

Replaces the traditional OrchestratorAgent with CAMEL Workforce for advanced
multi-agent task orchestration, decomposition, and coordination.
"""
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from core.config import settings
from core.logging import log
from core.models import AgentType, AgentMessage, MessageType
from core.redis_client import RedisClient
from agents.base_agent import BaseAgent
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
        
    async def initialize(self):
        """Initialize the Workforce orchestrator."""
        if not CAMEL_WORKFORCE_AVAILABLE:
            raise ImportError("CAMEL Workforce not installed. Install with: pip install camel-ai")
        
        try:
            # Initialize memory
            self.memory_manager = CamelMemoryManager(
                agent_id="workforce_orchestrator",
                collection_name="workforce_orchestrator_memory"
            )
            
            # Create coordinator agent
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
            await self._initialize_workers()
            
            # Create workforce
            self.workforce = Workforce(
                description="Trading System Workforce - Coordinates specialized workers for trading tasks",
                coordinator_agent=coordinator_agent,
                task_agent=task_agent,
                use_structured_output_handler=True,
                share_memory=True,
            )
            
            # Add workers to workforce
            await self._add_workers_to_workforce()
            
            log.info("Workforce Orchestrator initialized successfully")
            
        except Exception as e:
            log.error(f"Failed to initialize Workforce Orchestrator: {e}")
            raise
    
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
            # Add DQN worker
            dqn_worker = self.workers["dqn"]
            worker_agent = dqn_worker.agent
            if worker_agent:
                self.workforce.add_single_agent_worker(
                    description=dqn_worker.get_description(),
                    worker=worker_agent,
                )
            
            # Add Chart Analysis worker
            chart_worker = self.workers["chart"]
            if chart_worker.agent:
                self.workforce.add_single_agent_worker(
                    description=chart_worker.get_description(),
                    worker=chart_worker.agent,
                )
            
            # Add Risk Assessment worker
            risk_worker = self.workers["risk"]
            if risk_worker.agent:
                self.workforce.add_single_agent_worker(
                    description=risk_worker.get_description(),
                    worker=risk_worker.agent,
                )
            
            # Add Market Research worker
            research_worker = self.workers["research"]
            if research_worker.agent:
                self.workforce.add_single_agent_worker(
                    description=research_worker.get_description(),
                    worker=research_worker.agent,
                )
            
            # Add Trade Execution worker
            execution_worker = self.workers["execution"]
            if execution_worker.agent:
                self.workforce.add_single_agent_worker(
                    description=execution_worker.get_description(),
                    worker=execution_worker.agent,
                )
            
            log.info("Added all workers to workforce")
            
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
        try:
            signal_data = message.payload
            ticker = signal_data.get("ticker")
            action = signal_data.get("action")
            
            if not ticker:
                return
            
            # Create a task for the workforce
            task_description = (
                f"Analyze trading signal for {ticker} with action {action}. "
                f"Get DQN forecast, perform technical analysis, assess risk, "
                f"and if appropriate, execute the trade."
            )
            
            if self.workforce:
                task = Task(content=task_description)
                # Note: process_task may be async or sync depending on CAMEL version
                try:
                    if asyncio.iscoroutinefunction(self.workforce.process_task):
                        result = await self.workforce.process_task(task)
                    else:
                        result = self.workforce.process_task(task)
                except AttributeError:
                    # Fallback: use process_task_async if available
                    result = await self.workforce.process_task_async(task)
                
                log.info(f"Workforce processed task for {ticker}: {result}")
                
                # Store result in memory
                if self.memory_manager:
                    result_message = BaseMessage.make_assistant_message(
                        role_name="Workforce",
                        content=str(result)
                    )
                    self.memory_manager.write_record(result_message)
            
        except Exception as e:
            log.error(f"Error handling signal with workforce: {e}")
    
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
        
        try:
            # Check if trading is paused
            if await self.redis.exists("orchestrator:trading_paused"):
                log.debug("Trading is paused due to risk alert")
                return
            
            # Process each supported asset
            for ticker in settings.supported_assets:
                try:
                    # Create a comprehensive trading task
                    task_description = (
                        f"Evaluate trading opportunities for {ticker}. "
                        f"1. Get DQN forecast and action recommendation. "
                        f"2. Perform technical analysis. "
                        f"3. Assess risk and position sizing. "
                        f"4. If conditions are favorable, execute appropriate trade."
                    )
                    
                    if self.workforce:
                        task = Task(content=task_description)
                        # Note: process_task may be async or sync depending on CAMEL version
                        try:
                            if asyncio.iscoroutinefunction(self.workforce.process_task):
                                result = await self.workforce.process_task(task)
                            else:
                                result = self.workforce.process_task(task)
                        except AttributeError:
                            # Fallback: use process_task_async if available
                            result = await self.workforce.process_task_async(task)
                        
                        log.info(f"Workforce cycle result for {ticker}: {result}")
                        
                except Exception as e:
                    log.error(f"Error processing cycle for {ticker}: {e}")
            
        except Exception as e:
            log.error(f"Workforce Orchestrator cycle error: {e}")
    
    def get_cycle_interval(self) -> int:
        """Run every 5 minutes."""
        return 300
    
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

