"""
Risk Assessment Worker for CAMEL Workforce

Handles risk evaluation tasks including position sizing and drawdown analysis.
"""
from typing import Dict, Any, Optional
from core.config import settings
from core.logging import log
from core.memory.camel_memory_manager import CamelMemoryManager
from core.models.camel_models import CamelModelFactory
from core.camel_tools.dex_trading_toolkit import DEXTradingToolkit
from core.camel_tools.guidry_stats_toolkit import GuidryStatsToolkit

try:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    log.warning("CAMEL not available. Install with: pip install camel-ai")


class RiskAssessmentWorker:
    """Worker for risk assessment tasks."""
    
    def __init__(self, agent_id: str = "risk_assessment_worker"):
        """
        Initialize risk assessment worker.
        
        Args:
            agent_id: Unique identifier for the worker
        """
        if not CAMEL_AVAILABLE:
            raise ImportError("CAMEL not installed")
        
        self.agent_id = agent_id
        self.memory_manager: Optional[CamelMemoryManager] = None
        self.dex_toolkit: Optional[DEXTradingToolkit] = None
        self.stats_toolkit: Optional[GuidryStatsToolkit] = None
        self.agent: Optional[ChatAgent] = None
    
    async def initialize(self):
        """Initialize the worker with tools."""
        try:
            # Initialize memory
            self.memory_manager = CamelMemoryManager(
                agent_id=self.agent_id,
                collection_name=f"risk_worker_{self.agent_id}"
            )
            
            # Initialize DEX toolkit for portfolio access
            self.dex_toolkit = DEXTradingToolkit()
            await self.dex_toolkit.initialize()

            self.stats_toolkit = GuidryStatsToolkit()
            await self.stats_toolkit.initialize()

            # Create agent with tools
            model = CamelModelFactory.create_worker_model()
            system_message = BaseMessage.make_assistant_message(
                role_name="Risk Assessment Specialist",
                content=(
                    "You guard portfolio safety. Execute the following routine:\n"
                    "1. Verify portfolio context (exposure, balances, open positions). Clarify unknowns.\n"
                    "2. Inspect guidry-cloud stats via get_guidry_cloud_api_stats to understand forecasting reliability; "
                    "mirror any degradation in your risk guidance.\n"
                    "3. Use DEX tools to fetch wallet state, unrealised PnL, and current allocations. "
                    "Stress-test target trades against risk limits.\n"
                    "4. Quantify risk metrics (VAR, drawdown, stop-loss) and articulate guardrails.\n"
                    "5. Report clear go/no-go guidance including maximum size, stop levels, and monitoring actions."
                )
            )

            tools = self.dex_toolkit.get_all_tools()
            tools.extend(self.stats_toolkit.get_all_tools())
            
            self.agent = ChatAgent(
                system_message=system_message,
                model=model,
                tools=tools,
            )
            
            # Attach memory to agent
            self.agent.memory = self.memory_manager.memory
            
            log.info(f"Initialized Risk Assessment worker: {self.agent_id}")
            
        except Exception as e:
            log.error(f"Failed to initialize Risk Assessment worker: {e}")
            raise
    
    async def process_task(self, task_description: str) -> Dict[str, Any]:
        """
        Process a risk assessment task.
        
        Args:
            task_description: Description of the task to perform
            
        Returns:
            Task result dictionary
        """
        if not self.agent:
            raise RuntimeError("Worker not initialized")
        
        try:
            user_message = BaseMessage.make_user_message(
                role_name="Task Coordinator",
                content=task_description
            )
            
            response = self.agent.step(user_message)
            
            # Store in memory
            from camel.types import OpenAIBackendRole
            self.memory_manager.write_record(user_message, role=OpenAIBackendRole.USER)
            if response.msgs:
                for msg in response.msgs:
                    self.memory_manager.write_record(
                        msg,
                        role=OpenAIBackendRole.ASSISTANT
                    )
            
            return {
                "success": True,
                "worker_id": self.agent_id,
                "response": response.msgs[0].content if response.msgs else "No response",
                "messages": [msg.content for msg in response.msgs] if response.msgs else []
            }
            
        except Exception as e:
            log.error(f"Error processing task in Risk Assessment worker: {e}")
            return {
                "success": False,
                "worker_id": self.agent_id,
                "error": str(e)
            }
    
    def get_description(self) -> str:
        """Get worker description for task assignment."""
        return (
            "A worker for risk assessment tasks. Can evaluate portfolio risk, "
            "calculate position sizes, assess drawdowns, and provide risk alerts. "
            "This is a unique capability."
        )

