"""
Trade Execution Worker for CAMEL Workforce

Handles trade execution tasks via the DEX simulator.
"""
from typing import Dict, Any, Optional
from core.config import settings
from core.logging import log
from core.memory.camel_memory_manager import CamelMemoryManager
from core.models.camel_models import CamelModelFactory
from core.camel_tools.dex_trading_toolkit import DEXTradingToolkit

try:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    log.warning("CAMEL not available. Install with: pip install camel-ai")


class TradeExecutionWorker:
    """Worker for trade execution tasks."""
    
    def __init__(self, agent_id: str = "trade_execution_worker"):
        """
        Initialize trade execution worker.
        
        Args:
            agent_id: Unique identifier for the worker
        """
        if not CAMEL_AVAILABLE:
            raise ImportError("CAMEL not installed")
        
        self.agent_id = agent_id
        self.memory_manager: Optional[CamelMemoryManager] = None
        self.dex_toolkit: Optional[DEXTradingToolkit] = None
        self.agent: Optional[ChatAgent] = None
    
    async def initialize(self):
        """Initialize the worker with DEX tools."""
        try:
            # Initialize memory
            self.memory_manager = CamelMemoryManager(
                agent_id=self.agent_id,
                collection_name=f"execution_worker_{self.agent_id}"
            )
            
            # Initialize DEX toolkit
            self.dex_toolkit = DEXTradingToolkit()
            await self.dex_toolkit.initialize()
            
            # Create agent with tools
            model = CamelModelFactory.create_worker_model()
            system_message = BaseMessage.make_assistant_message(
                role_name="Trade Execution Worker",
                content=(
                    "You are a trade execution worker specializing in executing trades "
                    "on the DEX simulator. You can buy and sell assets, check portfolio status, "
                    "and get current prices. You execute trades based on decisions from other workers. "
                    "This is your unique capability - only you can execute trades."
                )
            )
            
            tools = self.dex_toolkit.get_all_tools()
            
            self.agent = ChatAgent(
                system_message=system_message,
                model=model,
                tools=tools,
            )
            
            # Attach memory to agent
            self.agent.memory = self.memory_manager.memory
            
            log.info(f"Initialized Trade Execution worker: {self.agent_id}")
            
        except Exception as e:
            log.error(f"Failed to initialize Trade Execution worker: {e}")
            raise
    
    async def process_task(self, task_description: str) -> Dict[str, Any]:
        """
        Process a trade execution task.
        
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
            log.error(f"Error processing task in Trade Execution worker: {e}")
            return {
                "success": False,
                "worker_id": self.agent_id,
                "error": str(e)
            }
    
    def get_description(self) -> str:
        """Get worker description for task assignment."""
        return (
            "A worker for trade execution tasks. Can buy and sell assets on the DEX simulator, "
            "check portfolio status, and get current prices. This is a unique capability."
        )

