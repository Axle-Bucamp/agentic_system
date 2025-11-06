"""
DQN Worker for CAMEL Workforce

Handles DQN prediction tasks including getting forecasts and action recommendations.
"""
from typing import Dict, Any, List, Optional
from core.config import settings
from core.logging import log
from core.camel_tools.mcp_forecasting_toolkit import MCPForecastingToolkit
from core.memory.camel_memory_manager import CamelMemoryManager
from core.models.camel_models import CamelModelFactory

try:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    log.warning("CAMEL not available. Install with: pip install camel-ai")


class DQNWorker:
    """Worker for DQN prediction and forecasting tasks."""
    
    def __init__(self, agent_id: str = "dqn_worker"):
        """
        Initialize DQN worker.
        
        Args:
            agent_id: Unique identifier for the worker
        """
        if not CAMEL_AVAILABLE:
            raise ImportError("CAMEL not installed")
        
        self.agent_id = agent_id
        self.forecasting_toolkit: Optional[MCPForecastingToolkit] = None
        self.memory_manager: Optional[CamelMemoryManager] = None
        self.agent: Optional[ChatAgent] = None
    
    async def initialize(self):
        """Initialize the worker with tools and memory."""
        try:
            # Initialize forecasting toolkit
            self.forecasting_toolkit = MCPForecastingToolkit()
            await self.forecasting_toolkit.initialize()
            
            # Initialize memory
            self.memory_manager = CamelMemoryManager(
                agent_id=self.agent_id,
                collection_name=f"dqn_worker_{self.agent_id}"
            )
            
            # Create agent with tools
            model = CamelModelFactory.create_worker_model()
            system_message = BaseMessage.make_assistant_message(
                role_name="DQN Worker",
                content=(
                    "You are a DQN prediction worker specializing in forecasting and "
                    "action recommendations for cryptocurrency trading. "
                    "You use MCP forecasting API tools to get predictions and forecasts. "
                    "You can handle multiple types of tasks: getting stock forecasts, "
                    "getting action recommendations, listing available tickers, and getting metrics."
                )
            )
            
            tools = self.forecasting_toolkit.get_all_tools()
            
            self.agent = ChatAgent(
                system_message=system_message,
                model=model,
                tools=tools,
            )
            
            # Attach memory to agent
            self.agent.memory = self.memory_manager.memory
            
            log.info(f"Initialized DQN worker: {self.agent_id}")
            
        except Exception as e:
            log.error(f"Failed to initialize DQN worker: {e}")
            raise
    
    async def process_task(self, task_description: str) -> Dict[str, Any]:
        """
        Process a task using the DQN worker.
        
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
            log.error(f"Error processing task in DQN worker: {e}")
            return {
                "success": False,
                "worker_id": self.agent_id,
                "error": str(e)
            }
    
    def get_description(self) -> str:
        """Get worker description for task assignment."""
        return (
            "A worker for DQN prediction tasks. Can get stock forecasts, "
            "action recommendations, list available tickers, and retrieve metrics."
        )

