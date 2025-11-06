"""
Market Research Worker for CAMEL Workforce

Handles market research tasks including news analysis and sentiment analysis.
"""
from typing import Dict, Any, Optional
from core.config import settings
from core.logging import log
from core.memory.camel_memory_manager import CamelMemoryManager
from core.models.camel_models import CamelModelFactory

try:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    log.warning("CAMEL not available. Install with: pip install camel-ai")


class MarketResearchWorker:
    """Worker for market research tasks."""
    
    def __init__(self, agent_id: str = "market_research_worker"):
        """
        Initialize market research worker.
        
        Args:
            agent_id: Unique identifier for the worker
        """
        if not CAMEL_AVAILABLE:
            raise ImportError("CAMEL not installed")
        
        self.agent_id = agent_id
        self.memory_manager: Optional[CamelMemoryManager] = None
        self.agent: Optional[ChatAgent] = None
    
    async def initialize(self):
        """Initialize the worker."""
        try:
            # Initialize memory
            self.memory_manager = CamelMemoryManager(
                agent_id=self.agent_id,
                collection_name=f"research_worker_{self.agent_id}"
            )
            
            # Create agent
            model = CamelModelFactory.create_worker_model()
            system_message = BaseMessage.make_assistant_message(
                role_name="Market Research Worker",
                content=(
                    "You are a market research worker specializing in analyzing "
                    "cryptocurrency news and sentiment. You can analyze news articles, "
                    "evaluate market sentiment, and provide research insights. "
                    "You can handle multiple types of tasks: news analysis and sentiment analysis."
                )
            )
            
            # Note: In a full implementation, news/sentiment tools would be added here
            
            self.agent = ChatAgent(
                system_message=system_message,
                model=model,
            )
            
            # Attach memory to agent
            self.agent.memory = self.memory_manager.memory
            
            log.info(f"Initialized Market Research worker: {self.agent_id}")
            
        except Exception as e:
            log.error(f"Failed to initialize Market Research worker: {e}")
            raise
    
    async def process_task(self, task_description: str) -> Dict[str, Any]:
        """
        Process a market research task.
        
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
            log.error(f"Error processing task in Market Research worker: {e}")
            return {
                "success": False,
                "worker_id": self.agent_id,
                "error": str(e)
            }
    
    def get_description(self) -> str:
        """Get worker description for task assignment."""
        return (
            "A worker for market research tasks. Can analyze news, evaluate sentiment, "
            "and provide research insights."
        )

