"""
Chart Analysis Worker for CAMEL Workforce

Handles technical analysis tasks including RSI, MACD, and Bollinger Bands calculations.
"""
from typing import Dict, Any, Optional
from core.config import settings
from core.logging import log
from core.memory.camel_memory_manager import CamelMemoryManager
from core.models.camel_models import CamelModelFactory
from core.camel_tools.market_data_toolkit import MarketDataToolkit
from core.camel_tools.guidry_stats_toolkit import GuidryStatsToolkit

try:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    log.warning("CAMEL not available. Install with: pip install camel-ai")


class ChartAnalysisWorker:
    """Worker for technical analysis tasks."""

    def __init__(self, agent_id: str = "chart_analysis_worker"):
        """
        Initialize chart analysis worker.

        Args:
            agent_id: Unique identifier for the worker
        """
        if not CAMEL_AVAILABLE:
            raise ImportError("CAMEL not installed")

        self.agent_id = agent_id
        self.memory_manager: Optional[CamelMemoryManager] = None
        self.market_data_toolkit: Optional[MarketDataToolkit] = None
        self.stats_toolkit: Optional[GuidryStatsToolkit] = None
        self.agent: Optional[ChatAgent] = None

    async def initialize(self):
        """Initialize the worker."""
        try:
            # Initialize memory
            self.memory_manager = CamelMemoryManager(
                agent_id=self.agent_id,
                collection_name=f"chart_worker_{self.agent_id}"
            )

            # Initialise market data toolkit for indicator retrieval
            self.market_data_toolkit = MarketDataToolkit()
            await self.market_data_toolkit.initialize()

            self.stats_toolkit = GuidryStatsToolkit()
            await self.stats_toolkit.initialize()

            tools = self.market_data_toolkit.get_all_tools()
            tools.extend(self.stats_toolkit.get_all_tools())

            # Create agent
            model = CamelModelFactory.create_worker_model()
            system_message = BaseMessage.make_assistant_message(
                role_name="Chart Analysis Worker",
                content=(
                    "You are the Technical Analysis Specialist. Execute every task in stages:\n"
                    "1. Confirm ticker, interval, and desired momentum horizon. Clarify missing context.\n"
                    "2. Inspect guidry-cloud stats via get_guidry_cloud_api_stats; note recent outages that could skew indicators.\n"
                    "3. Fetch OHLC and indicator data using market tools; compute RSI, MACD, Bollinger, and volume trends.\n"
                    "4. Interpret indicator confluence to label momentum (bullish, bearish, ranging) and quantify confidence.\n"
                    "5. Surface actionable insights, including invalid data or gaps that the coordinator must address."
                )
            )

            self.agent = ChatAgent(
                system_message=system_message,
                model=model,
                tools=tools,
            )

            # Attach memory to agent
            self.agent.memory = self.memory_manager.memory

            log.info(f"Initialized Chart Analysis worker: {self.agent_id}")

        except Exception as e:
            log.error(f"Failed to initialize Chart Analysis worker: {e}")
            raise

    async def process_task(self, task_description: str) -> Dict[str, Any]:
        """
        Process a technical analysis task.

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
            log.error(f"Error processing task in Chart Analysis worker: {e}")
            return {
                "success": False,
                "worker_id": self.agent_id,
                "error": str(e)
            }

    def get_description(self) -> str:
        """Get worker description for task assignment."""
        return (
            "A worker for technical analysis tasks. Can calculate RSI, MACD, "
            "Bollinger Bands, and retrieve market metadata to inform decisions."
        )

