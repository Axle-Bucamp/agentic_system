"""
Market Research Worker for CAMEL Workforce

Handles market research tasks including news analysis and sentiment analysis.
"""
from typing import Dict, Any, Optional
from core.config import settings
from core.logging import log
from core.memory.camel_memory_manager import CamelMemoryManager
from core.models.camel_models import CamelModelFactory
from core.camel_tools.crypto_tools import CryptoTools
from core.camel_tools.guidry_stats_toolkit import GuidryStatsToolkit
from core.camel_tools.playwright_toolkit import PlaywrightToolkit

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
        self.crypto_tools: Optional[CryptoTools] = None
        self.stats_toolkit: Optional[GuidryStatsToolkit] = None
        self.playwright_toolkit: Optional[PlaywrightToolkit] = None
        self.agent: Optional[ChatAgent] = None

    async def initialize(self):
        """Initialize the worker."""
        try:
            # Initialize memory
            self.memory_manager = CamelMemoryManager(
                agent_id=self.agent_id,
                collection_name=f"research_worker_{self.agent_id}"
            )

            # Initialise crypto toolkit for trend/sentiment lookups
            self.crypto_tools = CryptoTools()
            await self.crypto_tools.initialize()

            self.stats_toolkit = GuidryStatsToolkit()
            await self.stats_toolkit.initialize()

            self.playwright_toolkit = PlaywrightToolkit()
            try:
                await self.playwright_toolkit.initialize()
            except ImportError as exc:
                log.warning("Playwright toolkit unavailable: %s", exc)
                self.playwright_toolkit = None

            tools = self.crypto_tools.get_all_tools()
            if self.stats_toolkit:
                tools.extend(self.stats_toolkit.get_all_tools())
            if self.playwright_toolkit:
                tools.extend(self.playwright_toolkit.get_all_tools())

            # Create agent
            model = CamelModelFactory.create_worker_model()
            system_message = BaseMessage.make_assistant_message(
                role_name="Market Research Worker",
                content=(
                    "You are the Market Research & Narrative Specialist. For each request:\n"
                    "1. Clarify the asset, timeframe, and desired narrative (bullish/bearish catalysts).\n"
                    "2. Review guidry-cloud telemetry with get_guidry_cloud_api_stats to understand if forecasting data may be stale.\n"
                    "3. Use research tools to aggregate news, social sentiment, and macro context. Validate sources and cite them.\n"
                    "   When external validation is required, browse primary sources using Playwright tools (browse_url) headlessly.\n"
                    "4. Produce concise insights that highlight catalysts, risks, sentiment skew, and data quality flags.\n"
                    "5. Return structured key points for downstream fusion, calling out any contradictions or missing data."
                )
            )

            self.agent = ChatAgent(
                system_message=system_message,
                model=model,
                tools=tools,
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
            "A worker for market research tasks. Can analyse news, evaluate sentiment, "
            "and surface cryptocurrency trend intelligence using live tool calls."
        )

