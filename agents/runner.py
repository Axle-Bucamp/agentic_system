"""
Agent runner - Starts the appropriate agent based on environment variable.
"""
import asyncio
import signal
import os
from core.redis_client import RedisClient
from core.logging import log
from agents.memory_agent import MemoryAgent
from agents.dqn_agent import DQNAgent
from agents.chart_agent import ChartAgent
from agents.risk_agent import RiskAgent
from agents.news_agent import NewsAgent
from agents.copytrade_agent import CopyTradeAgent
from agents.orchestrator import OrchestratorAgent


async def main():
    """Main entry point for agent runner."""
    agent_name = os.getenv("AGENT_NAME", "").lower()
    
    if not agent_name:
        log.error("AGENT_NAME environment variable not set")
        return
    
    log.info(f"Starting {agent_name} agent...")
    
    # Create Redis client
    redis_client = RedisClient()
    
    # Create appropriate agent
    agent_map = {
        "memory": MemoryAgent,
        "dqn": DQNAgent,
        "chart": ChartAgent,
        "risk": RiskAgent,
        "news": NewsAgent,
        "copytrade": CopyTradeAgent,
        "orchestrator": OrchestratorAgent
    }
    
    agent_class = agent_map.get(agent_name)
    
    if not agent_class:
        log.error(f"Unknown agent name: {agent_name}")
        return
    
    agent = agent_class(redis_client)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        log.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(agent.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start agent
    try:
        await agent.start()
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")
    except Exception as e:
        log.error(f"Agent error: {e}")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())

