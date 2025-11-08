"""
Base agent class for all specialized agents in the system.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime
from core.models import AgentType, AgentMessage, MessageType
from core.redis_client import RedisClient
from core.logging import log
from core.config import settings


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, agent_type: AgentType, redis_client: RedisClient):
        self.agent_type = agent_type
        self.redis = redis_client
        self.running = False
        self.pubsub = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
    @abstractmethod
    async def initialize(self):
        """Initialize agent-specific resources."""
        pass
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message and optionally return a response."""
        pass
    
    @abstractmethod
    async def run_cycle(self):
        """Execute one cycle of agent-specific logic."""
        pass
    
    async def start(self):
        """Start the agent."""
        agent_logger = log.bind(
            agent=self.agent_type.value,
            cluster=settings.cluster_name,
            instance=settings.agent_instance_id,
        )
        agent_logger.info("Starting %s agent...", self.agent_type.value)
        
        # Connect to Redis
        await self.redis.connect()
        
        # Initialize agent
        await self.initialize()
        
        # Subscribe to relevant channels
        channels = self.get_subscribed_channels()
        if channels:
            self.pubsub = await self.redis.subscribe(*channels)
            agent_logger.info(
                "%s subscribed to channels: %s",
                self.agent_type.value,
                channels,
            )
        
        # Start heartbeat
        self.running = True
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Start message listener
        listener_task = asyncio.create_task(self._message_listener())
        
        # Start agent cycle
        cycle_task = asyncio.create_task(self._cycle_loop())
        
        # Wait for tasks
        try:
            await asyncio.gather(listener_task, cycle_task, self.heartbeat_task)
        except asyncio.CancelledError:
            log.info(f"{self.agent_type.value} agent tasks cancelled")
        except Exception as e:
            log.error(f"{self.agent_type.value} agent error: {e}")
            raise
    
    async def stop(self):
        """Stop the agent."""
        agent_logger = log.bind(
            agent=self.agent_type.value,
            cluster=settings.cluster_name,
            instance=settings.agent_instance_id,
        )
        agent_logger.info("Stopping %s agent...", self.agent_type.value)
        self.running = False
        
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        if self.pubsub:
            await self.redis.unsubscribe(*self.get_subscribed_channels())
        
        await self.redis.disconnect()
        agent_logger.info("%s agent stopped", self.agent_type.value)
    
    def get_subscribed_channels(self) -> list:
        """Get list of channels this agent subscribes to."""
        # All agents subscribe to broadcast channel
        channels = ["agent:broadcast"]
        
        # Add agent-specific channel
        channels.append(f"agent:{self.agent_type.value.lower()}")
        
        return channels
    
    async def _message_listener(self):
        """Listen for messages on subscribed channels."""
        if not self.pubsub:
            return
        
        agent_logger = log.bind(
            agent=self.agent_type.value,
            cluster=settings.cluster_name,
            instance=settings.agent_instance_id,
        )
        agent_logger.info("%s message listener started", self.agent_type.value)
        
        try:
            async for raw_message in self.redis.get_messages():
                try:
                    message = AgentMessage(**raw_message)
                    
                    # Skip messages from self
                    if message.sender == self.agent_type:
                        continue
                    
                    # Process message
                    agent_logger.debug(
                        "%s received %s from %s",
                        self.agent_type.value,
                        message.message_type.value,
                        message.sender.value,
                    )
                    response = await self.process_message(message)
                    
                    # Send response if any
                    if response:
                        await self.send_message(response)
                        
                except Exception as e:
                    agent_logger.error("%s error processing message: %s", self.agent_type.value, e)
        except asyncio.CancelledError:
            agent_logger.info("%s message listener cancelled", self.agent_type.value)
        except Exception as e:
            agent_logger.error("%s message listener error: %s", self.agent_type.value, e)
    
    async def _cycle_loop(self):
        """Run agent cycle periodically."""
        agent_logger = log.bind(
            agent=self.agent_type.value,
            cluster=settings.cluster_name,
            instance=settings.agent_instance_id,
        )
        agent_logger.info("%s cycle loop started", self.agent_type.value)
        
        try:
            while self.running:
                try:
                    await self.run_cycle()
                except Exception as e:
                    agent_logger.error("%s cycle error: %s", self.agent_type.value, e)
                
                # Wait before next cycle
                await asyncio.sleep(self.get_cycle_interval())
        except asyncio.CancelledError:
            agent_logger.info("%s cycle loop cancelled", self.agent_type.value)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages."""
        try:
            while self.running:
                await self.send_heartbeat()
                await asyncio.sleep(settings.agent_heartbeat_interval)
        except asyncio.CancelledError:
            log.bind(
                agent=self.agent_type.value,
                cluster=settings.cluster_name,
                instance=settings.agent_instance_id,
            ).debug("%s heartbeat loop cancelled", self.agent_type.value)
    
    async def send_heartbeat(self):
        """Send heartbeat message."""
        message = AgentMessage(
            message_type=MessageType.AGENT_HEARTBEAT,
            sender=self.agent_type,
            payload={
                "status": "alive",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        await self.send_message(message, channel="agent:heartbeat")
    
    async def send_message(self, message: AgentMessage, channel: Optional[str] = None):
        """Send message to specific channel or broadcast."""
        if channel is None:
            # Determine channel based on recipient
            if message.recipient:
                channel = f"agent:{message.recipient.value.lower()}"
            else:
                channel = "agent:broadcast"
        
        await self.redis.publish(channel, message.dict())
    
    async def send_signal(self, signal_data: Dict[str, Any]):
        """Send a signal to the orchestrator."""
        message = AgentMessage(
            message_type=MessageType.SIGNAL_GENERATED,
            sender=self.agent_type,
            recipient=AgentType.ORCHESTRATOR,
            payload=signal_data
        )
        await self.send_message(message)
        log.bind(
            agent=self.agent_type.value,
            cluster=settings.cluster_name,
            instance=settings.agent_instance_id,
        ).info("%s sent signal to orchestrator", self.agent_type.value)
    
    def get_cycle_interval(self) -> int:
        """Get interval between agent cycles in seconds."""
        return settings.get_agent_cycle_seconds(self.agent_type)
    
    async def get_shared_state(self, key: str) -> Optional[Dict]:
        """Get shared state from Redis."""
        return await self.redis.get_json(f"state:{key}")
    
    async def set_shared_state(self, key: str, value: Dict, expire: Optional[int] = None):
        """Set shared state in Redis."""
        await self.redis.set_json(f"state:{key}", value, expire)
    
    async def get_portfolio(self) -> Optional[Dict]:
        """Get current portfolio state."""
        return await self.get_shared_state("portfolio")
    
    async def get_market_data(self, ticker: str) -> Optional[Dict]:
        """Get latest market data for a ticker."""
        return await self.redis.get_json(f"market:{ticker}")

