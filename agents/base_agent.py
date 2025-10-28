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
        log.info(f"Starting {self.agent_type.value} agent...")
        
        # Connect to Redis
        await self.redis.connect()
        
        # Initialize agent
        await self.initialize()
        
        # Subscribe to relevant channels
        channels = self.get_subscribed_channels()
        if channels:
            self.pubsub = await self.redis.subscribe(*channels)
            log.info(f"{self.agent_type.value} subscribed to channels: {channels}")
        
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
        log.info(f"Stopping {self.agent_type.value} agent...")
        self.running = False
        
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        if self.pubsub:
            await self.redis.unsubscribe(*self.get_subscribed_channels())
        
        await self.redis.disconnect()
        log.info(f"{self.agent_type.value} agent stopped")
    
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
        
        log.info(f"{self.agent_type.value} message listener started")
        
        try:
            async for raw_message in self.redis.get_messages():
                try:
                    message = AgentMessage(**raw_message)
                    
                    # Skip messages from self
                    if message.sender == self.agent_type:
                        continue
                    
                    # Process message
                    log.debug(f"{self.agent_type.value} received {message.message_type.value} from {message.sender.value}")
                    response = await self.process_message(message)
                    
                    # Send response if any
                    if response:
                        await self.send_message(response)
                        
                except Exception as e:
                    log.error(f"{self.agent_type.value} error processing message: {e}")
        except asyncio.CancelledError:
            log.info(f"{self.agent_type.value} message listener cancelled")
        except Exception as e:
            log.error(f"{self.agent_type.value} message listener error: {e}")
    
    async def _cycle_loop(self):
        """Run agent cycle periodically."""
        log.info(f"{self.agent_type.value} cycle loop started")
        
        try:
            while self.running:
                try:
                    await self.run_cycle()
                except Exception as e:
                    log.error(f"{self.agent_type.value} cycle error: {e}")
                
                # Wait before next cycle
                await asyncio.sleep(self.get_cycle_interval())
        except asyncio.CancelledError:
            log.info(f"{self.agent_type.value} cycle loop cancelled")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages."""
        try:
            while self.running:
                await self.send_heartbeat()
                await asyncio.sleep(settings.agent_heartbeat_interval)
        except asyncio.CancelledError:
            log.debug(f"{self.agent_type.value} heartbeat loop cancelled")
    
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
        log.info(f"{self.agent_type.value} sent signal to orchestrator")
    
    def get_cycle_interval(self) -> int:
        """Get interval between agent cycles in seconds."""
        # Default to 60 seconds, can be overridden by subclasses
        return 60
    
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

