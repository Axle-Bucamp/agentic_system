"""
Redis client for caching, pub/sub, and shared state management.
"""
import json
from datetime import datetime
from typing import Any, Optional, Dict, List
import redis.asyncio as aioredis
from redis.asyncio import Redis
from core.config import settings
from core.logging import log


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class RedisClient:
    """Async Redis client wrapper with pub/sub support."""
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.pubsub = None
        
    async def connect(self):
        """Establish connection to Redis."""
        try:
            self.redis = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50
            )
            await self.redis.ping()
            log.info(f"Connected to Redis at {settings.redis_host}:{settings.redis_port}")
        except Exception as e:
            log.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            log.info("Disconnected from Redis")
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        try:
            return await self.redis.get(key)
        except Exception as e:
            log.error(f"Redis GET error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: str, expire: Optional[int] = None):
        """Set value in Redis with optional expiration."""
        try:
            await self.redis.set(key, value, ex=expire)
        except Exception as e:
            log.error(f"Redis SET error for key {key}: {e}")
    
    async def get_json(self, key: str) -> Optional[Dict]:
        """Get JSON value from Redis."""
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                log.error(f"Failed to decode JSON for key {key}")
        return None
    
    async def set_json(self, key: str, value: Dict, expire: Optional[int] = None):
        """Set JSON value in Redis."""
        await self.set(key, json.dumps(value, cls=DateTimeEncoder), expire)
    
    async def delete(self, key: str):
        """Delete key from Redis."""
        try:
            await self.redis.delete(key)
        except Exception as e:
            log.error(f"Redis DELETE error for key {key}: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            log.error(f"Redis EXISTS error for key {key}: {e}")
            return False
    
    async def publish(self, channel: str, message: Dict):
        """Publish message to Redis channel."""
        try:
            await self.redis.publish(channel, json.dumps(message, cls=DateTimeEncoder))
            log.debug(f"Published message to channel {channel}")
        except Exception as e:
            log.error(f"Redis PUBLISH error for channel {channel}: {e}")
    
    async def subscribe(self, *channels: str):
        """Subscribe to Redis channels."""
        try:
            self.pubsub = self.redis.pubsub()
            await self.pubsub.subscribe(*channels)
            log.info(f"Subscribed to channels: {channels}")
            return self.pubsub
        except Exception as e:
            log.error(f"Redis SUBSCRIBE error: {e}")
            raise
    
    async def unsubscribe(self, *channels: str):
        """Unsubscribe from Redis channels."""
        if self.pubsub:
            await self.pubsub.unsubscribe(*channels)
            log.info(f"Unsubscribed from channels: {channels}")
    
    async def get_messages(self):
        """Get messages from subscribed channels."""
        if not self.pubsub:
            raise RuntimeError("Not subscribed to any channels")
        
        async for message in self.pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    yield data
                except json.JSONDecodeError:
                    log.error(f"Failed to decode message: {message['data']}")
    
    async def hset(self, name: str, key: str, value: str):
        """Set hash field."""
        try:
            await self.redis.hset(name, key, value)
        except Exception as e:
            log.error(f"Redis HSET error for {name}:{key}: {e}")
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field."""
        try:
            return await self.redis.hget(name, key)
        except Exception as e:
            log.error(f"Redis HGET error for {name}:{key}: {e}")
            return None
    
    async def hgetall(self, name: str) -> Dict:
        """Get all hash fields."""
        try:
            return await self.redis.hgetall(name)
        except Exception as e:
            log.error(f"Redis HGETALL error for {name}: {e}")
            return {}
    
    async def lpush(self, key: str, *values: str):
        """Push values to list (left)."""
        try:
            await self.redis.lpush(key, *values)
        except Exception as e:
            log.error(f"Redis LPUSH error for key {key}: {e}")
    
    async def rpush(self, key: str, *values: str):
        """Push values to list (right)."""
        try:
            await self.redis.rpush(key, *values)
        except Exception as e:
            log.error(f"Redis RPUSH error for key {key}: {e}")
    
    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        """Get range of list elements."""
        try:
            return await self.redis.lrange(key, start, end)
        except Exception as e:
            log.error(f"Redis LRANGE error for key {key}: {e}")
            return []
    
    async def ltrim(self, key: str, start: int, end: int):
        """Trim list to specified range."""
        try:
            await self.redis.ltrim(key, start, end)
        except Exception as e:
            log.error(f"Redis LTRIM error for key {key}: {e}")


# Global Redis client instance
redis_client = RedisClient()

