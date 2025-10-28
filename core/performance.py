"""
Performance optimization utilities for the Agentic Trading System.
"""
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union
from functools import wraps, lru_cache
from datetime import datetime, timedelta
import weakref
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

from core.logging import log
from core.config import settings

T = TypeVar('T')

class CacheManager:
    """Manages caching for improved performance."""
    
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.ttl = {}
        self.default_ttl = default_ttl
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() > self.ttl.get(key, 0):
            self.delete(key)
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        self.cache[key] = value
        self.ttl[key] = time.time() + (ttl or self.default_ttl)
        
        # Cleanup if needed
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup()
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        self.cache.pop(key, None)
        self.ttl.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.ttl.clear()
    
    def _cleanup(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, expire_time in self.ttl.items()
            if current_time > expire_time
        ]
        
        for key in expired_keys:
            self.delete(key)
        
        self.last_cleanup = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "keys": list(self.cache.keys()),
            "hit_rate": getattr(self, 'hits', 0) / max(getattr(self, 'requests', 1), 1)
        }

class ConnectionPool:
    """Manages connection pooling for external services."""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections = deque()
        self.active_connections = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> Any:
        """Acquire a connection from the pool."""
        async with self.lock:
            if self.connections:
                return self.connections.popleft()
            elif self.active_connections < self.max_connections:
                self.active_connections += 1
                return await self._create_connection()
            else:
                # Wait for a connection to become available
                while not self.connections:
                    await asyncio.sleep(0.01)
                return self.connections.popleft()
    
    async def release(self, connection: Any) -> None:
        """Release a connection back to the pool."""
        async with self.lock:
            self.connections.append(connection)
    
    async def _create_connection(self) -> Any:
        """Create a new connection (to be implemented by subclasses)."""
        raise NotImplementedError

class BatchProcessor:
    """Processes operations in batches for better performance."""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batch = []
        self.last_flush = time.time()
        self.lock = asyncio.Lock()
        self.processor = None
    
    def set_processor(self, processor: Callable[[List[Any]], None]):
        """Set the batch processing function."""
        self.processor = processor
    
    async def add(self, item: Any) -> None:
        """Add item to batch."""
        async with self.lock:
            self.batch.append(item)
            
            # Flush if batch is full or time interval passed
            if (len(self.batch) >= self.batch_size or 
                time.time() - self.last_flush > self.flush_interval):
                await self._flush()
    
    async def _flush(self) -> None:
        """Flush current batch."""
        if not self.batch or not self.processor:
            return
        
        batch_to_process = self.batch.copy()
        self.batch.clear()
        self.last_flush = time.time()
        
        try:
            await self.processor(batch_to_process)
        except Exception as e:
            log.error(f"Batch processing error: {e}")
            # Re-add items to batch for retry
            self.batch.extend(batch_to_process)

class AsyncLimiter:
    """Limits concurrent async operations."""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def acquire(self):
        """Acquire a permit."""
        return await self.semaphore.acquire()
    
    def release(self):
        """Release a permit."""
        self.semaphore.release()
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()

class PerformanceMonitor:
    """Monitors and optimizes performance."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.thresholds = {
            'api_response_time': 1.0,  # seconds
            'trade_execution_time': 5.0,  # seconds
            'agent_processing_time': 0.5,  # seconds
        }
    
    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a performance metric."""
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': time.time()
        })
        
        # Keep only last 1000 records
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]
        
        # Check thresholds
        threshold = self.thresholds.get(metric_name)
        if threshold and value > threshold:
            log.warning(f"Performance threshold exceeded: {metric_name} = {value}s (threshold: {threshold}s)")
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        values = [m['value'] for m in self.metrics[metric_name]]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else 0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {
            metric: self.get_metric_stats(metric)
            for metric in self.metrics.keys()
        }

def timed_async(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to time async function execution."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            log.debug(f"{func.__name__} took {duration:.3f}s")
    return wrapper

def timed_sync(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to time sync function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            log.debug(f"{func.__name__} took {duration:.3f}s")
    return wrapper

def cached_async(ttl: int = 300):
    """Decorator to cache async function results."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = {}
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from args and kwargs
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            # Check cache
            if cache_key in cache:
                cached_result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache[cache_key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator

def retry_async(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry async functions on failure."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = delay * (backoff ** attempt)
                        log.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        log.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator

class ResourceManager:
    """Manages system resources efficiently."""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.resources = weakref.WeakSet()
    
    def register_resource(self, resource: Any) -> None:
        """Register a resource for cleanup."""
        self.resources.add(resource)
    
    async def cleanup_resources(self) -> None:
        """Cleanup all registered resources."""
        for resource in list(self.resources):
            if hasattr(resource, 'close'):
                try:
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
                except Exception as e:
                    log.error(f"Error closing resource: {e}")
    
    def run_in_thread(self, func: Callable, *args, **kwargs):
        """Run function in thread pool."""
        return self.thread_pool.submit(func, *args, **kwargs)
    
    def shutdown(self) -> None:
        """Shutdown resource manager."""
        self.thread_pool.shutdown(wait=True)

# Global instances
cache_manager = CacheManager()
performance_monitor = PerformanceMonitor()
resource_manager = ResourceManager()

# Performance optimization utilities
async def parallel_execute(tasks: List[Callable], max_concurrent: int = 10) -> List[Any]:
    """Execute multiple async tasks in parallel with concurrency limit."""
    limiter = AsyncLimiter(max_concurrent)
    
    async def limited_task(task):
        async with limiter:
            return await task()
    
    return await asyncio.gather(*[limited_task(task) for task in tasks])

async def batch_process(items: List[Any], processor: Callable[[Any], Any], 
                       batch_size: int = 100) -> List[Any]:
    """Process items in batches."""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[processor(item) for item in batch])
        results.extend(batch_results)
    
    return results

def optimize_memory_usage():
    """Optimize memory usage by running garbage collection."""
    import gc
    gc.collect()

async def warmup_cache():
    """Warm up cache with frequently accessed data."""
    # This would preload common data into cache
    log.info("Warming up cache...")
    # Implementation would depend on specific data needs
    log.info("Cache warmup complete")
