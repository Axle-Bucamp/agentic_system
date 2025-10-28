"""
Observability and monitoring utilities for the Agentic Trading System.
"""
import time
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import logging

from core.logging import log
from core.config import settings

# Prometheus metrics
registry = CollectorRegistry()

# Trading metrics
trades_total = Counter(
    'trading_trades_total',
    'Total number of trades executed',
    ['exchange', 'action', 'status'],
    registry=registry
)

trade_value_total = Counter(
    'trading_value_total',
    'Total value of trades executed',
    ['exchange', 'action'],
    registry=registry
)

trade_duration = Histogram(
    'trading_duration_seconds',
    'Time taken to execute trades',
    ['exchange', 'action'],
    registry=registry
)

portfolio_value = Gauge(
    'trading_portfolio_value_usd',
    'Current portfolio value in USD',
    registry=registry
)

portfolio_pnl = Gauge(
    'trading_portfolio_pnl_usd',
    'Current portfolio P&L in USD',
    registry=registry
)

active_positions = Gauge(
    'trading_active_positions',
    'Number of active positions',
    registry=registry
)

# API metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint'],
    registry=registry
)

# Agent metrics
agent_heartbeats = Gauge(
    'trading_agent_heartbeats',
    'Agent heartbeat status',
    ['agent_name'],
    registry=registry
)

agent_processing_time = Histogram(
    'trading_agent_processing_seconds',
    'Agent message processing time',
    ['agent_name', 'message_type'],
    registry=registry
)

# Exchange metrics
exchange_connection_status = Gauge(
    'trading_exchange_connection_status',
    'Exchange connection status',
    ['exchange_name'],
    registry=registry
)

exchange_latency = Histogram(
    'trading_exchange_latency_seconds',
    'Exchange API latency',
    ['exchange_name', 'operation'],
    registry=registry
)

# Forecasting API metrics
forecasting_requests_total = Counter(
    'trading_forecasting_requests_total',
    'Total forecasting API requests',
    ['endpoint', 'status'],
    registry=registry
)

forecasting_latency = Histogram(
    'trading_forecasting_latency_seconds',
    'Forecasting API latency',
    ['endpoint'],
    registry=registry
)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    portfolio_value: float
    total_pnl: float
    daily_pnl: float
    active_positions: int
    trades_today: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float

class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.trade_history = []
        self.performance_history = []
        
    async def record_trade(self, exchange: str, action: str, value: float, 
                          duration: float, status: str = "success"):
        """Record a trade execution."""
        trades_total.labels(exchange=exchange, action=action, status=status).inc()
        trade_value_total.labels(exchange=exchange, action=action).inc(value)
        trade_duration.labels(exchange=exchange, action=action).observe(duration)
        
        # Store in history
        self.trade_history.append({
            'timestamp': datetime.utcnow(),
            'exchange': exchange,
            'action': action,
            'value': value,
            'duration': duration,
            'status': status
        })
        
        # Keep only last 1000 trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    async def record_api_request(self, method: str, endpoint: str, 
                                status_code: int, duration: float):
        """Record an API request."""
        api_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
        api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    async def record_agent_processing(self, agent_name: str, message_type: str, 
                                    duration: float):
        """Record agent message processing time."""
        agent_processing_time.labels(
            agent_name=agent_name, 
            message_type=message_type
        ).observe(duration)
    
    async def record_exchange_latency(self, exchange_name: str, operation: str, 
                                    latency: float):
        """Record exchange API latency."""
        exchange_latency.labels(
            exchange_name=exchange_name, 
            operation=operation
        ).observe(latency)
    
    async def record_forecasting_request(self, endpoint: str, status: str, 
                                       latency: float):
        """Record forecasting API request."""
        forecasting_requests_total.labels(endpoint=endpoint, status=status).inc()
        forecasting_latency.labels(endpoint=endpoint).observe(latency)
    
    async def update_portfolio_metrics(self, portfolio_data: Dict[str, Any]):
        """Update portfolio-related metrics."""
        portfolio_value.set(portfolio_data.get('total_value_usdc', 0))
        portfolio_pnl.set(portfolio_data.get('total_pnl', 0))
        active_positions.set(len(portfolio_data.get('holdings', {})))
    
    async def update_agent_heartbeat(self, agent_name: str, is_online: bool):
        """Update agent heartbeat status."""
        agent_heartbeats.labels(agent_name=agent_name).set(1 if is_online else 0)
    
    async def update_exchange_status(self, exchange_name: str, is_connected: bool):
        """Update exchange connection status."""
        exchange_connection_status.labels(exchange_name=exchange_name).set(1 if is_connected else 0)
    
    async def calculate_performance_metrics(self, portfolio_data: Dict[str, Any]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        current_time = datetime.utcnow()
        
        # Basic metrics
        portfolio_value = portfolio_data.get('total_value_usdc', 0)
        total_pnl = portfolio_data.get('total_pnl', 0)
        daily_pnl = portfolio_data.get('daily_pnl', 0)
        active_positions = len(portfolio_data.get('holdings', {}))
        
        # Calculate trades today
        today = current_time.date()
        trades_today = len([
            trade for trade in self.trade_history
            if trade['timestamp'].date() == today and trade['status'] == 'success'
        ])
        
        # Calculate win rate
        successful_trades = len([t for t in self.trade_history if t['status'] == 'success'])
        total_trades = len(self.trade_history)
        win_rate = successful_trades / total_trades if total_trades > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        if len(self.performance_history) > 1:
            returns = [
                (self.performance_history[i]['portfolio_value'] - 
                 self.performance_history[i-1]['portfolio_value']) / 
                self.performance_history[i-1]['portfolio_value']
                for i in range(1, len(self.performance_history))
            ]
            if returns:
                avg_return = sum(returns) / len(returns)
                volatility = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        if self.performance_history:
            peak = max(h['portfolio_value'] for h in self.performance_history)
            current = portfolio_value
            max_drawdown = (peak - current) / peak if peak > 0 else 0
        else:
            max_drawdown = 0
        
        # Calculate volatility
        if len(self.performance_history) > 1:
            values = [h['portfolio_value'] for h in self.performance_history[-30:]]  # Last 30 days
            if len(values) > 1:
                returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
                volatility = (sum((r - sum(returns)/len(returns)) ** 2 for r in returns) / len(returns)) ** 0.5
            else:
                volatility = 0
        else:
            volatility = 0
        
        metrics = PerformanceMetrics(
            timestamp=current_time,
            portfolio_value=portfolio_value,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            active_positions=active_positions,
            trades_today=trades_today,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility
        )
        
        # Store in history
        self.performance_history.append(asdict(metrics))
        if len(self.performance_history) > 365:  # Keep 1 year of history
            self.performance_history = self.performance_history[-365:]
        
        return metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'uptime_hours': uptime / 3600,
            'total_trades': len(self.trade_history),
            'performance_records': len(self.performance_history),
            'trades_today': len([
                t for t in self.trade_history
                if t['timestamp'].date() == datetime.utcnow().date()
            ]),
            'last_trade': self.trade_history[-1] if self.trade_history else None,
            'last_performance': self.performance_history[-1] if self.performance_history else None
        }

# Global metrics collector instance
metrics_collector = MetricsCollector()

async def get_prometheus_metrics() -> str:
    """Get Prometheus metrics in text format."""
    return generate_latest(registry).decode('utf-8')

class CircuitBreaker:
    """Circuit breaker pattern for external service calls."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e

# Circuit breakers for external services
forecasting_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
exchange_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)

class HealthChecker:
    """Health checking utilities."""
    
    @staticmethod
    async def check_redis_health() -> Dict[str, Any]:
        """Check Redis health."""
        try:
            from core.redis_client import redis_client
            await redis_client.ping()
            return {"status": "healthy", "latency_ms": 0}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    @staticmethod
    async def check_exchange_health(exchange_name: str) -> Dict[str, Any]:
        """Check exchange health."""
        try:
            from core.exchange_manager import exchange_manager
            # This would need to be implemented in exchange_manager
            return {"status": "healthy", "exchange": exchange_name}
        except Exception as e:
            return {"status": "unhealthy", "exchange": exchange_name, "error": str(e)}
    
    @staticmethod
    async def check_forecasting_api_health() -> Dict[str, Any]:
        """Check forecasting API health."""
        try:
            from core.forecasting_client import forecasting_client
            # This would need to be implemented in forecasting_client
            return {"status": "healthy", "api": "forecasting"}
        except Exception as e:
            return {"status": "unhealthy", "api": "forecasting", "error": str(e)}
    
    @staticmethod
    async def get_system_health() -> Dict[str, Any]:
        """Get overall system health."""
        health = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "services": {}
        }
        
        # Check Redis
        redis_health = await HealthChecker.check_redis_health()
        health["services"]["redis"] = redis_health
        
        # Check exchanges
        health["services"]["exchanges"] = {}
        for exchange in ["DEX", "MEXC"]:
            exchange_health = await HealthChecker.check_exchange_health(exchange)
            health["services"]["exchanges"][exchange] = exchange_health
        
        # Check forecasting API
        forecasting_health = await HealthChecker.check_forecasting_api_health()
        health["services"]["forecasting_api"] = forecasting_health
        
        # Determine overall status
        unhealthy_services = [
            name for name, service in health["services"].items()
            if service.get("status") != "healthy"
        ]
        
        if unhealthy_services:
            health["overall_status"] = "degraded" if len(unhealthy_services) < 3 else "unhealthy"
            health["unhealthy_services"] = unhealthy_services
        
        return health
