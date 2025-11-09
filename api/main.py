"""
Main FastAPI application for the Agentic Trading System.
Enhanced to match forecasting API patterns with production-grade features.
"""
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from collections import deque
from pathlib import Path
import inspect
from datetime import datetime, timedelta
import time
import uuid
import asyncio
import json
from functools import wraps
from uuid import uuid4

from core.config import settings
from core import asset_registry
from core.logging import log
from core.redis_client import redis_client
from core.exchange_manager import exchange_manager
from core.forecasting_client import forecasting_client
from core.models import (
    Portfolio, PerformanceMetrics, HumanValidationRequest,
    HumanValidationResponse, TradeDecision, AgentMessage, MessageType,
    TradeAction, ExchangeType, RiskMetrics
)
from core.memory.graph_memory import GraphMemoryManager
from core.pipelines import (
    TrendPipeline,
    FactPipeline,
    FusionEngine,
    FusionInputs,
    MemoryPruningPipeline,
    get_trend_assessment,
    set_trend_assessment,
    get_fact_insight,
    set_fact_insight,
    get_fusion_recommendation,
    set_fusion_recommendation,
)
# Security imports (commented out for now due to missing dependencies)
# from core.security.security_middleware import create_security_middleware
# from core.security.security_monitor import start_security_monitoring
# from api.security_endpoints import router as security_router


# Security
security = HTTPBearer(auto_error=False)

# Rate limiting storage
rate_limit_storage = {}

def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """Rate limiting decorator."""
    def decorator(func):
        signature = inspect.signature(func)
        expects_request = "request" in signature.parameters

        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host if request.client else "unknown"
            endpoint = request.url.path
            key = f"{client_ip}:{endpoint}"
            current_time = time.time()
            window_start = current_time - window_seconds

            # Clean old entries
            if key in rate_limit_storage:
                rate_limit_storage[key] = [
                    req_time
                    for req_time in rate_limit_storage[key]
                    if req_time > window_start
                ]
            else:
                rate_limit_storage[key] = []

            # Check rate limit
            if len(rate_limit_storage[key]) >= max_requests:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later.",
                )

            # Add current request
            rate_limit_storage[key].append(current_time)

            if expects_request:
                return await func(request, *args, **kwargs)
            return await func(*args, **kwargs)

        if expects_request:
            wrapper.__signature__ = signature
        else:
            request_param = inspect.Parameter(
                "request",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Request,
            )
            wrapper.__signature__ = signature.replace(
                parameters=(request_param, *signature.parameters.values())
            )
        return wrapper
    return decorator


CHAT_HISTORY_KEY = "chat:history:{user_id}"
LOG_FILE_MAP = {
    "trading": Path(settings.log_file),
    "errors": Path(settings.log_file).with_name("errors.log"),
    "decisions": Path(settings.log_file).with_name("trading_decisions.log"),
    "portfolio": Path(settings.log_file).with_name("portfolio_plans.log"),
}
DASHBOARD_SETTINGS_KEY = "dashboard:settings"
PIPELINE_LIVE_KEY = "pipeline_live_config"
PIPELINE_LIVE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    name: {
        "enabled": bool(config.get("enabled", False)),
        "interval": str(config.get("interval", "hours")),
    }
    for name, config in settings.pipeline_live_defaults.items()
}
PIPELINE_LIVE_DEFAULTS = {
    name: {
        "enabled": entry["enabled"],
        "interval": entry["interval"] if entry["interval"] in {"minutes", "hours", "days"} else "hours",
    }
    for name, entry in PIPELINE_LIVE_DEFAULTS.items()
}
VALID_PIPELINE_INTERVALS = {"minutes", "hours", "days"}
DEFAULT_DASHBOARD_SETTINGS: Dict[str, Any] = {
    "schedule_profile": settings.agent_schedule_profile,
    "memory_prune_limit": settings.memory_prune_limit,
    "memory_prune_similarity_threshold": settings.memory_prune_similarity_threshold,
    "review_interval_hours": settings.review_interval_hours,
    "review_prompt": settings.review_prompt_default,
    "observation_interval": settings.observation_interval,
    "decision_interval": settings.decision_interval,
    "forecast_interval": settings.forecast_interval,
    PIPELINE_LIVE_KEY: {name: value.copy() for name, value in PIPELINE_LIVE_DEFAULTS.items()},
}


async def _record_pipeline_trade(
    agent_label: str,
    ticker: str,
    action: str,
    confidence: float,
    price: Optional[float],
    metadata: Dict[str, Any],
) -> None:
    """Persist a simulated trade generated by manual pipeline runs."""
    trade_id = f"pipeline:{agent_label}:{uuid4().hex}"
    record = {
        "trade_id": trade_id,
        "ticker": ticker,
        "action": action.upper(),
        "quantity": metadata.get("quantity", 0.0),
        "executed_price": price,
        "entry_price": price,
        "evaluation_price": price,
        "pnl": 0.0,
        "reward": 0.0,
        "status": "SIMULATED",
        "confidence": confidence,
        "generated_by_pipeline": True,
        "agent": agent_label,
        "metadata": metadata,
        "timestamp": datetime.utcnow().isoformat(),
    }

    await redis_client.set_json(f"memory:trade:{trade_id}", record)
    await redis_client.lpush("memory:trades", json.dumps({"trade_id": trade_id}))
    await redis_client.ltrim("memory:trades", 0, 199)
    await redis_client.lpush(f"memory:trades:{ticker}", json.dumps({"trade_id": trade_id}))
    await redis_client.ltrim(f"memory:trades:{ticker}", 0, 199)
    log.bind(event="pipeline_trade", agent=agent_label).info(
        "Recorded pipeline trade %s %s action=%s confidence=%.2f",
        trade_id,
        ticker,
        action.upper(),
        confidence,
    )


async def _append_chat_entry(user_id: str, role: str, content: str) -> None:
    entry = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
    }
    key = CHAT_HISTORY_KEY.format(user_id=user_id)
    await redis_client.lpush(key, json.dumps(entry))
    await redis_client.ltrim(key, 0, max(settings.memory_chat_history_limit - 1, 0))


async def _get_chat_history(user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    key = CHAT_HISTORY_KEY.format(user_id=user_id)
    end_index = (limit - 1) if limit else settings.memory_chat_history_limit - 1
    raw_entries = await redis_client.lrange(key, 0, max(end_index, 0))
    history = []
    for raw in reversed(raw_entries):
        try:
            history.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return history


async def _load_recent_ai_decisions(limit: int = 3) -> List[Dict[str, Any]]:
    """Fetch recent AI decision metadata for chat grounding."""
    if not redis_client.redis:
        return []

    try:
        keys = await redis_client.redis.keys("ai_decision:*")
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("Unable to enumerate ai_decision keys: %s", exc)
        return []

    decisions: List[Dict[str, Any]] = []
    for key in keys:
        try:
            decision = await redis_client.get_json(key)
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to load decision %s: %s", key, exc)
            continue

        if decision:
            decision.setdefault("decision_id", key.split(":", 1)[-1])
            decisions.append(decision)

    decisions.sort(key=lambda item: item.get("completed_at") or item.get("timestamp", ""), reverse=True)
    return decisions[:limit]


async def _generate_chat_response(user_id: str, prompt: str) -> str:
    fusion_items: List[Dict[str, Any]] = []
    for ticker in asset_registry.get_assets():
        data = await redis_client.get_json(f"pipeline:fusion:{ticker}")
        if data:
            fusion_items.append(data)

    fusion_items.sort(key=lambda item: item.get("percent_allocation", 0.0), reverse=True)
    lines = [f"User prompt: {prompt.strip()}"]
    recent_decisions = await _load_recent_ai_decisions(limit=4)

    if not fusion_items:
        if recent_decisions:
            head = recent_decisions[0]
            ticker = head.get("ticker") or "Market"
            action = head.get("action") or "HOLD"
            result = head.get("result") or head.get("result_text") or "[no agent response captured]"
            status = head.get("status", "unknown").upper()
            lines.append(
                f"Analyst: Latest workforce decision for {ticker} ({action}) [{status}] -> {result}"
            )
            if head.get("error"):
                lines.append(f"Guardrail: {head['error']}")
            if len(recent_decisions) > 1:
                summaries = []
                for decision in recent_decisions[1:]:
                    summaries.append(
                        f"{decision.get('ticker', 'Market')} {decision.get('action', 'HOLD')} "
                        f"({decision.get('status', 'unknown')})"
                    )
                lines.append("Recent queue: " + ", ".join(summaries))
        else:
            lines.append("Analyst: No fresh fusion signals available yet. Keep positions neutral for now.")
        return "\n".join(lines)

    top_pick = fusion_items[0]
    counter_pick = next((item for item in fusion_items[1:] if item.get("action") != top_pick.get("action")), None)

    def format_item(item: Dict[str, Any]) -> str:
        alloc = item.get("percent_allocation", 0.0)
        return (
            f"{item.get('ticker')} -> {item.get('action')} "
            f"(confidence {item.get('confidence', 0.0):.2f}, alloc {alloc:.2%}, "
            f"risk {item.get('risk_level', 'UNKNOWN')})"
        )

    lines.append(f"Analyst A: {format_item(top_pick)}; rationale: {top_pick.get('rationale', 'n/a')}")

    if counter_pick:
        lines.append(f"Analyst B: {format_item(counter_pick)}; counterpoint: {counter_pick.get('rationale', 'n/a')}")
    else:
        lines.append("Analyst B: No opposing signal with meaningful conviction. Focus on disciplined sizing.")

    lines.append("Moderator: Balance confidence with stop-loss levels before acting.")

    if recent_decisions:
        lines.append("")
        lines.append("Decision Log Highlights:")
        for decision in recent_decisions[:3]:
            lines.append(
                f"- {decision.get('ticker', 'Market')} {decision.get('action', 'HOLD')} "
                f"[{decision.get('status', 'unknown')}]: {decision.get('result', decision.get('result_text', 'n/a'))}"
            )
            if decision.get("error"):
                lines.append(f"  Guardrail: {decision['error']}")

    return "\n".join(lines)


def _resolve_log_path(source: str) -> Path:
    base = LOG_FILE_MAP.get(source, LOG_FILE_MAP["trading"])
    if base.exists():
        return base
    fallback = Path.cwd() / "logs" / base.name
    if fallback.exists():
        return fallback
    return fallback


async def _read_recent_logs(path: Path, limit: int) -> List[str]:
    limit = max(limit, 1)

    def _read() -> List[str]:
        lines = deque(maxlen=limit)
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    lines.append(line.rstrip("\n"))
        except FileNotFoundError:
            return []
        return list(lines)

    return await asyncio.to_thread(_read)


async def _load_dashboard_settings() -> Dict[str, Any]:
    stored = await redis_client.get_json(DASHBOARD_SETTINGS_KEY) or {}
    merged = DEFAULT_DASHBOARD_SETTINGS.copy()
    merged.update(stored)
    return merged


async def _save_dashboard_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    merged = await _load_dashboard_settings()
    merged.update(payload)
    await redis_client.set_json(DASHBOARD_SETTINGS_KEY, merged)
    log.info("Dashboard settings updated: %s", payload)
    return merged


async def _load_pipeline_live_config() -> Dict[str, Dict[str, Any]]:
    """Load pipeline live-mode configuration with defaults."""
    dashboard_settings = await _load_dashboard_settings()
    stored = dashboard_settings.get(PIPELINE_LIVE_KEY) or {}
    config: Dict[str, Dict[str, Any]] = {}
    for name, defaults in PIPELINE_LIVE_DEFAULTS.items():
        entry = stored.get(name, {})
        enabled = bool(entry.get("enabled", defaults["enabled"]))
        interval = str(entry.get("interval", defaults["interval"])).lower()
        if interval not in VALID_PIPELINE_INTERVALS:
            interval = defaults["interval"]
        config[name] = {"enabled": enabled, "interval": interval}
    return config


async def _save_pipeline_live_config(updates: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Persist pipeline live-mode configuration with validation."""
    if not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail="Invalid payload; expected object.")

    current = await _load_pipeline_live_config()

    for name, override in updates.items():
        if name not in PIPELINE_LIVE_DEFAULTS:
            raise HTTPException(status_code=400, detail=f"Unknown pipeline '{name}'")
        if not isinstance(override, dict):
            raise HTTPException(status_code=400, detail=f"Invalid configuration for pipeline '{name}'")

        if "enabled" in override:
            current[name]["enabled"] = bool(override["enabled"])

        if "interval" in override:
            interval = str(override["interval"]).lower()
            if interval not in VALID_PIPELINE_INTERVALS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid interval '{interval}' for pipeline '{name}'",
                )
            current[name]["interval"] = interval

    stored = await redis_client.get_json(DASHBOARD_SETTINGS_KEY) or {}
    stored[PIPELINE_LIVE_KEY] = current
    await redis_client.set_json(DASHBOARD_SETTINGS_KEY, stored)
    log.info("Pipeline live configuration updated: %s", updates)
    return current


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token (optional authentication)."""
    if not credentials:
        return None  # Allow anonymous access for now
    
    # In production, validate JWT token here
    # For now, just return a mock user
    return {"user_id": "anonymous", "permissions": ["read", "write"]}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    log.info("Starting Agentic Trading System API...")
    await redis_client.connect()
    
    # Initialize exchange manager
    await exchange_manager.initialize()
    
    # Initialize forecasting client
    await forecasting_client.initialize()
           
    # Start security monitoring (commented out for now)
    # await start_security_monitoring()
    
    # Initialize portfolio if not exists
    portfolio = await redis_client.get_json("state:portfolio")
    if not portfolio:
        initial_portfolio = {
            "balance_usdc": settings.initial_capital,
            "holdings": {},
            "total_value_usdc": settings.initial_capital,
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "positions": []
        }
        await redis_client.set_json("state:portfolio", initial_portfolio)
        log.info(f"Initialized portfolio with {settings.initial_capital} USDC")
    
    yield
    
    # Shutdown
    log.info("Shutting down Agentic Trading System API...")
    await redis_client.disconnect()
    await exchange_manager.disconnect_all()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multi-agent trading system with DQN predictions, technical analysis, and risk management",
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
)

# Security middleware
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment != "production" else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security middleware (commented out for now due to missing dependencies)
# security_middleware = create_security_middleware(app, {
#     "scan_requests": True,
#     "scan_responses": True,
#     "block_threats": settings.environment == "production",
#     "rate_limit_scans": 100
# })
# app.add_middleware(type(security_middleware))

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Error handling middleware
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


# Enhanced health check with dependencies
@app.get("/health")
@rate_limit(max_requests=10, window_seconds=60)
async def health_check(request: Request):
    """Comprehensive health check endpoint."""
    log.debug(f"Health check requested from {request.client.host if request.client else 'unknown'}")
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.environment,
        "version": settings.app_version,
        "dependencies": {}
    }
    
    # Check Redis connection
    try:
        await redis_client.ping()
        health_status["dependencies"]["redis"] = "healthy"
    except Exception as e:
        health_status["dependencies"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check exchange manager
    try:
        # Check if exchange manager has any connected exchanges
        status = exchange_manager.get_status()
        connected_count = sum(1 for ex in status.get("exchanges", {}).values() if ex.get("connected", False))
        if connected_count > 0:
            health_status["dependencies"]["exchanges"] = f"healthy ({connected_count} connected)"
        else:
            health_status["dependencies"]["exchanges"] = "disconnected"
            health_status["status"] = "degraded"
        log.debug(f"Exchange manager status: {status}")
    except Exception as e:
        log.error(f"Exchange manager health check failed: {e}")
        health_status["dependencies"]["exchanges"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check forecasting API
    try:
        # Check if forecasting client is connected
        if hasattr(forecasting_client, 'client') and forecasting_client.client is not None:
            health_status["dependencies"]["forecasting_api"] = "healthy"
            log.debug("Forecasting API client is connected")
        else:
            health_status["dependencies"]["forecasting_api"] = "disconnected"
            health_status["status"] = "degraded"
            log.warning("Forecasting API client is not connected")
    except Exception as e:
        log.error(f"Forecasting API health check failed: {e}")
        health_status["dependencies"]["forecasting_api"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

# Forecasting API compatible endpoints
@app.get("/api/portfolios")
@rate_limit(max_requests=50, window_seconds=60)
async def get_portfolios(user: Optional[Dict] = Depends(get_current_user)):
    """Get all portfolios (compatible with forecasting API)."""
    portfolios = []
    
    # Get main portfolio
    portfolio_data = await redis_client.get_json("state:portfolio")
    if portfolio_data:
        portfolios.append({
            "id": "main",
            "name": "Main Trading Portfolio",
            "total_value": portfolio_data.get("total_value_usdc", 0),
            "balance": portfolio_data.get("balance_usdc", 0),
            "holdings": portfolio_data.get("holdings", {}),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        })
    
    return {"portfolios": portfolios, "count": len(portfolios)}

@app.get("/api/portfolios/{portfolio_id}")
@rate_limit(max_requests=50, window_seconds=60)
async def get_portfolio(portfolio_id: str, user: Optional[Dict] = Depends(get_current_user)):
    """Get specific portfolio details."""
    if portfolio_id != "main":
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    portfolio_data = await redis_client.get_json("state:portfolio")
    if not portfolio_data:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    return {
        "id": portfolio_id,
        "name": "Main Trading Portfolio",
        "total_value": portfolio_data.get("total_value_usdc", 0),
        "balance": portfolio_data.get("balance_usdc", 0),
        "holdings": portfolio_data.get("holdings", {}),
        "daily_pnl": portfolio_data.get("daily_pnl", 0),
        "total_pnl": portfolio_data.get("total_pnl", 0),
        "positions": portfolio_data.get("positions", []),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

@app.get("/api/portfolios/{portfolio_id}/trades")
@rate_limit(max_requests=100, window_seconds=60)
async def get_portfolio_trades(
    portfolio_id: str,
    limit: int = 50,
    offset: int = 0,
    user: Optional[Dict] = Depends(get_current_user)
):
    """Get trades for a specific portfolio."""
    if portfolio_id != "main":
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    trades_raw = await redis_client.lrange("memory:trades", offset, offset + limit - 1)
    trades: List[Dict[str, Any]] = []
    
    for trade_data in trades_raw:
        try:
            trade = json.loads(trade_data)
        except json.JSONDecodeError:
            continue

        trade_id = (
            trade.get("trade_id")
            or trade.get("decision_id")
            or trade.get("id")
        )
        if trade_id:
            enriched = await redis_client.get_json(f"memory:trade:{trade_id}")
            if enriched:
                trades.append(enriched)
                continue
        trades.append(trade)
    
    return {
        "trades": trades,
        "count": len(trades),
        "limit": limit,
        "offset": offset,
        "total": len(await redis_client.lrange("memory:trades", 0, -1))
    }

@app.post("/api/portfolios/{portfolio_id}/trades")
@rate_limit(max_requests=10, window_seconds=60)
async def execute_trade(
    portfolio_id: str,
    trade_request: Dict[str, Any],
    user: Optional[Dict] = Depends(get_current_user)
):
    """Execute a trade (compatible with forecasting API)."""
    if portfolio_id != "main":
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    try:
        # Extract trade parameters
        ticker = trade_request.get("ticker")
        action = trade_request.get("action")
        quantity = trade_request.get("quantity")
        order_type = trade_request.get("order_type", "market")
        
        if not all([ticker, action, quantity]):
            raise HTTPException(status_code=400, detail="Missing required trade parameters")
        
        # Convert action to enum
        try:
            trade_action = TradeAction(action.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid action. Must be BUY or SELL")
        
        # Determine exchange
        exchange_type = ExchangeType.DEX if ticker in ["BTC", "ETH", "SOL"] else ExchangeType.MEXC
        
        # Create symbol
        symbol = f"{ticker}USDC" if exchange_type == ExchangeType.DEX else f"{ticker}USDT"
        
        # Execute trade
        order = await exchange_manager.place_order(
            symbol=symbol,
            side=trade_action,
            order_type=order_type,
            amount=quantity,
            exchange_type=exchange_type
        )
        
        if not order:
            raise HTTPException(status_code=500, detail="Failed to execute trade")
        
        # Store trade in history
        trade_record = {
            "id": str(uuid.uuid4()),
            "portfolio_id": portfolio_id,
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "price": order.average_price,
            "total_cost": order.total_cost,
            "fee": order.fee,
            "exchange": exchange_type.value,
            "status": "filled" if order.is_filled else "pending",
            "created_at": datetime.utcnow().isoformat()
        }
        
        await redis_client.lpush("memory:trades", json.dumps(trade_record))
        
        return {
            "trade": trade_record,
            "status": "success",
            "message": "Trade executed successfully"
        }
        
    except Exception as e:
        log.error(f"Trade execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Trade execution failed: {str(e)}")


@app.get("/api/chat/{user_id}")
@rate_limit(max_requests=30, window_seconds=60)
async def get_chat_history_endpoint(user_id: str, limit: int = 50):
    """Return chat history for a given user."""
    history = await _get_chat_history(user_id, limit=limit)
    return {"user_id": user_id, "history": history}


@app.post("/api/chat")
@rate_limit(max_requests=30, window_seconds=60)
async def post_chat_message(payload: Dict[str, Any], user: Optional[Dict] = Depends(get_current_user)):
    """Append a chat message and return the assistant reply."""
    user_id = payload.get("user_id") or (user.get("user_id") if user else "anonymous")
    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    await _append_chat_entry(user_id, "user", message)
    reply = await _generate_chat_response(user_id, message)
    await _append_chat_entry(user_id, "assistant", reply)
    history = await _get_chat_history(user_id, limit=settings.memory_chat_history_limit)
    return {"user_id": user_id, "reply": reply, "history": history}


@app.get("/api/settings")
@rate_limit(max_requests=20, window_seconds=60)
async def get_dashboard_settings():
    """Return tunable dashboard settings."""
    return {"settings": await _load_dashboard_settings()}


@app.post("/api/settings")
@rate_limit(max_requests=20, window_seconds=60)
async def update_dashboard_settings(settings_payload: Dict[str, Any]):
    """Persist dashboard settings and apply runtime overrides."""
    merged = await _save_dashboard_settings(settings_payload)
    # Apply runtime overrides for intervals
    settings.agent_schedule_profile = merged.get("schedule_profile", settings.agent_schedule_profile)
    settings.memory_prune_limit = int(merged.get("memory_prune_limit", settings.memory_prune_limit))
    settings.memory_prune_similarity_threshold = float(
        merged.get("memory_prune_similarity_threshold", settings.memory_prune_similarity_threshold)
    )
    settings.review_interval_hours = int(merged.get("review_interval_hours", settings.review_interval_hours))
    settings.review_prompt_default = merged.get("review_prompt", settings.review_prompt_default)
    settings.observation_interval = merged.get("observation_interval", settings.observation_interval)
    settings.decision_interval = merged.get("decision_interval", settings.decision_interval)
    settings.forecast_interval = merged.get("forecast_interval", settings.forecast_interval)
    return {"settings": merged}


@app.get("/api/pipelines/live-config")
@rate_limit(max_requests=30, window_seconds=60)
async def get_pipeline_live_config():
    """Return live-mode configuration for pipelines."""
    config = await _load_pipeline_live_config()
    return {
        "pipelines": config,
        "intervals": sorted(VALID_PIPELINE_INTERVALS),
    }


@app.post("/api/pipelines/live-config")
@rate_limit(max_requests=30, window_seconds=60)
async def update_pipeline_live_config(payload: Dict[str, Any]):
    """Update live-mode configuration for pipelines."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload; expected object.")
    updates = payload.get("pipelines")
    if updates is None:
        updates = payload
    config = await _save_pipeline_live_config(updates)
    return {"pipelines": config}


@app.post("/api/review/run")
@rate_limit(max_requests=10, window_seconds=60)
async def trigger_review_run():
    """Trigger the weight review pipeline on the next memory agent cycle."""
    await redis_client.set("review:trigger", "1")
    return {"queued": True}


@app.post("/api/agents/copytrade/toggle")
@rate_limit(max_requests=20, window_seconds=60)
async def toggle_copytrade_agent(payload: Dict[str, Any]):
    """Enable or disable the copy trade agent runtime."""
    if "enabled" not in payload:
        raise HTTPException(status_code=400, detail="enabled flag required")
    enabled = bool(payload["enabled"])
    await redis_client.set("copytrade:enabled", "1" if enabled else "0")
    await redis_client.set_json(
        "copytrade:status",
        {"enabled": enabled, "updated_at": datetime.utcnow().isoformat()},
    )
    return {"enabled": enabled}

@app.get("/api/agents/rewards")
@rate_limit(max_requests=60, window_seconds=60)
async def agent_reward_summary():
    """Return aggregated reward metrics per agent."""
    rewards = await redis_client.get_json("memory:agent_rewards") or {}
    return {"rewards": rewards}


@app.get("/api/agents/status")
@rate_limit(max_requests=20, window_seconds=60)
async def get_agent_status():
    """Return high-level status for orchestrated agents and pipelines."""
    copytrade_status = await redis_client.get_json("copytrade:status") or {"enabled": True}

    trend_status = []
    fact_status = []
    fusion_status = []
    for ticker in asset_registry.get_assets():
        trend = await redis_client.get_json(f"pipeline:trend:{ticker}")
        if trend:
            trend_status.append({"ticker": ticker, "generated_at": trend.get("generated_at")})
        fact = await redis_client.get_json(f"pipeline:fact:{ticker}")
        if fact:
            fact_status.append({"ticker": ticker, "generated_at": fact.get("generated_at")})
        fusion = await redis_client.get_json(f"pipeline:fusion:{ticker}")
        if fusion:
            fusion_status.append(
                {
                    "ticker": ticker,
                    "generated_at": fusion.get("generated_at"),
                    "action": fusion.get("action"),
                    "confidence": fusion.get("confidence"),
                }
            )

    return {
        "copytrade": copytrade_status,
        "trend": trend_status,
        "fact": fact_status,
        "fusion": fusion_status,
        "logfire_enabled": bool(settings.logfire_token),
    }


@app.get("/api/pipelines/fusion")
@rate_limit(max_requests=20, window_seconds=60)
async def list_fusion_recommendations():
    """Return latest fusion recommendations for all assets."""
    items = []
    for ticker in asset_registry.get_assets():
        fusion = await redis_client.get_json(f"pipeline:fusion:{ticker}")
        if fusion:
            items.append(fusion)
    items.sort(key=lambda item: item.get("percent_allocation", 0.0), reverse=True)
    return {"items": items}


@app.post("/api/pipelines/trend/run")
@rate_limit(max_requests=20, window_seconds=60)
async def run_trend_pipeline_endpoint(payload: Dict[str, Any]):
    ticker = (payload or {}).get("ticker")
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    ticker = ticker.upper()
    pipeline = TrendPipeline(redis_client)
    assessment = await pipeline.run_for_ticker(ticker)
    if assessment:
        await set_trend_assessment(redis_client, assessment)
        chart_data = (assessment.supporting_signals or {}).get("chart", {})
        price = chart_data.get("current_price")
        assessment_payload = assessment.model_dump(mode="json")
        await _record_pipeline_trade(
            agent_label="TREND",
            ticker=ticker,
            action=assessment.recommended_action.value,
            confidence=assessment.confidence,
            price=price,
            metadata={"assessment": assessment_payload},
        )
        log.bind(event="pipeline_run", agent="TREND").info(
            "Trend pipeline run for %s -> %s (confidence=%.2f)",
            ticker,
            assessment.recommended_action.value,
            assessment.confidence,
        )
        return {"success": True, "assessment": assessment.dict()}
    log.bind(event="pipeline_run", agent="TREND").warning(
        "Trend pipeline insufficient data for %s", ticker
    )
    return {"success": False, "message": "Insufficient data to compute trend assessment"}


@app.post("/api/pipelines/fact/run")
@rate_limit(max_requests=20, window_seconds=60)
async def run_fact_pipeline_endpoint(payload: Dict[str, Any]):
    ticker = (payload or {}).get("ticker")
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    ticker = ticker.upper()
    async with FactPipeline(redis_client) as pipeline:
        insight = await pipeline.run_for_ticker(ticker)
    if insight:
        await set_fact_insight(redis_client, insight)
        action = "HOLD"
        if insight.sentiment_score > 0.2:
            action = "BUY"
        elif insight.sentiment_score < -0.2:
            action = "SELL"
        insight_payload = insight.model_dump(mode="json")
        await _record_pipeline_trade(
            agent_label="FACT",
            ticker=ticker,
            action=action,
            confidence=insight.confidence,
            price=None,
            metadata={"insight": insight_payload},
        )
        log.bind(event="pipeline_run", agent="FACT").info(
            "Fact pipeline run for %s sentiment=%.2f -> %s",
            ticker,
            insight.sentiment_score,
            action,
        )
        return {"success": True, "insight": insight.dict()}
    log.bind(event="pipeline_run", agent="FACT").warning(
        "Fact pipeline insufficient data for %s", ticker
    )
    return {"success": False, "message": "Insufficient data to compute fact insight"}


@app.post("/api/pipelines/fusion/run")
@rate_limit(max_requests=20, window_seconds=60)
async def run_fusion_pipeline_endpoint(payload: Dict[str, Any]):
    ticker = (payload or {}).get("ticker")
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    ticker = ticker.upper()
    trend = await get_trend_assessment(redis_client, ticker)
    fact = await get_fact_insight(redis_client, ticker)
    risk_payload = await redis_client.get_json(f"risk:asset:{ticker}") or {}
    risk = None
    if risk_payload:
        try:
            risk = RiskMetrics(**risk_payload)
        except Exception:
            risk = None
    inputs = FusionInputs(
        trend=trend,
        fact=fact,
        risk=risk,
        copy_confidence=0.0,
    )
    fusion_engine = FusionEngine()
    recommendation = fusion_engine.combine(ticker, inputs)
    await set_fusion_recommendation(redis_client, recommendation)
    trend_payload = trend.model_dump(mode="json") if trend else None
    fact_payload = fact.model_dump(mode="json") if fact else None
    recommendation_payload = recommendation.model_dump(mode="json")
    await _record_pipeline_trade(
        agent_label="FUSION",
        ticker=ticker,
        action=recommendation.action.value,
        confidence=recommendation.confidence,
        price=None,
        metadata={
            "recommendation": recommendation_payload,
            "trend": trend_payload,
            "fact": fact_payload,
            "risk": risk_payload,
        },
    )
    log.bind(event="pipeline_run", agent="FUSION").info(
        "Fusion pipeline run for %s -> %s (confidence=%.2f)",
        ticker,
        recommendation.action.value,
        recommendation.confidence,
    )
    return {"success": True, "recommendation": recommendation.dict()}


@app.post("/api/pipelines/prune/run")
@rate_limit(max_requests=20, window_seconds=60)
async def run_prune_pipeline_endpoint():
    pipeline = MemoryPruningPipeline(redis_client)
    await pipeline.prune_all()
    log.bind(event="pipeline_run", agent="PRUNE").info("Memory pruning pipeline executed on demand")
    return {"success": True}


@app.get("/api/pipelines/trend/{ticker}")
@rate_limit(max_requests=30, window_seconds=60)
async def get_trend_assessment_endpoint(ticker: str):
    trend = await redis_client.get_json(f"pipeline:trend:{ticker.upper()}")
    if not trend:
        raise HTTPException(status_code=404, detail="Trend assessment not found")
    return {"assessment": trend}


@app.get("/api/pipelines/fact/{ticker}")
@rate_limit(max_requests=30, window_seconds=60)
async def get_fact_insight_endpoint(ticker: str):
    fact = await redis_client.get_json(f"pipeline:fact:{ticker.upper()}")
    if not fact:
        raise HTTPException(status_code=404, detail="Fact insight not found")
    return {"insight": fact}


@app.get("/api/memory/graph")
@rate_limit(max_requests=120, window_seconds=60)
async def get_graph_memory_snapshot(nodes: int = 100, edges: int = 200):
    """Return a snapshot of the knowledge graph for monitoring."""
    manager = GraphMemoryManager(redis_client)
    snapshot = await manager.get_snapshot(node_limit=min(nodes, 500), edge_limit=min(edges, 1000))
    nodes_payload = [node.dict() for node in snapshot["nodes"]]
    edges_payload = [edge.dict() for edge in snapshot["edges"]]
    return {"nodes": nodes_payload, "edges": edges_payload}


@app.get("/api/logs/recent")
@rate_limit(max_requests=120, window_seconds=60)
async def get_recent_logs(limit: int = 100, source: str = "trading"):
    """Return tail of server logs for observability dashboards."""
    path = _resolve_log_path(source)
    lines = await _read_recent_logs(path, limit)
    return {
        "source": source,
        "path": str(path),
        "lines": lines,
        "logfire_enabled": bool(settings.logfire_token),
    }


@app.get("/api/watchlists")
@rate_limit(max_requests=50, window_seconds=60)
async def get_watchlists(user: Optional[Dict] = Depends(get_current_user)):
    """Get all watchlists."""
    watchlist_data = await redis_client.get_json("watchlists")
    watchlists = watchlist_data or []
    
    return {"watchlists": watchlists, "count": len(watchlists)}

@app.post("/api/watchlists")
@rate_limit(max_requests=20, window_seconds=60)
async def create_watchlist(
    watchlist_data: Dict[str, Any],
    user: Optional[Dict] = Depends(get_current_user)
):
    """Create a new watchlist."""
    watchlist = {
        "id": str(uuid.uuid4()),
        "name": watchlist_data.get("name", "New Watchlist"),
        "tickers": watchlist_data.get("tickers", []),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    # Get existing watchlists
    watchlists = await redis_client.get_json("watchlists") or []
    watchlists.append(watchlist)
    
    await redis_client.set_json("watchlists", watchlists)
    
    return {"watchlist": watchlist, "status": "success"}

@app.get("/api/tickers/available")
@rate_limit(max_requests=100, window_seconds=60)
async def get_available_tickers():
    """Get available tickers from forecasting API."""
    try:
        tickers = await forecasting_client.get_available_tickers()
        return {"tickers": tickers, "count": len(tickers)}
    except Exception as e:
        log.error(f"Error fetching available tickers: {e}")
        # Fallback to supported assets
        return {"tickers": settings.supported_assets, "count": len(settings.supported_assets)}

@app.get("/api/tickers/{ticker}/forecast")
@rate_limit(max_requests=50, window_seconds=60)
async def get_ticker_forecast(ticker: str, interval: str = "hours"):
    """Get forecast for a specific ticker."""
    try:
        forecast = await forecasting_client.get_stock_forecast(ticker, interval)
        if not forecast:
            raise HTTPException(status_code=404, detail=f"No forecast available for {ticker}")
        return forecast
    except Exception as e:
        log.error(f"Error fetching forecast for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch forecast: {str(e)}")

@app.get("/api/tickers/{ticker}/recommendation")
@rate_limit(max_requests=50, window_seconds=60)
async def get_ticker_recommendation(ticker: str, interval: str = "hours"):
    """Get trading recommendation for a specific ticker."""
    try:
        recommendation = await forecasting_client.get_action_recommendation(ticker, interval)
        if not recommendation:
            raise HTTPException(status_code=404, detail=f"No recommendation available for {ticker}")
        return recommendation
    except Exception as e:
        log.error(f"Error fetching recommendation for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch recommendation: {str(e)}")

@app.get("/api/user/preferences")
@rate_limit(max_requests=20, window_seconds=60)
async def get_user_preferences(user: Optional[Dict] = Depends(get_current_user)):
    """Get user trading preferences."""
    preferences = await redis_client.get_json("user:preferences")
    if not preferences:
        # Return default preferences
        preferences = {
            "risk_tolerance": "medium",
            "max_position_size": settings.max_position_size,
            "max_daily_loss": settings.max_daily_loss,
            "min_confidence": settings.min_confidence,
            "preferred_exchanges": ["DEX", "MEXC"],
            "notification_settings": {
                "email": True,
                "push": False,
                "sms": False
            }
        }
    
    return {"preferences": preferences}

@app.put("/api/user/preferences")
@rate_limit(max_requests=10, window_seconds=60)
async def update_user_preferences(
    preferences: Dict[str, Any],
    user: Optional[Dict] = Depends(get_current_user)
):
    """Update user trading preferences."""
    await redis_client.set_json("user:preferences", preferences)
    return {"status": "success", "message": "Preferences updated successfully"}

@app.get("/api/ai/decisions")
@rate_limit(max_requests=50, window_seconds=60)
async def get_ai_decisions(limit: int = 50):
    """Get recent AI decisions and task solving logs."""
    try:
        # Get all decision keys from Redis
        keys = await redis_client.redis.keys("ai_decision:*")
        decisions = []
        
        for key in keys[:limit]:
            decision = await redis_client.get_json(key)
            if decision:
                decision.setdefault("decision_id", key.split(":", 1)[-1])
                decisions.append(decision)
        
        # Sort by timestamp (newest first)
        decisions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return {
            "decisions": decisions,
            "count": len(decisions),
            "total": len(keys)
        }
    except Exception as e:
        log.error(f"Error fetching AI decisions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch AI decisions: {str(e)}")

@app.get("/api/ai/decisions/{decision_id}")
@rate_limit(max_requests=50, window_seconds=60)
async def get_ai_decision(decision_id: str):
    """Get a specific AI decision by ID."""
    try:
        decision = await redis_client.get_json(f"ai_decision:{decision_id}")
        if not decision:
            raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
        decision.setdefault("decision_id", decision_id)
        return decision
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error fetching AI decision {decision_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch AI decision: {str(e)}")

@app.post("/api/ai/analyze")
@rate_limit(max_requests=10, window_seconds=60)
async def trigger_ai_analysis(
    request_data: Dict[str, Any],
    user: Optional[Dict] = Depends(get_current_user)
):
    """Trigger AI analysis for a ticker based on user preferences."""
    try:
        ticker = request_data.get("ticker", "").upper()
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker is required")
        
        # Get user preferences
        preferences = await redis_client.get_json("user:preferences") or {}
        
        # Create a signal message for the workforce orchestrator
        from core.models import AgentMessage, MessageType
        from agents.workforce_orchestrator import WorkforceOrchestratorAgent
        
        # Get orchestrator instance (this would need to be injected or retrieved)
        # For now, we'll create a message and publish it to Redis
        signal_message = {
            "ticker": ticker,
            "action": request_data.get("action", "ANALYZE"),
            "preferences": preferences,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Publish to Redis for orchestrator to pick up
        await redis_client.redis.publish(
            "agent_messages",
            json.dumps({
                "message_type": MessageType.SIGNAL_GENERATED.value,
                "payload": signal_message
            })
        )
        
        return {
            "status": "success",
            "message": f"AI analysis triggered for {ticker}",
            "ticker": ticker
        }
    except Exception as e:
        log.error(f"Error triggering AI analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger AI analysis: {str(e)}")


# Market order endpoint for UI
@app.post("/api/trades/market")
@rate_limit(max_requests=20, window_seconds=60)
async def create_market_order(
    order_data: Dict[str, Any],
    user: Optional[Dict] = Depends(get_current_user)
):
    """
    Create a market order (fake trading mode).
    
    Body:
        - ticker: str (e.g., "BTC")
        - side: str ("BUY" or "SELL")
        - quantity: float (amount in USDC for BUY, asset units for SELL)
    """
    try:
        ticker = order_data.get("ticker", "").upper()
        side = order_data.get("side", "BUY").upper()
        quantity = float(order_data.get("quantity", 0))
        
        if not ticker or quantity <= 0:
            raise HTTPException(status_code=400, detail="Invalid order parameters")
        
        if side not in ["BUY", "SELL"]:
            raise HTTPException(status_code=400, detail="Side must be BUY or SELL")
        
        # Use DEX simulator client for fake trading
        from core.dex_simulator_client import dex_simulator_client
        await dex_simulator_client.connect()
        
        if side == "BUY":
            result = await dex_simulator_client.buy_asset(ticker, quantity)
        else:
            result = await dex_simulator_client.sell_asset(ticker, quantity)
        
        await dex_simulator_client.disconnect()
        
        if result.get("success"):
            return {
                "status": "success",
                "order": {
                    "ticker": ticker,
                    "side": side,
                    "quantity": quantity,
                    "executed_at": datetime.utcnow().isoformat(),
                    **result
                }
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Order failed"))
            
    except Exception as e:
        log.error(f"Error creating market order: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create order: {str(e)}")

# Legacy endpoints (for backward compatibility)
@app.get("/api/portfolio", response_model=Portfolio)
@rate_limit(max_requests=50, window_seconds=60)
async def get_portfolio_legacy():
    """Get current portfolio state (legacy endpoint)."""
    portfolio_data = await redis_client.get_json("state:portfolio")
    
    if not portfolio_data:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    return Portfolio(**portfolio_data)


@app.get("/api/orchestrator/wallet-plan")
@rate_limit(max_requests=50, window_seconds=60)
async def get_wallet_plan(request: Request):
    """Fetch the latest wallet balancing plan and optional summary."""
    plan = await redis_client.get_json("orchestrator:wallet_plan")
    summary = await redis_client.get_json("orchestrator:wallet_plan_summary")

    # Return empty payloads instead of 404s to keep the UI happy during cold start
    return {
        "plan": plan,
        "summary": summary,
    }


@app.get("/api/orchestrator/wallet-plan/history")
@rate_limit(max_requests=50, window_seconds=60)
async def get_wallet_plan_history(request: Request, limit: int = 10):
    """Fetch historical wallet plans stored by the orchestrator."""
    try:
        if limit <= 0:
            limit = 1
        raw_entries = await redis_client.lrange(
            "orchestrator:wallet_plan_history",
            0,
            max(0, limit - 1),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("Failed to read wallet plan history: %s", exc)
        raw_entries = []

    history: List[Dict[str, Any]] = []
    for raw in raw_entries:
        try:
            history.append(json.loads(raw))
        except json.JSONDecodeError:
            log.warning("Skipping malformed wallet plan entry")

    return {"history": history}


@app.get("/api/copytrade/wallet-scores")
@rate_limit(max_requests=50, window_seconds=60)
async def get_copytrade_wallet_scores(request: Request):
    """Expose the copy-trade agent's latest wallet scoring snapshot."""
    scores = await redis_client.get_json("copytrade:wallet_scores") or {}
    if not isinstance(scores, dict):
        log.warning("Unexpected wallet score payload type: %s", type(scores))
        scores = {}
    return {"wallets": scores}


@app.get("/api/news/weighted")
@rate_limit(max_requests=50, window_seconds=60)
async def get_weighted_news_entries(request: Request):
    """Return recency-weighted news sentiment entries from memory agent."""
    keys: List[str] = []
    entries: List[Dict[str, Any]] = []

    if not redis_client.redis:
        return {"entries": entries}

    try:
        keys = await redis_client.redis.keys("memory:news:weighted:*")
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("Failed to enumerate weighted news keys: %s", exc)
        keys = []

    for key in keys:
        try:
            entry = await redis_client.get_json(key)
            if entry:
                # Preserve ticker ordering by pushing market-wide sentiment last
                entries.append(entry)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.warning("Unable to parse weighted news from %s: %s", key, exc)

    entries.sort(
        key=lambda item: item.get("last_updated", ""),
        reverse=True,
    )

    return {"entries": entries}


@app.get("/api/agent-schedule/profile")
@rate_limit(max_requests=50, window_seconds=60)
async def get_agent_schedule_profile(request: Request):
    """Return the active agent scheduling profile for display in the UI."""
    return {
        "profile": settings.agent_schedule_profile,
        "profiles": list(settings.agent_schedule_profiles.keys()),
    }


@app.get("/api/performance", response_model=PerformanceMetrics)
@rate_limit(max_requests=50, window_seconds=60)
async def get_performance():
    """Get performance metrics."""
    metrics_data = await redis_client.get_json("memory:performance")
    
    if not metrics_data:
        raise HTTPException(status_code=404, detail="Performance metrics not found")
    
    return PerformanceMetrics(**metrics_data)


@app.get("/api/assets")
@rate_limit(max_requests=100, window_seconds=60)
async def get_supported_assets():
    """Get list of supported assets."""
    return {
        "assets": settings.supported_assets,
        "tiers": {
            "tier_1": settings.tier_1_assets,
            "tier_2": settings.tier_2_assets,
            "tier_3": settings.tier_3_assets,
            "tier_4": settings.tier_4_assets
        }
    }


@app.get("/api/signals/{ticker}")
@rate_limit(max_requests=100, window_seconds=60)
async def get_ticker_signals(ticker: str):
    """Get latest signals for a ticker."""
    signals = {}
    
    # Get DQN signal
    dqn_signal = await redis_client.get_json(f"dqn:prediction:{ticker}")
    if dqn_signal:
        signals["dqn"] = dqn_signal
    
    # Get chart signal
    chart_signal = await redis_client.get_json(f"chart:signal:{ticker}")
    if chart_signal:
        signals["chart"] = chart_signal
    
    # Get news sentiment
    news_sentiment = await redis_client.get_json(f"news:sentiment:{ticker}")
    if news_sentiment:
        signals["news"] = news_sentiment
    
    # Get risk metrics
    risk_metrics = await redis_client.get_json(f"risk:asset:{ticker}")
    if risk_metrics:
        signals["risk"] = risk_metrics
    
    return signals


@app.get("/api/market-data/{ticker}")
@rate_limit(max_requests=200, window_seconds=60)
async def get_market_data(ticker: str):
    """Get latest market data for a ticker."""
    market_data = await redis_client.get_json(f"market:{ticker}")
    
    if not market_data:
        raise HTTPException(status_code=404, detail=f"Market data not found for {ticker}")
    
    return market_data


@app.get("/api/validations/pending")
@rate_limit(max_requests=50, window_seconds=60)
async def get_pending_validations():
    """Get pending human validation requests."""
    # Get all validation request keys
    keys = []
    
    # This is a simplified version - in production, use Redis SCAN
    for i in range(100):  # Check up to 100 possible requests
        key = f"validation:request:{i}"
        if await redis_client.exists(key):
            keys.append(key)
    
    requests = []
    for key in keys:
        request_data = await redis_client.get_json(key)
        if request_data:
            requests.append(request_data)
    
    return {"pending_validations": requests}


@app.post("/api/validations/{request_id}/respond")
@rate_limit(max_requests=20, window_seconds=60)
async def respond_to_validation(request_id: str, response: HumanValidationResponse):
    """Respond to a human validation request."""
    # Check if request exists
    request_data = await redis_client.get_json(f"validation:request:{request_id}")
    
    if not request_data:
        raise HTTPException(status_code=404, detail="Validation request not found")
    
    # Send response to orchestrator
    message = AgentMessage(
        message_type=MessageType.HUMAN_VALIDATION_RESPONSE,
        sender=None,  # From API/human
        payload=response.dict()
    )
    
    await redis_client.publish("agent:orchestrator", message.dict())
    
    # Remove request
    await redis_client.delete(f"validation:request:{request_id}")
    
    return {"status": "success", "message": "Validation response sent"}


@app.get("/api/history/trades")
@rate_limit(max_requests=100, window_seconds=60)
async def get_trade_history(limit: int = 50):
    """Get trade history."""
    trades_raw = await redis_client.lrange("memory:trades", 0, limit - 1)
    
    trades: List[Dict[str, Any]] = []
    for raw in trades_raw:
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        trade_id = (
            entry.get("trade_id")
            or entry.get("decision_id")
            or entry.get("id")
        )
        if trade_id:
            enriched = await redis_client.get_json(f"memory:trade:{trade_id}")
            if enriched:
                trades.append(enriched)
                continue
        trades.append(entry)
    
    return {"trades": trades, "count": len(trades)}


@app.get("/api/history/trades/{ticker}")
@rate_limit(max_requests=100, window_seconds=60)
async def get_ticker_trade_history(ticker: str, limit: int = 20):
    """Get trade history for a specific ticker."""
    trades_raw = await redis_client.lrange(f"memory:trades:{ticker}", 0, limit - 1)
    
    trades: List[Dict[str, Any]] = []
    for raw in trades_raw:
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        trade_id = (
            entry.get("trade_id")
            or entry.get("decision_id")
            or entry.get("id")
        )
        if trade_id:
            enriched = await redis_client.get_json(f"memory:trade:{trade_id}")
            if enriched:
                trades.append(enriched)
                continue
        trades.append(entry)
    
    return {"ticker": ticker, "trades": trades, "count": len(trades)}


@app.get("/api/agents/status")
@rate_limit(max_requests=50, window_seconds=60)
async def get_agents_status():
    """Get status of all agents."""
    # Get recent heartbeats
    agents = [
        "memory", "dqn", "chart", "risk", "news", "copytrade", "orchestrator"
    ]
    
    status = {}
    
    for agent in agents:
        # Check if agent has sent a heartbeat recently
        heartbeat_key = f"agent:heartbeat:{agent}"
        heartbeat = await redis_client.get(heartbeat_key)
        
        status[agent] = {
            "status": "online" if heartbeat else "unknown",
            "last_heartbeat": heartbeat if heartbeat else None
        }
    
    return {"agents": status}


@app.post("/api/wallets/track")
@rate_limit(max_requests=20, window_seconds=60)
async def add_wallet_to_track(wallet_data: Dict):
    """Add a wallet to track for copy trading."""
    # Publish message to copy trade agent
    message = AgentMessage(
        message_type=MessageType.MARKET_DATA_UPDATE,
        sender=None,
        payload={"add_wallet": wallet_data}
    )
    
    await redis_client.publish("agent:copytrade", message.dict())
    
    return {"status": "success", "message": "Wallet added to tracking"}


@app.get("/api/wallets/tracked")
@rate_limit(max_requests=50, window_seconds=60)
async def get_tracked_wallets():
    """Get list of tracked wallets."""
    wallets = await redis_client.get_json("copytrade:tracked_wallets")
    
    return {"wallets": wallets or []}


@app.get("/api/risk/portfolio")
@rate_limit(max_requests=50, window_seconds=60)
async def get_portfolio_risk():
    """Get portfolio-level risk metrics."""
    risk_data = await redis_client.get_json("risk:portfolio")
    
    if not risk_data:
        raise HTTPException(status_code=404, detail="Portfolio risk data not found")
    
    return risk_data


@app.get("/api/risk/{ticker}")
@rate_limit(max_requests=100, window_seconds=60)
async def get_ticker_risk(ticker: str):
    """Get risk metrics for a specific ticker."""
    risk_data = await redis_client.get_json(f"risk:asset:{ticker}")
    
    if not risk_data:
        raise HTTPException(status_code=404, detail=f"Risk data not found for {ticker}")
    
    return risk_data


@app.get("/api/news/market-sentiment")
@rate_limit(max_requests=50, window_seconds=60)
async def get_market_sentiment():
    """Get overall market sentiment from news analysis."""
    sentiment = await redis_client.get_json("news:market_sentiment")
    
    if not sentiment:
        return {"message": "No market sentiment data available"}
    
    return sentiment


@app.get("/api/config")
@rate_limit(max_requests=20, window_seconds=60)
async def get_config():
    """Get system configuration (non-sensitive)."""
    return {
        "initial_capital": settings.initial_capital,
        "max_position_size": settings.max_position_size,
        "max_daily_loss": settings.max_daily_loss,
        "max_drawdown": settings.max_drawdown,
        "min_confidence": settings.min_confidence,
        "trading_fee": settings.trading_fee,
        "supported_assets": settings.supported_assets,
        "observation_interval": settings.observation_interval,
        "decision_interval": settings.decision_interval,
        "forecast_interval": settings.forecast_interval
    }


@app.get("/")
@rate_limit(max_requests=100, window_seconds=60)
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
           "docs": "/docs" if settings.environment != "production" else None,
           "health": "/health",
           "api_version": "v1",
           "endpoints": {
               "portfolios": "/api/portfolios",
               "tickers": "/api/tickers",
               "watchlists": "/api/watchlists",
               "trades": "/api/portfolios/{id}/trades",
               "forecasts": "/api/tickers/{ticker}/forecast",
               "recommendations": "/api/tickers/{ticker}/recommendation",
               "security": "/api/security"
           }
       }


# Include security router (commented out for now)
# app.include_router(security_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

