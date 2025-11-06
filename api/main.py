"""
Main FastAPI application for the Agentic Trading System.
Enhanced to match forecasting API patterns with production-grade features.
"""
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import time
import uuid
import asyncio
import json
from functools import wraps

from core.config import settings
from core.logging import log
from core.redis_client import redis_client
from core.exchange_manager import exchange_manager
from core.forecasting_client import forecasting_client
from core.models import (
    Portfolio, PerformanceMetrics, HumanValidationRequest,
    HumanValidationResponse, TradeDecision, AgentMessage, MessageType,
    TradeAction, ExchangeType
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
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            current_time = time.time()
            window_start = current_time - window_seconds
            
            # Clean old entries
            if client_ip in rate_limit_storage:
                rate_limit_storage[client_ip] = [
                    req_time for req_time in rate_limit_storage[client_ip]
                    if req_time > window_start
                ]
            else:
                rate_limit_storage[client_ip] = []
            
            # Check rate limit
            if len(rate_limit_storage[client_ip]) >= max_requests:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
            
            # Add current request
            rate_limit_storage[client_ip].append(current_time)
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

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
    trades = []
    
    for trade_data in trades_raw:
        try:
            import json
            trade = json.loads(trade_data)
            trades.append(trade)
        except json.JSONDecodeError:
            continue
    
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
    
    trades = [json.loads(t) for t in trades_raw]
    
    return {"trades": trades, "count": len(trades)}


@app.get("/api/history/trades/{ticker}")
@rate_limit(max_requests=100, window_seconds=60)
async def get_ticker_trade_history(ticker: str, limit: int = 20):
    """Get trade history for a specific ticker."""
    trades_raw = await redis_client.lrange(f"memory:trades:{ticker}", 0, limit - 1)
    
    trades = [json.loads(t) for t in trades_raw]
    
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

