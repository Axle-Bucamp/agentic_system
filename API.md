# API Documentation

This document provides comprehensive documentation for the Agentic Trading System API.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## Authentication

The API supports optional JWT authentication. Include the token in the Authorization header:

```bash
Authorization: Bearer <your-jwt-token>
```

## Rate Limiting

All endpoints are rate limited:
- **API endpoints**: 10 requests/second per IP
- **Health check**: 1 request/second per IP
- **Trading endpoints**: 5 requests/second per IP

## Error Handling

All errors return a consistent JSON format:

```json
{
  "error": {
    "code": 400,
    "message": "Error description",
    "request_id": "uuid",
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

## Endpoints

### Health Check

#### GET /health

Check system health and dependencies.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "environment": "production",
  "version": "1.0.0",
  "dependencies": {
    "redis": "healthy",
    "exchanges": "healthy",
    "forecasting_api": "healthy"
  }
}
```

### Portfolio Management

#### GET /api/portfolios

Get all portfolios.

**Response:**
```json
{
  "portfolios": [
    {
      "id": "main",
      "name": "Main Trading Portfolio",
      "total_value": 10000.0,
      "balance": 5000.0,
      "holdings": {
        "BTC": 0.1,
        "ETH": 1.0
      },
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  ],
  "count": 1
}
```

#### GET /api/portfolios/{portfolio_id}

Get specific portfolio details.

**Parameters:**
- `portfolio_id` (string): Portfolio identifier

**Response:**
```json
{
  "id": "main",
  "name": "Main Trading Portfolio",
  "total_value": 10000.0,
  "balance": 5000.0,
  "holdings": {
    "BTC": 0.1,
    "ETH": 1.0
  },
  "daily_pnl": 100.0,
  "total_pnl": 1000.0,
  "positions": [
    {
      "ticker": "BTC",
      "quantity": 0.1,
      "average_price": 50000.0,
      "current_price": 51000.0,
      "unrealized_pnl": 100.0
    }
  ],
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

#### GET /api/portfolios/{portfolio_id}/trades

Get trades for a specific portfolio.

**Parameters:**
- `portfolio_id` (string): Portfolio identifier
- `limit` (int, optional): Number of trades to return (default: 50)
- `offset` (int, optional): Number of trades to skip (default: 0)

**Response:**
```json
{
  "trades": [
    {
      "id": "trade-uuid",
      "portfolio_id": "main",
      "ticker": "BTC",
      "action": "BUY",
      "quantity": 0.1,
      "price": 50000.0,
      "total_cost": 5000.0,
      "fee": 5.0,
      "exchange": "DEX",
      "status": "filled",
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "count": 1,
  "limit": 50,
  "offset": 0,
  "total": 1
}
```

#### POST /api/portfolios/{portfolio_id}/trades

Execute a trade.

**Parameters:**
- `portfolio_id` (string): Portfolio identifier

**Request Body:**
```json
{
  "ticker": "BTC",
  "action": "BUY",
  "quantity": 0.1,
  "order_type": "market"
}
```

**Response:**
```json
{
  "trade": {
    "id": "trade-uuid",
    "portfolio_id": "main",
    "ticker": "BTC",
    "action": "BUY",
    "quantity": 0.1,
    "price": 50000.0,
    "total_cost": 5000.0,
    "fee": 5.0,
    "exchange": "DEX",
    "status": "filled",
    "created_at": "2024-01-01T00:00:00Z"
  },
  "status": "success",
  "message": "Trade executed successfully"
}
```

### Watchlist Management

#### GET /api/watchlists

Get all watchlists.

**Response:**
```json
{
  "watchlists": [
    {
      "id": "watchlist-uuid",
      "name": "My Watchlist",
      "tickers": ["BTC", "ETH", "SOL"],
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  ],
  "count": 1
}
```

#### POST /api/watchlists

Create a new watchlist.

**Request Body:**
```json
{
  "name": "My Watchlist",
  "tickers": ["BTC", "ETH", "SOL"]
}
```

**Response:**
```json
{
  "watchlist": {
    "id": "watchlist-uuid",
    "name": "My Watchlist",
    "tickers": ["BTC", "ETH", "SOL"],
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  },
  "status": "success"
}
```

### Ticker Information

#### GET /api/tickers/available

Get available tickers from forecasting API.

**Response:**
```json
{
  "tickers": ["BTC-USD", "ETH-USD", "SOL-USD"],
  "count": 3
}
```

#### GET /api/tickers/{ticker}/forecast

Get forecast for a specific ticker.

**Parameters:**
- `ticker` (string): Ticker symbol
- `interval` (string, optional): Forecast interval (default: "hours")

**Response:**
```json
{
  "ticker": "BTC-USD",
  "forecast": [
    {
      "timestamp": "2024-01-01T01:00:00Z",
      "price": 51000.0,
      "confidence": 0.85
    }
  ],
  "interval": "hours",
  "model_version": "v1.0.0"
}
```

#### GET /api/tickers/{ticker}/recommendation

Get trading recommendation for a specific ticker.

**Parameters:**
- `ticker` (string): Ticker symbol
- `interval` (string, optional): Recommendation interval (default: "hours")

**Response:**
```json
{
  "ticker": "BTC-USD",
  "recommendation": "BUY",
  "confidence": 0.85,
  "reasoning": "Strong bullish signals from technical analysis",
  "price_target": 55000.0,
  "stop_loss": 48000.0,
  "interval": "hours",
  "model_version": "v1.0.0"
}
```

### User Preferences

#### GET /api/user/preferences

Get user trading preferences.

**Response:**
```json
{
  "preferences": {
    "risk_tolerance": "medium",
    "max_position_size": 0.20,
    "max_daily_loss": 0.05,
    "min_confidence": 0.70,
    "preferred_exchanges": ["DEX", "MEXC"],
    "notification_settings": {
      "email": true,
      "push": false,
      "sms": false
    }
  }
}
```

#### PUT /api/user/preferences

Update user trading preferences.

**Request Body:**
```json
{
  "risk_tolerance": "high",
  "max_position_size": 0.30,
  "max_daily_loss": 0.10,
  "min_confidence": 0.80,
  "preferred_exchanges": ["DEX"],
  "notification_settings": {
    "email": true,
    "push": true,
    "sms": false
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Preferences updated successfully"
}
```

### Legacy Endpoints

#### GET /api/portfolio

Get current portfolio state (legacy endpoint).

**Response:**
```json
{
  "balance_usdc": 5000.0,
  "holdings": {
    "BTC": 0.1,
    "ETH": 1.0
  },
  "total_value_usdc": 10000.0,
  "daily_pnl": 100.0,
  "total_pnl": 1000.0,
  "positions": []
}
```

#### GET /api/performance

Get performance metrics.

**Response:**
```json
{
  "total_pnl": 1000.0,
  "daily_pnl": 100.0,
  "win_rate": 0.65,
  "sharpe_ratio": 1.2,
  "max_drawdown": 0.15,
  "total_trades": 100,
  "winning_trades": 65,
  "losing_trades": 35
}
```

#### GET /api/signals/{ticker}

Get latest signals for a ticker.

**Parameters:**
- `ticker` (string): Ticker symbol

**Response:**
```json
{
  "dqn": {
    "recommendation": "BUY",
    "confidence": 0.85,
    "timestamp": "2024-01-01T00:00:00Z"
  },
  "chart": {
    "rsi": 45.0,
    "macd": 100.0,
    "bollinger_position": 0.6,
    "recommendation": "HOLD"
  },
  "news": {
    "sentiment": 0.7,
    "confidence": 0.8,
    "timestamp": "2024-01-01T00:00:00Z"
  },
  "risk": {
    "risk_score": 0.3,
    "position_size": 0.1,
    "recommendation": "APPROVED"
  }
}
```

#### GET /api/market-data/{ticker}

Get latest market data for a ticker.

**Parameters:**
- `ticker` (string): Ticker symbol

**Response:**
```json
{
  "ticker": "BTC",
  "price": 50000.0,
  "volume": 1000000.0,
  "change_24h": 0.05,
  "high_24h": 52000.0,
  "low_24h": 48000.0,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### GET /api/validations/pending

Get pending human validation requests.

**Response:**
```json
{
  "pending_validations": [
    {
      "request_id": "validation-uuid",
      "ticker": "BTC",
      "action": "BUY",
      "quantity": 1.0,
      "reason": "High risk trade",
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

#### POST /api/validations/{request_id}/respond

Respond to a human validation request.

**Parameters:**
- `request_id` (string): Validation request identifier

**Request Body:**
```json
{
  "approved": true,
  "reason": "Risk within acceptable limits"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Validation response sent"
}
```

#### GET /api/history/trades

Get trade history.

**Parameters:**
- `limit` (int, optional): Number of trades to return (default: 50)

**Response:**
```json
{
  "trades": [
    {
      "id": "trade-uuid",
      "ticker": "BTC",
      "action": "BUY",
      "quantity": 0.1,
      "price": 50000.0,
      "timestamp": "2024-01-01T00:00:00Z"
    }
  ],
  "count": 1
}
```

#### GET /api/history/trades/{ticker}

Get trade history for a specific ticker.

**Parameters:**
- `ticker` (string): Ticker symbol
- `limit` (int, optional): Number of trades to return (default: 20)

**Response:**
```json
{
  "ticker": "BTC",
  "trades": [
    {
      "id": "trade-uuid",
      "action": "BUY",
      "quantity": 0.1,
      "price": 50000.0,
      "timestamp": "2024-01-01T00:00:00Z"
    }
  ],
  "count": 1
}
```

#### GET /api/agents/status

Get status of all agents.

**Response:**
```json
{
  "agents": {
    "memory": {
      "status": "online",
      "last_heartbeat": "2024-01-01T00:00:00Z"
    },
    "dqn": {
      "status": "online",
      "last_heartbeat": "2024-01-01T00:00:00Z"
    },
    "chart": {
      "status": "online",
      "last_heartbeat": "2024-01-01T00:00:00Z"
    },
    "risk": {
      "status": "online",
      "last_heartbeat": "2024-01-01T00:00:00Z"
    },
    "news": {
      "status": "online",
      "last_heartbeat": "2024-01-01T00:00:00Z"
    },
    "copytrade": {
      "status": "online",
      "last_heartbeat": "2024-01-01T00:00:00Z"
    },
    "orchestrator": {
      "status": "online",
      "last_heartbeat": "2024-01-01T00:00:00Z"
    }
  }
}
```

#### POST /api/wallets/track

Add a wallet to track for copy trading.

**Request Body:**
```json
{
  "address": "0x1234567890abcdef",
  "chain": "ethereum",
  "name": "Whale Wallet"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Wallet added to tracking"
}
```

#### GET /api/wallets/tracked

Get list of tracked wallets.

**Response:**
```json
{
  "wallets": [
    {
      "address": "0x1234567890abcdef",
      "chain": "ethereum",
      "name": "Whale Wallet",
      "added_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

#### GET /api/risk/portfolio

Get portfolio-level risk metrics.

**Response:**
```json
{
  "total_risk_score": 0.3,
  "var_95": 500.0,
  "max_drawdown": 0.15,
  "correlation_risk": 0.4,
  "concentration_risk": 0.2,
  "liquidity_risk": 0.1
}
```

#### GET /api/risk/{ticker}

Get risk metrics for a specific ticker.

**Parameters:**
- `ticker` (string): Ticker symbol

**Response:**
```json
{
  "ticker": "BTC",
  "risk_score": 0.3,
  "volatility": 0.4,
  "beta": 1.2,
  "var_95": 100.0,
  "max_loss": 200.0
}
```

#### GET /api/news/market-sentiment

Get overall market sentiment from news analysis.

**Response:**
```json
{
  "overall_sentiment": 0.7,
  "confidence": 0.8,
  "sources": [
    {
      "name": "CoinDesk",
      "sentiment": 0.8,
      "weight": 0.3
    }
  ],
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### GET /api/config

Get system configuration (non-sensitive).

**Response:**
```json
{
  "initial_capital": 10000.0,
  "max_position_size": 0.20,
  "max_daily_loss": 0.05,
  "max_drawdown": 0.15,
  "min_confidence": 0.70,
  "trading_fee": 0.001,
  "supported_assets": ["BTC", "ETH", "SOL"],
  "observation_interval": 300,
  "decision_interval": 1800,
  "forecast_interval": 3600
}
```

## WebSocket Endpoints

### /ws/trades

Real-time trade updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/trades');
ws.onmessage = (event) => {
  const trade = JSON.parse(event.data);
  console.log('New trade:', trade);
};
```

**Message Format:**
```json
{
  "type": "trade",
  "data": {
    "id": "trade-uuid",
    "ticker": "BTC",
    "action": "BUY",
    "quantity": 0.1,
    "price": 50000.0,
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

### /ws/signals

Real-time signal updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/signals');
ws.onmessage = (event) => {
  const signal = JSON.parse(event.data);
  console.log('New signal:', signal);
};
```

**Message Format:**
```json
{
  "type": "signal",
  "data": {
    "ticker": "BTC",
    "agent": "dqn",
    "recommendation": "BUY",
    "confidence": 0.85,
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |

## Rate Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/health` | 1 req/s | 60s |
| `/api/*` | 10 req/s | 60s |
| `/api/portfolios/*/trades` | 5 req/s | 60s |
| `/api/validations/*` | 2 req/s | 60s |

## SDK Examples

### Python

```python
import httpx

class TradingClient:
    def __init__(self, base_url="http://localhost:8000", token=None):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    async def get_portfolio(self):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/portfolios/main",
                headers=self.headers
            )
            return response.json()
    
    async def execute_trade(self, ticker, action, quantity):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/portfolios/main/trades",
                json={
                    "ticker": ticker,
                    "action": action,
                    "quantity": quantity
                },
                headers=self.headers
            )
            return response.json()
```

### JavaScript

```javascript
class TradingClient {
  constructor(baseUrl = 'http://localhost:8000', token = null) {
    this.baseUrl = baseUrl;
    this.headers = token ? { 'Authorization': `Bearer ${token}` } : {};
  }
  
  async getPortfolio() {
    const response = await fetch(`${this.baseUrl}/api/portfolios/main`, {
      headers: this.headers
    });
    return response.json();
  }
  
  async executeTrade(ticker, action, quantity) {
    const response = await fetch(`${this.baseUrl}/api/portfolios/main/trades`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.headers
      },
      body: JSON.stringify({
        ticker,
        action,
        quantity
      })
    });
    return response.json();
  }
}
```
