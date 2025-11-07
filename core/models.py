"""
Data models for the Agentic Trading System.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TradeAction(str, Enum):
    """Trading action types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalType(str, Enum):
    """Signal types from different agents."""
    DQN_PREDICTION = "DQN_PREDICTION"
    TECHNICAL_ANALYSIS = "TECHNICAL_ANALYSIS"
    NEWS_SENTIMENT = "NEWS_SENTIMENT"
    RISK_ALERT = "RISK_ALERT"
    COPY_TRADE = "COPY_TRADE"
    MEMORY_INSIGHT = "MEMORY_INSIGHT"


class AgentType(str, Enum):
    """Agent types in the system."""
    MEMORY = "MEMORY"
    DQN = "DQN"
    CHART = "CHART"
    RISK = "RISK"
    NEWS = "NEWS"
    COPYTRADE = "COPYTRADE"
    ORCHESTRATOR = "ORCHESTRATOR"


class MessageType(str, Enum):
    """Message types for inter-agent communication."""
    MARKET_DATA_UPDATE = "MARKET_DATA_UPDATE"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    RISK_ALERT = "RISK_ALERT"
    TRADE_EXECUTED = "TRADE_EXECUTED"
    NEWS_EVENT = "NEWS_EVENT"
    AGENT_HEARTBEAT = "AGENT_HEARTBEAT"
    HUMAN_VALIDATION_REQUEST = "HUMAN_VALIDATION_REQUEST"
    HUMAN_VALIDATION_RESPONSE = "HUMAN_VALIDATION_RESPONSE"


class MarketData(BaseModel):
    """Market data for a specific asset."""
    ticker: str
    price: float
    volume: float
    timestamp: datetime
    interval: str  # minutes, hours, days
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None


class DQNPrediction(BaseModel):
    """DQN prediction from MCP API."""
    ticker: str
    action: TradeAction
    confidence: float
    forecast_price: Optional[float] = None
    forecast_horizon: str  # e.g., "T+14days"
    timestamp: datetime


class TechnicalSignal(BaseModel):
    """Technical analysis signal."""
    ticker: str
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    volume_sma: Optional[float] = None
    recommendation: TradeAction
    strength: float  # 0-1
    timestamp: datetime


class NewsSentiment(BaseModel):
    """News sentiment analysis result."""
    ticker: Optional[str] = None  # None for general market sentiment
    sentiment_score: float  # -1 to 1
    confidence: float
    summary: str
    sources: List[str]
    timestamp: datetime


class RiskMetrics(BaseModel):
    """Risk metrics for portfolio or specific position."""
    ticker: Optional[str] = None  # None for portfolio-level metrics
    var_95: Optional[float] = None  # Value at Risk 95%
    cvar_95: Optional[float] = None  # Conditional VaR 95%
    position_size: Optional[float] = None
    max_position_size: Optional[float] = None
    current_drawdown: Optional[float] = None
    max_drawdown: Optional[float] = None
    daily_pnl: Optional[float] = None
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    warnings: List[str] = []
    risk_score: Optional[float] = None
    risk_metric: Optional[float] = None
    stop_loss_upper: Optional[float] = None
    stop_loss_lower: Optional[float] = None
    timestamp: datetime


class CopyTradeSignal(BaseModel):
    """Copy trade signal from on-chain analysis."""
    wallet_address: str
    blockchain: str  # BSC, ETH, SOL, etc.
    ticker: str
    action: TradeAction
    amount: float
    price: float
    wallet_performance: Optional[float] = None  # Historical performance
    confidence: float
    timestamp: datetime


class AgentSignal(BaseModel):
    """Generic signal from any agent."""
    agent_type: AgentType
    signal_type: SignalType
    ticker: Optional[str] = None
    action: Optional[TradeAction] = None
    confidence: float
    data: Dict  # Agent-specific data
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TradeDecision(BaseModel):
    """Final trade decision from orchestrator."""
    ticker: str
    action: TradeAction
    quantity: float
    expected_price: Optional[float] = None
    confidence: float
    reasoning: str
    contributing_signals: List[AgentSignal]
    risk_approved: bool
    requires_human_validation: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TradeExecution(BaseModel):
    """Trade execution result."""
    decision_id: str
    ticker: str
    action: TradeAction
    quantity: float
    executed_price: float
    total_cost: float
    fee: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Portfolio(BaseModel):
    """Current portfolio state."""
    balance_usdc: float
    holdings: Dict[str, float]  # ticker -> quantity
    total_value_usdc: float
    daily_pnl: float
    total_pnl: float
    positions: List[Dict]  # Detailed position info
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentMessage(BaseModel):
    """Message for inter-agent communication."""
    message_type: MessageType
    sender: AgentType
    recipient: Optional[AgentType] = None  # None for broadcast
    payload: Dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None


class MemoryRecord(BaseModel):
    """Memory record for historical tracking."""
    record_type: str  # trade, signal, performance, etc.
    ticker: Optional[str] = None
    data: Dict
    outcome: Optional[str] = None
    performance_impact: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PerformanceMetrics(BaseModel):
    """Performance metrics for the trading system."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: float
    current_drawdown: float
    roi: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HumanValidationRequest(BaseModel):
    """Request for human validation of a trade decision."""
    request_id: str
    decision: TradeDecision
    urgency: str  # LOW, MEDIUM, HIGH
    reason: str
    timeout_seconds: int = 300  # 5 minutes default
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HumanValidationResponse(BaseModel):
    """Response to human validation request."""
    request_id: str
    approved: bool
    feedback: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TradeMemoryEntry(BaseModel):
    """FIFO trade tape entry stored by the memory agent."""
    trade_id: str
    ticker: str
    action: TradeAction
    quantity: float
    price: float
    pnl: Optional[float] = None
    status: Optional[str] = None  # WIN / LOSS / BREAKEVEN / UNKNOWN
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class NewsMemoryEntry(BaseModel):
    """Recency-weighted news sentiment memory entry."""
    news_id: str
    ticker: Optional[str] = None
    sentiment_score: float
    confidence: float
    summary: str
    sources: List[str]
    weight: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GraphMemoryNode(BaseModel):
    """Node within the long-term knowledge graph."""
    node_id: str
    label: str
    node_type: str
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class GraphMemoryEdge(BaseModel):
    """Relationship edge within the long-term knowledge graph."""
    edge_id: str
    source: str
    target: str
    relation: str
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

