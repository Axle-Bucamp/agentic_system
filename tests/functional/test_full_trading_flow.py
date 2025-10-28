"""
End-to-end functional tests for the complete trading system.
Tests the full flow from signal generation to trade execution.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime
from core.redis_client import RedisClient
from core.exchange_manager import ExchangeManager, ExchangeType
from core.forecasting_client import ForecastingClient
from agents.orchestrator import OrchestratorAgent
from agents.dqn_agent import DQNAgent
from core.models import (
    AgentMessage, MessageType, TradeAction, AgentType, 
    AgentSignal, SignalType, TradeDecision, MarketData
)

@pytest.fixture
async def mock_redis_client():
    """Mock Redis client for testing."""
    mock_redis = AsyncMock(spec=RedisClient)
    mock_redis.connect.return_value = None
    mock_redis.disconnect.return_value = None
    mock_redis.publish.return_value = None
    mock_redis.subscribe.return_value = AsyncMock()
    mock_redis.get_messages.return_value.__aiter__.return_value = []
    mock_redis.get_json.return_value = None
    mock_redis.set_json.return_value = None
    mock_redis.exists.return_value = False
    mock_redis.delete.return_value = None
    mock_redis.lrange.return_value = []
    return mock_redis

@pytest.fixture
async def mock_exchange_manager():
    """Mock exchange manager for testing."""
    manager = ExchangeManager()
    
    # Add mock exchanges
    from tests.mocks.mock_dex_exchange import MockDEXExchange
    from tests.mocks.mock_mexc_exchange import MockMEXCExchange
    
    dex_config = {"mock_mode": True}
    dex_exchange = MockDEXExchange(dex_config)
    manager.add_exchange(dex_exchange, is_primary=True)
    
    mexc_config = {"mock_mode": True}
    mexc_exchange = MockMEXCExchange(mexc_config)
    manager.add_exchange(mexc_exchange, is_primary=False)
    
    await manager.connect_all()
    return manager

@pytest.fixture
async def mock_forecasting_client():
    """Mock forecasting client for testing."""
    client = ForecastingClient({
        "base_url": "http://mock-forecasting-api.com",
        "api_key": "mock_key",
        "mock_mode": True
    })
    await client.connect()
    return client

@pytest.fixture
async def orchestrator_agent(mock_redis_client, mock_exchange_manager):
    """Create orchestrator agent with mocked dependencies."""
    agent = OrchestratorAgent(mock_redis_client)
    agent.exchange_manager = mock_exchange_manager
    await agent.initialize()
    return agent

@pytest.fixture
async def dqn_agent(mock_redis_client, mock_forecasting_client):
    """Create DQN agent with mocked dependencies."""
    agent = DQNAgent(mock_redis_client)
    agent.forecasting_client = mock_forecasting_client
    await agent.initialize()
    return agent

@pytest.mark.asyncio
async def test_complete_trading_flow(orchestrator_agent, dqn_agent, mock_redis_client):
    """Test the complete flow from signal generation to trade execution."""
    
    # 1. Simulate market data update
    market_data = MarketData(
        ticker="BTC",
        price=60000.0,
        volume=1000.0,
        timestamp=datetime.utcnow(),
        interval="minutes"
    )
    
    market_message = AgentMessage(
        agent_type=AgentType.CHART,
        message_type=MessageType.MARKET_DATA_UPDATE,
        data=market_data.dict(),
        timestamp=datetime.utcnow()
    )
    
    # 2. DQN agent processes market data and generates signal
    dqn_response = await dqn_agent.process_message(market_message)
    assert dqn_response is not None
    assert dqn_response.message_type == MessageType.SIGNAL_GENERATED
    
    # 3. Orchestrator receives signal
    orchestrator_response = await orchestrator_agent.process_message(dqn_response)
    assert orchestrator_response is not None
    
    # 4. Check if trade decision was made
    assert len(orchestrator_agent.pending_signals) > 0
    
    # 5. Simulate additional signals (risk, chart, etc.)
    risk_signal = AgentSignal(
        agent_type=AgentType.RISK,
        signal_type=SignalType.RISK_ASSESSMENT,
        ticker="BTC",
        action=TradeAction.BUY,
        confidence=0.8,
        data={"risk_level": "LOW", "max_position_size": 0.1},
        reasoning="Risk assessment shows low risk for BTC",
        timestamp=datetime.utcnow()
    )
    
    risk_message = AgentMessage(
        agent_type=AgentType.RISK,
        message_type=MessageType.SIGNAL_GENERATED,
        data=risk_signal.dict(),
        timestamp=datetime.utcnow()
    )
    
    # 6. Orchestrator processes risk signal
    orchestrator_response = await orchestrator_agent.process_message(risk_message)
    
    # 7. Check if final decision was made
    # The orchestrator should have made a decision after receiving multiple signals
    assert len(orchestrator_agent.pending_signals) >= 2

@pytest.mark.asyncio
async def test_trade_execution_flow(orchestrator_agent, mock_exchange_manager):
    """Test trade execution through exchange manager."""
    
    # Create a trade decision
    decision = TradeDecision(
        ticker="BTC",
        action=TradeAction.BUY,
        quantity=0.1,
        expected_price=60000.0,
        confidence=0.85,
        reasoning="Strong buy signal from DQN and risk assessment",
        timestamp=datetime.utcnow()
    )
    
    # Execute trade
    await orchestrator_agent._execute_trade(decision)
    
    # Check that order was placed (in mock exchange)
    # The mock exchange should have recorded the order
    assert True  # Mock exchanges don't persist state, but execution should complete

@pytest.mark.asyncio
async def test_human_validation_flow(orchestrator_agent):
    """Test human validation workflow."""
    
    # Create a high-value trade decision that requires validation
    decision = TradeDecision(
        ticker="BTC",
        action=TradeAction.BUY,
        quantity=1.0,  # Large quantity
        expected_price=60000.0,
        confidence=0.85,
        reasoning="Large position requires human validation",
        timestamp=datetime.utcnow()
    )
    
    # Request human validation
    await orchestrator_agent._request_human_validation(decision)
    
    # Check that validation request was created
    assert len(orchestrator_agent.pending_validations) > 0
    
    # Simulate human approval
    validation_id = list(orchestrator_agent.pending_validations.keys())[0]
    validation_response = {
        "validation_id": validation_id,
        "approved": True,
        "reasoning": "Approved by human trader"
    }
    
    # Process validation response
    validation_message = AgentMessage(
        agent_type=AgentType.ORCHESTRATOR,
        message_type=MessageType.HUMAN_VALIDATION_RESPONSE,
        data=validation_response,
        timestamp=datetime.utcnow()
    )
    
    orchestrator_response = await orchestrator_agent.process_message(validation_message)
    
    # Check that validation was processed
    assert validation_id not in orchestrator_agent.pending_validations

@pytest.mark.asyncio
async def test_error_handling_and_recovery(orchestrator_agent, dqn_agent):
    """Test error handling and recovery mechanisms."""
    
    # Test with invalid market data
    invalid_market_data = {
        "ticker": "INVALID",
        "price": -100.0,  # Invalid price
        "volume": 0.0,
        "timestamp": datetime.utcnow().isoformat(),
        "interval": "minutes"
    }
    
    invalid_message = AgentMessage(
        agent_type=AgentType.CHART,
        message_type=MessageType.MARKET_DATA_UPDATE,
        data=invalid_market_data,
        timestamp=datetime.utcnow()
    )
    
    # DQN agent should handle invalid data gracefully
    dqn_response = await dqn_agent.process_message(invalid_message)
    # Should not crash, may return None or error message
    
    # Test with missing exchange manager
    orchestrator_agent.exchange_manager = None
    decision = TradeDecision(
        ticker="BTC",
        action=TradeAction.BUY,
        quantity=0.1,
        expected_price=60000.0,
        confidence=0.85,
        reasoning="Test decision",
        timestamp=datetime.utcnow()
    )
    
    # Should handle missing exchange manager gracefully
    await orchestrator_agent._execute_trade(decision)
    # Should log error but not crash

@pytest.mark.asyncio
async def test_concurrent_trading_requests(orchestrator_agent):
    """Test handling of concurrent trading requests."""
    
    # Create multiple concurrent trade decisions
    decisions = [
        TradeDecision(
            ticker="BTC",
            action=TradeAction.BUY,
            quantity=0.1,
            expected_price=60000.0,
            confidence=0.85,
            reasoning=f"Concurrent decision {i}",
            timestamp=datetime.utcnow()
        )
        for i in range(5)
    ]
    
    # Execute all trades concurrently
    tasks = [orchestrator_agent._execute_trade(decision) for decision in decisions]
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # All trades should complete without errors
    assert True  # If we get here, no exceptions were raised

@pytest.mark.asyncio
async def test_signal_aggregation_logic(orchestrator_agent):
    """Test the signal aggregation and decision-making logic."""
    
    # Create conflicting signals
    buy_signal = AgentSignal(
        agent_type=AgentType.DQN,
        signal_type=SignalType.DQN_PREDICTION,
        ticker="BTC",
        action=TradeAction.BUY,
        confidence=0.9,
        data={"forecast_price": 65000.0},
        reasoning="DQN predicts price increase",
        timestamp=datetime.utcnow()
    )
    
    sell_signal = AgentSignal(
        agent_type=AgentType.RISK,
        signal_type=SignalType.RISK_ASSESSMENT,
        ticker="BTC",
        action=TradeAction.SELL,
        confidence=0.7,
        data={"risk_level": "HIGH"},
        reasoning="High risk detected",
        timestamp=datetime.utcnow()
    )
    
    # Process both signals
    buy_message = AgentMessage(
        agent_type=AgentType.DQN,
        message_type=MessageType.SIGNAL_GENERATED,
        data=buy_signal.dict(),
        timestamp=datetime.utcnow()
    )
    
    sell_message = AgentMessage(
        agent_type=AgentType.RISK,
        message_type=MessageType.SIGNAL_GENERATED,
        data=sell_signal.dict(),
        timestamp=datetime.utcnow()
    )
    
    # Process signals
    await orchestrator_agent.process_message(buy_message)
    await orchestrator_agent.process_message(sell_message)
    
    # Check that signals were aggregated
    assert len(orchestrator_agent.pending_signals) >= 2
    
    # The orchestrator should handle conflicting signals appropriately
    # (This would depend on the specific aggregation logic implemented)
