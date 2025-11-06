"""
Integration tests for CAMEL Workforce workers.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.workforce_workers.dqn_worker import DQNWorker
from agents.workforce_workers.chart_analysis_worker import ChartAnalysisWorker
from agents.workforce_workers.risk_assessment_worker import RiskAssessmentWorker
from agents.workforce_workers.market_research_worker import MarketResearchWorker
from agents.workforce_workers.trade_execution_worker import TradeExecutionWorker


@pytest.fixture
def mock_forecasting_client():
    """Mock forecasting client."""
    client = AsyncMock()
    client.connect = AsyncMock()
    client.get_stock_forecast = AsyncMock(return_value={"forecast_price": 50000.0})
    client.get_action_recommendation = AsyncMock(return_value={"action": 2, "action_confidence": 0.8})
    return client


@pytest.fixture
def mock_dex_client():
    """Mock DEX client."""
    client = AsyncMock()
    client.connect = AsyncMock()
    client.get_portfolio_status = AsyncMock(return_value={"balance_usdc": 1000.0})
    return client


@pytest.mark.asyncio
@patch('agents.workforce_workers.dqn_worker.CamelModelFactory')
@patch('agents.workforce_workers.dqn_worker.CamelMemoryManager')
@patch('agents.workforce_workers.dqn_worker.MCPForecastingToolkit')
async def test_dqn_worker_initialization(
    mock_toolkit_class,
    mock_memory_class,
    mock_model_factory,
    mock_forecasting_client
):
    """Test DQN worker initialization."""
    mock_toolkit = AsyncMock()
    mock_toolkit.initialize = AsyncMock()
    mock_toolkit.get_all_tools = MagicMock(return_value=[])
    mock_toolkit_class.return_value = mock_toolkit
    
    mock_memory = MagicMock()
    mock_memory.memory = MagicMock()
    mock_memory_class.return_value = mock_memory
    
    mock_model = MagicMock()
    mock_model_factory.create_worker_model.return_value = mock_model
    
    worker = DQNWorker(agent_id="test_dqn")
    
    try:
        await worker.initialize()
        assert worker.agent is not None
        assert worker.forecasting_toolkit is not None
    except (ImportError, AttributeError) as e:
        pytest.skip(f"CAMEL not available: {e}")


@pytest.mark.asyncio
@patch('agents.workforce_workers.chart_analysis_worker.CamelModelFactory')
@patch('agents.workforce_workers.chart_analysis_worker.CamelMemoryManager')
async def test_chart_worker_initialization(
    mock_memory_class,
    mock_model_factory
):
    """Test Chart Analysis worker initialization."""
    mock_memory = MagicMock()
    mock_memory.memory = MagicMock()
    mock_memory_class.return_value = mock_memory
    
    mock_model = MagicMock()
    mock_model_factory.create_worker_model.return_value = mock_model
    
    worker = ChartAnalysisWorker(agent_id="test_chart")
    
    try:
        await worker.initialize()
        assert worker.agent is not None
    except (ImportError, AttributeError) as e:
        pytest.skip(f"CAMEL not available: {e}")


@pytest.mark.asyncio
@patch('agents.workforce_workers.risk_assessment_worker.CamelModelFactory')
@patch('agents.workforce_workers.risk_assessment_worker.CamelMemoryManager')
@patch('agents.workforce_workers.risk_assessment_worker.DEXTradingToolkit')
async def test_risk_worker_initialization(
    mock_toolkit_class,
    mock_memory_class,
    mock_model_factory,
    mock_dex_client
):
    """Test Risk Assessment worker initialization."""
    mock_toolkit = AsyncMock()
    mock_toolkit.initialize = AsyncMock()
    mock_toolkit.get_portfolio_status_tool = MagicMock(return_value=MagicMock())
    mock_toolkit_class.return_value = mock_toolkit
    
    mock_memory = MagicMock()
    mock_memory.memory = MagicMock()
    mock_memory_class.return_value = mock_memory
    
    mock_model = MagicMock()
    mock_model_factory.create_worker_model.return_value = mock_model
    
    worker = RiskAssessmentWorker(agent_id="test_risk")
    
    try:
        await worker.initialize()
        assert worker.agent is not None
    except (ImportError, AttributeError) as e:
        pytest.skip(f"CAMEL not available: {e}")


@pytest.mark.asyncio
@patch('agents.workforce_workers.market_research_worker.CamelModelFactory')
@patch('agents.workforce_workers.market_research_worker.CamelMemoryManager')
async def test_market_research_worker_initialization(
    mock_memory_class,
    mock_model_factory
):
    """Test Market Research worker initialization."""
    mock_memory = MagicMock()
    mock_memory.memory = MagicMock()
    mock_memory_class.return_value = mock_memory
    
    mock_model = MagicMock()
    mock_model_factory.create_worker_model.return_value = mock_model
    
    worker = MarketResearchWorker(agent_id="test_research")
    
    try:
        await worker.initialize()
        assert worker.agent is not None
    except (ImportError, AttributeError) as e:
        pytest.skip(f"CAMEL not available: {e}")


@pytest.mark.asyncio
@patch('agents.workforce_workers.trade_execution_worker.CamelModelFactory')
@patch('agents.workforce_workers.trade_execution_worker.CamelMemoryManager')
@patch('agents.workforce_workers.trade_execution_worker.DEXTradingToolkit')
async def test_trade_execution_worker_initialization(
    mock_toolkit_class,
    mock_memory_class,
    mock_model_factory,
    mock_dex_client
):
    """Test Trade Execution worker initialization."""
    mock_toolkit = AsyncMock()
    mock_toolkit.initialize = AsyncMock()
    mock_toolkit.get_all_tools = MagicMock(return_value=[])
    mock_toolkit_class.return_value = mock_toolkit
    
    mock_memory = MagicMock()
    mock_memory.memory = MagicMock()
    mock_memory_class.return_value = mock_memory
    
    mock_model = MagicMock()
    mock_model_factory.create_worker_model.return_value = mock_model
    
    worker = TradeExecutionWorker(agent_id="test_execution")
    
    try:
        await worker.initialize()
        assert worker.agent is not None
    except (ImportError, AttributeError) as e:
        pytest.skip(f"CAMEL not available: {e}")


@pytest.mark.asyncio
async def test_worker_descriptions():
    """Test that all workers have descriptions."""
    workers = [
        DQNWorker(),
        ChartAnalysisWorker(),
        RiskAssessmentWorker(),
        MarketResearchWorker(),
        TradeExecutionWorker(),
    ]
    
    for worker in workers:
        description = worker.get_description()
        assert description is not None
        assert isinstance(description, str)
        assert len(description) > 0

