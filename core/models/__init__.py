"""
Model configuration modules.
"""
from core.models.camel_models import CamelModelFactory
# Import from core.models (the models.py file) for backward compatibility
# Use importlib to avoid circular import issues
import importlib.util
from pathlib import Path

# Load the models.py file directly to avoid circular imports
_models_py_path = Path(__file__).parent.parent / "models.py"
if _models_py_path.exists():
    _spec = importlib.util.spec_from_file_location("_core_models_py", _models_py_path)
    _models_py = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_models_py)
    
    AgentType = _models_py.AgentType
    AgentMessage = _models_py.AgentMessage
    MessageType = _models_py.MessageType
    TradeAction = _models_py.TradeAction
    SignalType = _models_py.SignalType
    MarketData = _models_py.MarketData
    DQNPrediction = _models_py.DQNPrediction
    AgentSignal = _models_py.AgentSignal
    TechnicalSignal = _models_py.TechnicalSignal
    RiskMetrics = _models_py.RiskMetrics
    TradeDecision = _models_py.TradeDecision
    # Check if MemoryRecord exists (it might be in models.py)
    if hasattr(_models_py, 'MemoryRecord'):
        MemoryRecord = _models_py.MemoryRecord
    else:
        # Import from CAMEL if not in models.py
        try:
            from camel.memories import MemoryRecord as CamelMemoryRecord
            MemoryRecord = CamelMemoryRecord
        except ImportError:
            MemoryRecord = None
    # Additional types
    if hasattr(_models_py, 'PerformanceMetrics'):
        PerformanceMetrics = _models_py.PerformanceMetrics
    else:
        PerformanceMetrics = None
    if hasattr(_models_py, 'NewsSentiment'):
        NewsSentiment = _models_py.NewsSentiment
    else:
        NewsSentiment = None
    if hasattr(_models_py, 'CopyTradeSignal'):
        CopyTradeSignal = _models_py.CopyTradeSignal
    else:
        CopyTradeSignal = None
    if hasattr(_models_py, 'HumanValidationRequest'):
        HumanValidationRequest = _models_py.HumanValidationRequest
    else:
        HumanValidationRequest = None
    if hasattr(_models_py, 'HumanValidationResponse'):
        HumanValidationResponse = _models_py.HumanValidationResponse
    else:
        HumanValidationResponse = None
    if hasattr(_models_py, 'TradeExecution'):
        TradeExecution = _models_py.TradeExecution
    else:
        TradeExecution = None
    if hasattr(_models_py, 'Portfolio'):
        Portfolio = _models_py.Portfolio
    else:
        Portfolio = None
    if hasattr(_models_py, 'GraphMemoryNode'):
        GraphMemoryNode = _models_py.GraphMemoryNode
    else:
        GraphMemoryNode = None
    if hasattr(_models_py, 'GraphMemoryEdge'):
        GraphMemoryEdge = _models_py.GraphMemoryEdge
    else:
        GraphMemoryEdge = None
    if hasattr(_models_py, 'TradeMemoryEntry'):
        TradeMemoryEntry = _models_py.TradeMemoryEntry
    else:
        TradeMemoryEntry = None
    if hasattr(_models_py, 'NewsMemoryEntry'):
        NewsMemoryEntry = _models_py.NewsMemoryEntry
    else:
        NewsMemoryEntry = None
    # ExchangeType is in exchange_interface, not models.py
    try:
        from core.exchange_interface import ExchangeType
    except ImportError:
        ExchangeType = None
else:
    # Fallback: raise error if models.py not found
    raise ImportError(f"Could not find models.py at {_models_py_path}")

__all__ = [
    "CamelModelFactory",
    "AgentType",
    "AgentMessage",
    "MessageType",
    "TradeAction",
    "SignalType",
    "MarketData",
    "DQNPrediction",
    "AgentSignal",
    "TechnicalSignal",
    "RiskMetrics",
    "TradeDecision",
]
if MemoryRecord is not None:
    __all__.append("MemoryRecord")
if PerformanceMetrics is not None:
    __all__.append("PerformanceMetrics")
if NewsSentiment is not None:
    __all__.append("NewsSentiment")
if CopyTradeSignal is not None:
    __all__.append("CopyTradeSignal")
if HumanValidationRequest is not None:
    __all__.append("HumanValidationRequest")
if HumanValidationResponse is not None:
    __all__.append("HumanValidationResponse")
if TradeExecution is not None:
    __all__.append("TradeExecution")
if Portfolio is not None:
    __all__.append("Portfolio")
if ExchangeType is not None:
    __all__.append("ExchangeType")
if 'GraphMemoryNode' in globals() and GraphMemoryNode is not None:
    __all__.append("GraphMemoryNode")
if 'GraphMemoryEdge' in globals() and GraphMemoryEdge is not None:
    __all__.append("GraphMemoryEdge")
if 'TradeMemoryEntry' in globals() and TradeMemoryEntry is not None:
    __all__.append("TradeMemoryEntry")
if 'NewsMemoryEntry' in globals() and NewsMemoryEntry is not None:
    __all__.append("NewsMemoryEntry")

