"""
Workforce worker agents for CAMEL Workforce orchestration.
"""
from agents.workforce_workers.dqn_worker import DQNWorker
from agents.workforce_workers.chart_analysis_worker import ChartAnalysisWorker
from agents.workforce_workers.risk_assessment_worker import RiskAssessmentWorker
from agents.workforce_workers.market_research_worker import MarketResearchWorker
from agents.workforce_workers.trade_execution_worker import TradeExecutionWorker

__all__ = [
    "DQNWorker",
    "ChartAnalysisWorker",
    "RiskAssessmentWorker",
    "MarketResearchWorker",
    "TradeExecutionWorker",
]

