"""
Risk Analysis Agent - Evaluates and manages trading risks.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from core.models import (
    AgentType, AgentMessage, MessageType, RiskMetrics,
    TradeAction, AgentSignal, SignalType, TradeDecision
)
from core.config import settings
from core.logging import log
from agents.base_agent import BaseAgent
from core import asset_registry


class RiskAgent(BaseAgent):
    """Agent responsible for risk analysis and position management."""
    
    def __init__(self, redis_client):
        super().__init__(AgentType.RISK, redis_client)
        
    async def initialize(self):
        """Initialize risk agent."""
        log.info("Risk Agent initialized")
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages."""
        if message.message_type == MessageType.SIGNAL_GENERATED:
            # Evaluate risk for the signal
            await self._evaluate_signal_risk(message.payload)
        
        return None
    
    async def run_cycle(self):
        """Run periodic risk assessment."""
        log.debug("Risk Agent running cycle...")
        
        try:
            # Calculate portfolio-level risk metrics
            portfolio = await self.get_portfolio()
            if portfolio:
                risk_metrics = await self._calculate_portfolio_risk(portfolio)
                
                # Store risk metrics
                await self.redis.set_json(
                    "risk:portfolio",
                    risk_metrics.dict(),
                    expire=300
                )
                
                # Check for risk alerts
                if risk_metrics.risk_level in ["HIGH", "CRITICAL"]:
                    await self._send_risk_alert(risk_metrics)
            
            # Calculate per-asset risk metrics
            for ticker in asset_registry.get_assets():
                await self._calculate_asset_risk(ticker)
                
        except Exception as e:
            log.error(f"Risk Agent cycle error: {e}")
    
    def get_cycle_interval(self) -> int:
        return settings.get_agent_cycle_seconds(self.agent_type)
    
    async def _evaluate_signal_risk(self, signal_data: Dict):
        """Evaluate risk for a trading signal."""
        try:
            ticker = signal_data.get("ticker")
            action = signal_data.get("action")
            
            if not ticker or not action:
                return
            
            # Get current portfolio
            portfolio = await self.get_portfolio()
            if not portfolio:
                return
            
            # Calculate position risk
            risk_metrics = await self._calculate_position_risk(ticker, action, portfolio)
            
            # Send risk assessment
            risk_signal = AgentSignal(
                agent_type=self.agent_type,
                signal_type=SignalType.RISK_ALERT,
                ticker=ticker,
                action=None,
                confidence=1.0,
                data=risk_metrics.dict(),
                reasoning=self._build_risk_reasoning(risk_metrics)
            )
            
            await self.send_signal(risk_signal.dict())
            
        except Exception as e:
            log.error(f"Error evaluating signal risk: {e}")
    
    async def _calculate_portfolio_risk(self, portfolio: Dict) -> RiskMetrics:
        """Calculate portfolio-level risk metrics."""
        balance = portfolio.get("balance_usdc", 0)
        holdings = portfolio.get("holdings", {})
        total_value = portfolio.get("total_value_usdc", balance)
        
        # Get historical performance
        daily_pnl = portfolio.get("daily_pnl", 0)
        
        # Calculate drawdown
        peak_value = await self._get_peak_portfolio_value()
        current_drawdown = (peak_value - total_value) / peak_value if peak_value > 0 else 0
        
        # Determine risk level
        warnings = []
        risk_level = "LOW"
        
        # Check daily loss
        daily_loss_pct = abs(daily_pnl / total_value) if total_value > 0 else 0
        if daily_loss_pct > settings.max_daily_loss:
            warnings.append(f"Daily loss {daily_loss_pct:.1%} exceeds limit {settings.max_daily_loss:.1%}")
            risk_level = "CRITICAL"
        elif daily_loss_pct > settings.max_daily_loss * 0.7:
            warnings.append(f"Daily loss {daily_loss_pct:.1%} approaching limit")
            risk_level = "HIGH"
        
        # Check drawdown
        if current_drawdown > settings.max_drawdown:
            warnings.append(f"Drawdown {current_drawdown:.1%} exceeds limit {settings.max_drawdown:.1%}")
            risk_level = "CRITICAL"
        elif current_drawdown > settings.max_drawdown * 0.7:
            warnings.append(f"Drawdown {current_drawdown:.1%} approaching limit")
            if risk_level == "LOW":
                risk_level = "HIGH"
        
        # Check position concentration
        if holdings:
            max_position_value = 0
            for ticker, quantity in holdings.items():
                market_data = await self.get_market_data(ticker)
                if market_data:
                    price = market_data.get("price", 0)
                    position_value = quantity * price
                    max_position_value = max(max_position_value, position_value)
            
            max_position_pct = max_position_value / total_value if total_value > 0 else 0
            if max_position_pct > settings.max_position_size:
                warnings.append(f"Position concentration {max_position_pct:.1%} exceeds limit {settings.max_position_size:.1%}")
                if risk_level == "LOW":
                    risk_level = "MEDIUM"
        
        risk_score = self._map_risk_level_to_score(risk_level)
        return RiskMetrics(
            ticker=None,
            current_drawdown=current_drawdown,
            max_drawdown=settings.max_drawdown,
            daily_pnl=daily_pnl,
            risk_level=risk_level,
            warnings=warnings,
            risk_score=risk_score,
            risk_metric=round(risk_score * 100, 2),
            timestamp=datetime.utcnow()
        )
    
    async def _calculate_position_risk(self, ticker: str, action: str, portfolio: Dict) -> RiskMetrics:
        """Calculate risk metrics for a specific position."""
        balance = portfolio.get("balance_usdc", 0)
        holdings = portfolio.get("holdings", {})
        total_value = portfolio.get("total_value_usdc", balance)
        
        # Get current position
        current_quantity = holdings.get(ticker, 0)
        
        # Get market data
        market_data = await self.get_market_data(ticker)
        current_price = market_data.get("price", 0) if market_data else 0
        
        current_position_value = current_quantity * current_price
        current_position_pct = current_position_value / total_value if total_value > 0 else 0
        
        # Get max allowed position for this asset
        max_position_pct = settings.get_max_position_for_asset(ticker)
        
        warnings = []
        risk_level = "LOW"
        
        # Check if action would exceed position limits
        if action == "BUY":
            # Estimate new position size (assuming 10% of balance for the trade)
            estimated_trade_value = balance * 0.1
            estimated_new_value = current_position_value + estimated_trade_value
            estimated_new_pct = estimated_new_value / total_value if total_value > 0 else 0
            
            if estimated_new_pct > max_position_pct:
                warnings.append(f"BUY would exceed max position {max_position_pct:.1%} for {ticker}")
                risk_level = "HIGH"
            elif estimated_new_pct > max_position_pct * 0.8:
                warnings.append(f"BUY would approach max position limit for {ticker}")
                risk_level = "MEDIUM"
        
        # Check asset tier risk
        tier = settings.get_asset_tier(ticker)
        if tier >= 3 and action == "BUY":
            warnings.append(f"{ticker} is tier {tier} (higher risk) - exercise caution")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
        
        # Check RSI for overbought/oversold
        chart_signal = await self.redis.get_json(f"chart:signal:{ticker}")
        if chart_signal:
            rsi = chart_signal.get("data", {}).get("rsi")
            if rsi:
                if rsi > 80 and action == "BUY":
                    warnings.append(f"RSI {rsi:.1f} indicates overbought - risky to BUY")
                    risk_level = "HIGH"
                elif rsi < 20 and action == "SELL":
                    warnings.append(f"RSI {rsi:.1f} indicates oversold - risky to SELL")
                    risk_level = "HIGH"
        
        risk_score = self._map_risk_level_to_score(risk_level)
        stop_loss_upper, stop_loss_lower = self._calculate_stop_loss_window(risk_score, tier)
        return RiskMetrics(
            ticker=ticker,
            position_size=current_position_pct,
            max_position_size=max_position_pct,
            risk_level=risk_level,
            warnings=warnings,
            risk_score=risk_score,
            risk_metric=round(risk_score * 100, 2),
            stop_loss_upper=stop_loss_upper,
            stop_loss_lower=stop_loss_lower,
            timestamp=datetime.utcnow()
        )
    
    async def _calculate_asset_risk(self, ticker: str):
        """Calculate and cache risk metrics for an asset."""
        try:
            portfolio = await self.get_portfolio()
            if not portfolio:
                return
            
            risk_metrics = await self._calculate_position_risk(ticker, "HOLD", portfolio)
            
            await self.redis.set_json(
                f"risk:asset:{ticker}",
                risk_metrics.dict(),
                expire=300
            )
            
        except Exception as e:
            log.error(f"Error calculating risk for {ticker}: {e}")
    
    async def _get_peak_portfolio_value(self) -> float:
        """Get peak portfolio value for drawdown calculation."""
        peak = await self.redis.get("risk:peak_value")
        if peak:
            return float(peak)
        
        # Initialize with current value
        portfolio = await self.get_portfolio()
        if portfolio:
            current_value = portfolio.get("total_value_usdc", settings.initial_capital)
            await self.redis.set("risk:peak_value", str(current_value))
            return current_value
        
        return settings.initial_capital
    
    async def _send_risk_alert(self, risk_metrics: RiskMetrics):
        """Send risk alert to orchestrator."""
        alert = AgentMessage(
            message_type=MessageType.RISK_ALERT,
            sender=self.agent_type,
            recipient=AgentType.ORCHESTRATOR,
            payload={
                "risk_level": risk_metrics.risk_level,
                "warnings": risk_metrics.warnings,
                "current_drawdown": risk_metrics.current_drawdown,
                "daily_pnl": risk_metrics.daily_pnl
            }
        )
        
        await self.send_message(alert)
        log.warning(f"Risk alert sent: {risk_metrics.risk_level} - {risk_metrics.warnings}")
    
    def _build_risk_reasoning(self, risk_metrics: RiskMetrics) -> str:
        """Build human-readable risk reasoning."""
        reasoning = f"Risk assessment: {risk_metrics.risk_level} level. "
        
        if risk_metrics.warnings:
            reasoning += "Warnings: " + "; ".join(risk_metrics.warnings)
        else:
            reasoning += "No significant risk factors detected."
        
        if risk_metrics.ticker:
            reasoning += f" Current position: {risk_metrics.position_size:.1%} of portfolio."
        
        return reasoning

    def _map_risk_level_to_score(self, risk_level: str) -> float:
        mapping = {
            "LOW": 0.1,
            "MEDIUM": 0.35,
            "HIGH": 0.65,
            "CRITICAL": 0.9,
        }
        return mapping.get(risk_level.upper(), 0.35)

    def _calculate_stop_loss_window(self, risk_score: float, tier: int) -> Tuple[float, float]:
        tier_modifier = 0.02 if tier >= 3 else 0.0
        upside = 0.01 + (1 - risk_score) * 0.02 - tier_modifier
        downside = -0.02 - risk_score * 0.05 - tier_modifier
        return upside, downside
    
    async def validate_trade_decision(self, decision: TradeDecision) -> Dict:
        """Validate a trade decision against risk constraints."""
        portfolio = await self.get_portfolio()
        if not portfolio:
            return {
                "approved": False,
                "reason": "Portfolio data unavailable"
            }
        
        # Get risk metrics
        risk_metrics = await self._calculate_position_risk(
            decision.ticker,
            decision.action.value,
            portfolio
        )
        
        # Check if risk level is acceptable
        if risk_metrics.risk_level == "CRITICAL":
            return {
                "approved": False,
                "reason": f"CRITICAL risk level: {'; '.join(risk_metrics.warnings)}"
            }
        
        # Check confidence threshold
        if decision.confidence < settings.min_confidence:
            return {
                "approved": False,
                "reason": f"Confidence {decision.confidence:.2f} below minimum {settings.min_confidence}"
            }
        
        # Check portfolio risk
        portfolio_risk = await self.redis.get_json("risk:portfolio")
        if portfolio_risk and portfolio_risk.get("risk_level") == "CRITICAL":
            return {
                "approved": False,
                "reason": "Portfolio at CRITICAL risk level - trading paused"
            }
        
        # Approved with warnings
        return {
            "approved": True,
            "risk_level": risk_metrics.risk_level,
            "warnings": risk_metrics.warnings
        }

