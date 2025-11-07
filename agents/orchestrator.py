"""Orchestrator Agent - Coordinates all agents and makes final trading decisions."""
import json
import math
import uuid
from datetime import datetime, timedelta
from statistics import fmean
from typing import Any, Dict, List, Optional, Tuple

import httpx

from agents.base_agent import BaseAgent
from core.config import settings
from core.dex_simulator_client import DEXSimulatorError, dex_simulator_client
from core.exchange_interface import ExchangeType, OrderSide, OrderType
from core.logging import log
from core.models import (
    AgentMessage,
    AgentSignal,
    AgentType,
    HumanValidationRequest,
    HumanValidationResponse,
    MessageType,
    SignalType,
    TradeAction,
    TradeDecision,
    TradeExecution,
)
from core import asset_registry


class OrchestratorAgent(BaseAgent):
    """Agent responsible for coordinating all other agents and making final decisions."""
    
    def __init__(self, redis_client):
        super().__init__(AgentType.ORCHESTRATOR, redis_client)
        self.pending_signals: Dict[str, List[AgentSignal]] = {}
        self.pending_validations: Dict[str, HumanValidationRequest] = {}
        self.dex_client = dex_simulator_client
        self.wallet_plan_cache_key = "orchestrator:wallet_plan"
        self.wallet_plan_history_key = "orchestrator:wallet_plan_history"
        self.wallet_plan_ttl_seconds = 900
        
    async def initialize(self):
        """Initialize orchestrator."""
        # Connect to DEX simulator
        try:
            await self.dex_client.connect()
            log.info("Orchestrator Agent initialized with DEX simulator")
        except Exception as e:
            log.error(f"Failed to connect to DEX simulator: {e}")
            raise
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages from other agents."""
        try:
            if message.message_type == MessageType.SIGNAL_GENERATED:
                await self._handle_signal(message)
            elif message.message_type == MessageType.RISK_ALERT:
                await self._handle_risk_alert(message)
            elif message.message_type == MessageType.HUMAN_VALIDATION_RESPONSE:
                await self._handle_validation_response(message)
            
        except Exception as e:
            log.error(f"Orchestrator error processing message: {e}")
        
        return None
    
    async def run_cycle(self):
        """Run periodic decision-making cycle."""
        log.debug("Orchestrator running cycle...")
        
        try:
            await self._evaluate_wallet_balance_task()
            # Process accumulated signals for each ticker
            for ticker in asset_registry.get_assets():
                await self._make_trading_decision(ticker)
            
            # Clean up old pending validations
            await self._cleanup_pending_validations()
            
        except Exception as e:
            log.error(f"Orchestrator cycle error: {e}")
    
    def get_cycle_interval(self) -> int:
        """Run every 5 minutes."""
        return 300
    
    async def _handle_signal(self, message: AgentMessage):
        """Handle incoming signal from an agent."""
        try:
            signal_data = message.payload
            ticker = signal_data.get("ticker")
            
            if not ticker:
                return
            
            # Store signal for aggregation
            if ticker not in self.pending_signals:
                self.pending_signals[ticker] = []
            
            # Convert to AgentSignal
            signal = AgentSignal(**signal_data)
            self.pending_signals[ticker].append(signal)
            
            # Keep only recent signals (last 10)
            self.pending_signals[ticker] = self.pending_signals[ticker][-10:]
            
            log.debug(f"Received signal from {signal.agent_type.value} for {ticker}: {signal.action}")
            
        except Exception as e:
            log.error(f"Error handling signal: {e}")

    async def _evaluate_wallet_balance_task(self):
        """Generate a cross-agent wallet balancing plan with structured logging."""
        if not await self._should_refresh_wallet_plan():
            return

        portfolio = await self._get_portfolio_from_dex()
        if not portfolio:
            return

        contexts: List[Dict[str, Any]] = []
        for ticker in asset_registry.get_assets():
            context = await self._collect_asset_context(ticker, portfolio)
            if context:
                contexts.append(context)

        if not contexts:
            return

        plan_payload = await self._normalize_allocations(contexts, portfolio)
        await self.redis.set_json(
            self.wallet_plan_cache_key,
            plan_payload,
            expire=self.wallet_plan_ttl_seconds,
        )
        await self.redis.lpush(
            self.wallet_plan_history_key,
            json.dumps(plan_payload),
        )
        await self.redis.ltrim(self.wallet_plan_history_key, 0, 49)
        log.bind(PORTFOLIO_PLAN=True).info("Wallet balance plan generated: {}", plan_payload)

    async def _should_refresh_wallet_plan(self) -> bool:
        try:
            plan = await self.redis.get_json(self.wallet_plan_cache_key)
        except Exception:
            return True
        if not plan:
            return True
        generated_at = self._parse_timestamp(plan.get("generated_at"))
        return (datetime.utcnow() - generated_at).total_seconds() > self.wallet_plan_ttl_seconds

    async def _collect_asset_context(self, ticker: str, portfolio: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            market_data = await self.get_market_data(ticker)
        except Exception as exc:
            log.warning("Failed to fetch market data for %s: %s", ticker, exc)
            market_data = None

        if not market_data:
            fallback_price = (
                portfolio.get("prices", {}).get(ticker)
                if isinstance(portfolio, dict)
                else None
            )
            if fallback_price is not None:
                market_data = {"price": fallback_price}
            else:
                return None

        price = float(market_data.get("price") or market_data.get("close") or 0.0)
        if price <= 0:
            return None

        dqn_signal = await self.redis.get_json(f"dqn:prediction:{ticker}") or {}
        chart_signal = await self.redis.get_json(f"chart:signal:{ticker}") or {}
        news_sentiment = await self.redis.get_json(f"memory:news:weighted:{ticker}") or {}
        market_sentiment = await self.redis.get_json("memory:news:weighted:__market__") or {}
        risk_metrics = await self.redis.get_json(f"risk:asset:{ticker}") or {}

        holdings = portfolio.get("holdings", {})
        position_quantity = float(holdings.get(ticker, 0.0))
        position_value = position_quantity * price
        total_value = float(portfolio.get("total_value_usdc", settings.initial_capital))

        copy_signals = [
            signal
            for signal in self.pending_signals.get(ticker, [])
            if signal.agent_type == AgentType.COPYTRADE
        ]
        copy_confidence = (
            fmean([max(0.0, signal.confidence) for signal in copy_signals])
            if copy_signals
            else 0.0
        )
        copy_wallets: List[str] = []
        for signal in copy_signals:
            wallet = signal.data.get("wallet_address") if isinstance(signal.data, dict) else None
            if wallet:
                copy_wallets.append(wallet[:8] + "..." + wallet[-4:] if len(wallet) > 12 else wallet)

        risk_level = (risk_metrics.get("risk_level") or "LOW").upper()
        risk_score = float(
            risk_metrics.get("risk_score", self._map_risk_level_to_score(risk_level))
        )

        dqn_action = self._normalize_action_value(dqn_signal.get("action")) if dqn_signal else "HOLD"
        dqn_confidence = float(dqn_signal.get("confidence", 0.0)) if dqn_signal else 0.0

        chart_action = self._normalize_action_value(chart_signal.get("action")) if chart_signal else "HOLD"
        chart_confidence = float(chart_signal.get("confidence", 0.0)) if chart_signal else 0.0

        news_score = float(news_sentiment.get("weighted_score", 0.0))
        news_weight = float(news_sentiment.get("total_weight", 0.0))
        global_sentiment = float(market_sentiment.get("weighted_score", 0.0))

        tier = settings.get_asset_tier(ticker)
        base_bias = 1.0 if tier == 1 else 0.85 if tier == 2 else 0.65
        buy_bias = 1.0 if dqn_action == "BUY" else 0.6 if dqn_action == "HOLD" else 0.2
        sentiment_component = ((news_score + global_sentiment + 2.0) / 4.0)
        composite_score = (
            dqn_confidence * buy_bias * 0.4
            + chart_confidence * 0.2
            + sentiment_component * 0.15
            + copy_confidence * 0.15
            + base_bias * 0.1
        )
        composite_score *= (1 - risk_score)
        composite_score = max(composite_score, 0.0)

        stop_loss = (
            float(risk_metrics.get("stop_loss_upper", 0.0)),
            float(risk_metrics.get("stop_loss_lower", 0.0)),
        )
        if stop_loss == (0.0, 0.0):
            stop_loss = self._calculate_stop_loss_window(risk_score, news_score, tier)
        gas_fee_estimate = self._estimate_gas_fee(max(position_value, total_value * 0.05))
        pnl_estimate = float(portfolio.get("total_pnl", 0.0)) * (
            position_value / total_value if total_value else 0.0
        )

        return {
            "ticker": ticker,
            "price": price,
            "position_quantity": position_quantity,
            "position_value": position_value,
            "dqn_action": dqn_action,
            "dqn_confidence": dqn_confidence,
            "chart_action": chart_action,
            "chart_confidence": chart_confidence,
            "sentiment_score": news_score,
            "global_sentiment": global_sentiment,
            "news_weight": news_weight,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "copy_wallets": copy_wallets,
            "copy_confidence": copy_confidence,
            "composite_score": composite_score,
            "stop_loss": stop_loss,
            "gas_fee_estimate": gas_fee_estimate,
            "pnl_estimate": pnl_estimate,
            "tier": tier,
        }

    async def _normalize_allocations(
        self, contexts: List[Dict[str, Any]], portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        total_value = float(portfolio.get("total_value_usdc", settings.initial_capital))

        raw_scores = [max(ctx["composite_score"], 0.0) for ctx in contexts]
        if not any(raw_scores):
            equal_weight = 1.0 / len(contexts)
            for ctx in contexts:
                ctx["target_allocation"] = equal_weight
        else:
            total_raw = sum(raw_scores)
            for ctx, score in zip(contexts, raw_scores):
                allocation = score / total_raw if total_raw else 0.0
                allocation = min(allocation, settings.get_max_position_for_asset(ctx["ticker"]))
                ctx["target_allocation"] = allocation

        allocation_sum = sum(ctx["target_allocation"] for ctx in contexts)
        if allocation_sum > 1.0:
            scale_factor = 1.0 / allocation_sum
            for ctx in contexts:
                ctx["target_allocation"] *= scale_factor
            allocation_sum = 1.0

        residual = max(0.0, 1.0 - allocation_sum)

        allocations: Dict[str, Any] = {}
        for ctx in contexts:
            allocations[ctx["ticker"]] = self._build_allocation_entry(
                ctx,
                ctx["target_allocation"],
                total_value,
            )

        if residual > 0.005:
            allocations["USDC"] = {
                "allocation": round(residual, 4),
                "allocation_pct": round(residual * 100, 2),
                "target_usdc": round(residual * total_value, 2),
                "confidence": 1.0,
                "risk_level": "LOW",
                "risk_score": 0.05,
                "gas_fee_estimate": 0.0,
                "stop_loss": [0.0, 0.0],
                "pnl_estimate": 0.0,
                "signals": {"note": "Reserve for stable holdings"},
            }

        context_snapshot: List[Dict[str, Any]] = []
        for ctx in contexts:
            snapshot = {
                key: (list(value) if key == "stop_loss" else value)
                for key, value in ctx.items()
                if key != "target_allocation"
            }
            context_snapshot.append(snapshot)

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "profile": settings.agent_schedule_profile,
            "total_value_usdc": total_value,
            "allocations": allocations,
            "context": context_snapshot,
        }

    def _build_allocation_entry(
        self, context: Dict[str, Any], allocation: float, total_value: float
    ) -> Dict[str, Any]:
        stop_loss_high, stop_loss_low = context["stop_loss"]
        return {
            "allocation": round(allocation, 4),
            "allocation_pct": round(allocation * 100, 2),
            "target_usdc": round(allocation * total_value, 2),
            "confidence": round(
                min(1.0, context["dqn_confidence"] * 0.6 + context["chart_confidence"] * 0.4),
                3,
            ),
            "composite_score": round(context["composite_score"], 3),
            "risk_level": context["risk_level"],
            "risk_score": round(context["risk_score"], 3),
            "gas_fee_estimate": round(context["gas_fee_estimate"], 4),
            "stop_loss": [round(stop_loss_high, 4), round(stop_loss_low, 4)],
            "pnl_estimate": round(context["pnl_estimate"], 2),
            "signals": {
                "dqn_action": context["dqn_action"],
                "dqn_confidence": round(context["dqn_confidence"], 3),
                "chart_action": context["chart_action"],
                "chart_confidence": round(context["chart_confidence"], 3),
                "sentiment": round(context["sentiment_score"], 3),
                "global_sentiment": round(context["global_sentiment"], 3),
                "news_weight": round(context["news_weight"], 3),
                "copy_confidence": round(context["copy_confidence"], 3),
                "copy_wallets": context["copy_wallets"],
            },
        }

    def _map_risk_level_to_score(self, level: str) -> float:
        mapping = {
            "LOW": 0.1,
            "MEDIUM": 0.25,
            "HIGH": 0.55,
            "CRITICAL": 0.85,
        }
        return mapping.get(level.upper(), 0.3)

    def _calculate_stop_loss_window(
        self, risk_score: float, sentiment: float, tier: int
    ) -> Tuple[float, float]:
        tier_modifier = 0.02 if tier >= 3 else 0.0
        upside = 0.01 + max(0.0, sentiment) * 0.02 - risk_score * 0.01 - tier_modifier
        downside = -0.02 - risk_score * 0.05 + min(0.0, sentiment) * 0.01 - tier_modifier
        return upside, downside

    def _estimate_gas_fee(self, notionally_traded: float) -> float:
        base_fee = notionally_traded * settings.trading_fee
        return max(base_fee, 0.25)

    def _normalize_action_value(self, action: Optional[Any]) -> str:
        if isinstance(action, TradeAction):
            return action.value
        if isinstance(action, str):
            return action.upper()
        if isinstance(action, dict) and "value" in action:
            return str(action["value"]).upper()
        return "HOLD"

    def _parse_timestamp(self, value: Optional[str]) -> datetime:
        if not value:
            return datetime.utcnow() - timedelta(seconds=self.wallet_plan_ttl_seconds * 2)
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.utcnow() - timedelta(seconds=self.wallet_plan_ttl_seconds * 2)
    
    async def _handle_risk_alert(self, message: AgentMessage):
        """Handle risk alert from risk agent."""
        try:
            payload = message.payload
            risk_level = payload.get("risk_level")
            warnings = payload.get("warnings", [])
            
            log.warning(f"Risk alert received: {risk_level} - {warnings}")
            
            # If critical, pause trading
            if risk_level == "CRITICAL":
                await self.redis.set("orchestrator:trading_paused", "1", expire=3600)
                log.error("CRITICAL risk level - Trading paused for 1 hour")
            
        except Exception as e:
            log.error(f"Error handling risk alert: {e}")
    
    async def _handle_validation_response(self, message: AgentMessage):
        """Handle human validation response."""
        try:
            response_data = message.payload
            response = HumanValidationResponse(**response_data)
            
            request_id = response.request_id
            
            if request_id not in self.pending_validations:
                log.warning(f"Validation response for unknown request: {request_id}")
                return
            
            request = self.pending_validations[request_id]
            
            if response.approved:
                # Execute the trade
                await self._execute_trade(request.decision)
                log.info(f"Trade approved by human: {request.decision.ticker} {request.decision.action.value}")
            else:
                log.info(f"Trade rejected by human: {request.decision.ticker} {request.decision.action.value}")
                if response.feedback:
                    log.info(f"Feedback: {response.feedback}")
            
            # Remove from pending
            del self.pending_validations[request_id]
            
        except Exception as e:
            log.error(f"Error handling validation response: {e}")
    
    async def _make_trading_decision(self, ticker: str):
        """Make trading decision for a ticker based on all signals."""
        try:
            # Check if trading is paused
            if await self.redis.exists("orchestrator:trading_paused"):
                log.debug("Trading is paused due to risk alert")
                return
            
            # Get signals for this ticker
            signals = self.pending_signals.get(ticker, [])
            
            if not signals:
                return
            
            # Aggregate signals
            decision = await self._aggregate_signals(ticker, signals)
            
            if not decision:
                return
            
            # Validate with risk agent
            risk_validation = await self._validate_with_risk_agent(decision)
            
            if not risk_validation.get("approved", False):
                log.info(f"Trade decision rejected by risk agent: {risk_validation.get('reason')}")
                return
            
            decision.risk_approved = True
            
            # Check if human validation is required
            if await self._requires_human_validation(decision):
                await self._request_human_validation(decision)
                return
            
            # Execute trade
            await self._execute_trade(decision)
            
            # Clear processed signals
            self.pending_signals[ticker] = []
            
        except Exception as e:
            log.error(f"Error making trading decision for {ticker}: {e}")
    
    async def _aggregate_signals(self, ticker: str, signals: List[AgentSignal]) -> Optional[TradeDecision]:
        """Aggregate signals from multiple agents into a single decision."""
        try:
            # Weight signals by agent type and confidence
            agent_weights = {
                AgentType.DQN: 0.35,        # DQN predictions are primary
                AgentType.CHART: 0.25,      # Technical analysis is important
                AgentType.RISK: 0.20,       # Risk assessment is critical
                AgentType.NEWS: 0.10,       # News sentiment is supplementary
                AgentType.COPYTRADE: 0.10,  # Copy trading is supplementary
                AgentType.MEMORY: 0.0       # Memory provides context, not direct signals
            }
            
            # Calculate weighted action scores
            action_scores = {TradeAction.BUY: 0.0, TradeAction.SELL: 0.0, TradeAction.HOLD: 0.0}
            total_weight = 0.0
            
            for signal in signals:
                if signal.action is None:
                    continue
                
                agent_weight = agent_weights.get(signal.agent_type, 0.1)
                signal_weight = agent_weight * signal.confidence
                
                action_scores[signal.action] += signal_weight
                total_weight += signal_weight
            
            if total_weight == 0:
                return None
            
            # Normalize scores
            for action in action_scores:
                action_scores[action] /= total_weight
            
            # Determine final action
            final_action = max(action_scores, key=action_scores.get)
            final_confidence = action_scores[final_action]
            
            # Don't trade if confidence is too low or action is HOLD
            if final_confidence < settings.min_confidence or final_action == TradeAction.HOLD:
                return None
            
            # Calculate quantity based on portfolio
            portfolio = await self._get_portfolio_from_dex()
            if not portfolio:
                return None
            
            balance = portfolio.get("balance_usdc", 0)
            holdings = portfolio.get("holdings", {})
            
            # Get market data
            market_data = await self.get_market_data(ticker)
            if not market_data:
                return None
            
            current_price = market_data.get("price", 0)
            
            # Calculate quantity
            if final_action == TradeAction.BUY:
                # Use 10% of balance for each trade
                trade_amount = balance * 0.1
                quantity = trade_amount / current_price if current_price > 0 else 0
            else:  # SELL
                # Sell 50% of holdings
                current_holdings = holdings.get(ticker, 0)
                quantity = current_holdings * 0.5
            
            if quantity <= 0:
                return None
            
            # Build reasoning
            reasoning = self._build_decision_reasoning(signals, action_scores)
            
            decision = TradeDecision(
                ticker=ticker,
                action=final_action,
                quantity=quantity,
                expected_price=current_price,
                confidence=final_confidence,
                reasoning=reasoning,
                contributing_signals=signals,
                risk_approved=False
            )
            
            return decision
            
        except Exception as e:
            log.error(f"Error aggregating signals: {e}")
            return None
    
    def _build_decision_reasoning(self, signals: List[AgentSignal], action_scores: Dict) -> str:
        """Build human-readable reasoning for the decision."""
        reasoning_parts = []
        
        # Summarize signals by agent
        agent_summaries = {}
        for signal in signals:
            agent_name = signal.agent_type.value
            if agent_name not in agent_summaries:
                agent_summaries[agent_name] = []
            agent_summaries[agent_name].append(f"{signal.action.value if signal.action else 'N/A'} ({signal.confidence:.2f})")
        
        for agent, summaries in agent_summaries.items():
            reasoning_parts.append(f"{agent}: {', '.join(summaries)}")
        
        # Add action scores
        scores_str = ", ".join([f"{action.value}: {score:.2f}" for action, score in action_scores.items()])
        reasoning_parts.append(f"Aggregated scores: {scores_str}")
        
        return " | ".join(reasoning_parts)
    
    async def _validate_with_risk_agent(self, decision: TradeDecision) -> Dict:
        """Validate decision with risk agent."""
        try:
            # Get risk validation from Redis (risk agent caches its assessments)
            risk_data = await self.redis.get_json(f"risk:asset:{decision.ticker}")
            
            if not risk_data:
                return {"approved": True, "warnings": []}
            
            risk_level = risk_data.get("risk_level", "LOW")
            warnings = risk_data.get("warnings", [])
            
            if risk_level == "CRITICAL":
                return {
                    "approved": False,
                    "reason": f"CRITICAL risk level: {'; '.join(warnings)}"
                }
            
            return {
                "approved": True,
                "risk_level": risk_level,
                "warnings": warnings
            }
            
        except Exception as e:
            log.error(f"Error validating with risk agent: {e}")
            return {"approved": True}  # Fail open
    
    async def _requires_human_validation(self, decision: TradeDecision) -> bool:
        """Determine if a decision requires human validation."""
        # Require validation for:
        # 1. Large trades (>15% of portfolio)
        # 2. High-risk assets (tier 3-4)
        # 3. Low confidence (<0.75)
        
        portfolio = await self._get_portfolio_from_dex()
        if not portfolio:
            return True  # Require validation if portfolio unavailable
        
        total_value = portfolio.get("total_value_usdc", settings.initial_capital)
        market_data = await self.get_market_data(decision.ticker)
        current_price = market_data.get("price", 0) if market_data else 0
        
        trade_value = decision.quantity * current_price
        trade_pct = trade_value / total_value if total_value > 0 else 0
        
        # Large trade
        if trade_pct > 0.15:
            return True
        
        # High-risk asset
        tier = settings.get_asset_tier(decision.ticker)
        if tier >= 3:
            return True
        
        # Low confidence
        if decision.confidence < 0.75:
            return True
        
        return False
    
    async def _request_human_validation(self, decision: TradeDecision):
        """Request human validation for a trade decision."""
        try:
            request_id = str(uuid.uuid4())
            
            # Determine urgency
            urgency = "MEDIUM"
            if decision.confidence > 0.8:
                urgency = "LOW"
            elif decision.confidence < 0.7:
                urgency = "HIGH"
            
            request = HumanValidationRequest(
                request_id=request_id,
                decision=decision,
                urgency=urgency,
                reason="Trade requires human approval based on risk parameters"
            )
            
            # Store in pending validations
            self.pending_validations[request_id] = request
            
            # Publish validation request
            message = AgentMessage(
                message_type=MessageType.HUMAN_VALIDATION_REQUEST,
                sender=self.agent_type,
                payload=request.dict()
            )
            
            await self.send_message(message, channel="agent:broadcast")
            
            # Also store in Redis for API access
            await self.redis.set_json(
                f"validation:request:{request_id}",
                request.dict(),
                expire=600  # 10 minutes
            )
            
            log.info(f"Human validation requested for {decision.ticker} {decision.action.value} (ID: {request_id})")
            
        except Exception as e:
            log.error(f"Error requesting human validation: {e}")
    
    async def _execute_trade(self, decision: TradeDecision):
        """Execute a trade via the DEX simulator."""
        try:
            if not self.dex_client or not self.dex_client.client:
                log.error("DEX simulator client not initialized")
                return
            
            result = None
            
            if decision.action == TradeAction.BUY:
                # Buy: amount in USDC
                trade_amount_usdc = decision.quantity * decision.expected_price
                result = await self.dex_client.buy_asset(decision.ticker, trade_amount_usdc)
            else:  # SELL
                # Sell: amount in asset units
                result = await self.dex_client.sell_asset(decision.ticker, decision.quantity)
            
            # Check if trade was successful
            if not result or not result.get("success", False):
                log.error(f"Trade failed for {decision.ticker}: {result.get('error', 'Unknown error')}")
                return
            
            # Get current price after execution
            portfolio = await self.dex_client.get_portfolio_status()
            current_price = portfolio.get("prices", {}).get(decision.ticker, decision.expected_price)
            
            # Create execution record
            execution = TradeExecution(
                decision_id=str(uuid.uuid4()),
                ticker=decision.ticker,
                action=decision.action,
                quantity=decision.quantity,
                executed_price=current_price,
                total_cost=decision.quantity * current_price,
                fee=decision.quantity * current_price * 0.001,  # 0.1% fee
                success=True
            )
            
            # Update portfolio in Redis
            await self.redis.set_json("state:portfolio", portfolio)
            
            # Notify other agents
            await self._notify_trade_executed(execution, decision)
            
            # Log decision with special marker for trading_decisions.log
            log.bind(TRADE_DECISION=True).info(
                f"TRADE EXECUTED: {decision.ticker} {decision.action.value} "
                f"Qty: {decision.quantity:.4f} Price: ${execution.executed_price:.2f} "
                f"Confidence: {decision.confidence:.2f} | {decision.reasoning}"
            )
            
        except DEXSimulatorError as e:
            log.error(f"DEX simulator error: {e}")
        except Exception as e:
            log.error(f"Error executing trade: {e}")
    
    async def _notify_trade_executed(self, execution: TradeExecution, decision: TradeDecision):
        """Notify all agents that a trade was executed."""
        try:
            message = AgentMessage(
                message_type=MessageType.TRADE_EXECUTED,
                sender=self.agent_type,
                payload={
                    **execution.dict(),
                    "decision": decision.dict()
                }
            )
            
            await self.send_message(message, channel="agent:broadcast")
            
        except Exception as e:
            log.error(f"Error notifying trade execution: {e}")
    
    async def _cleanup_pending_validations(self):
        """Clean up expired pending validations."""
        try:
            current_time = datetime.utcnow()
            expired = []
            
            for request_id, request in self.pending_validations.items():
                age = (current_time - request.timestamp).total_seconds()
                if age > request.timeout_seconds:
                    expired.append(request_id)
            
            for request_id in expired:
                log.warning(f"Validation request {request_id} expired without response")
                del self.pending_validations[request_id]
                
        except Exception as e:
            log.error(f"Error cleaning up validations: {e}")
    
    async def _get_portfolio_from_dex(self) -> Optional[Dict]:
        """Get portfolio data from DEX simulator."""
        try:
            if not self.dex_client or not self.dex_client.client:
                log.warning("DEX client not available, using cached portfolio")
                return await self.get_portfolio()
            
            return await self.dex_client.get_portfolio_status()
        except Exception as e:
            log.error(f"Error getting portfolio from DEX: {e}")
            return await self.get_portfolio()

