"""
Orchestrator Agent - Coordinates all agents and makes final trading decisions.
"""
import uuid
import httpx
from typing import Optional, Dict, List
from datetime import datetime
from core.models import (
    AgentType, AgentMessage, MessageType, TradeDecision,
    TradeAction, AgentSignal, HumanValidationRequest,
    HumanValidationResponse, TradeExecution
)
from core.config import settings
from core.logging import log
from core.exchange_manager import ExchangeManager
from core.exchange_interface import ExchangeType, OrderSide, OrderType
from agents.base_agent import BaseAgent


class OrchestratorAgent(BaseAgent):
    """Agent responsible for coordinating all other agents and making final decisions."""
    
    def __init__(self, redis_client):
        super().__init__(AgentType.ORCHESTRATOR, redis_client)
        self.pending_signals: Dict[str, List[AgentSignal]] = {}
        self.pending_validations: Dict[str, HumanValidationRequest] = {}
        self.exchange_manager: Optional[ExchangeManager] = None
        
    async def initialize(self):
        """Initialize orchestrator."""
        # Initialize exchange manager
        self.exchange_manager = ExchangeManager()
        
        # Add exchanges based on configuration
        if settings.environment == "test" or settings.environment == "development":
            # Use mock exchanges for testing/development
            from tests.mocks.mock_dex_exchange import MockDEXExchange
            from tests.mocks.mock_mexc_exchange import MockMEXCExchange
            
            dex_config = {"mock_mode": True}
            dex_exchange = MockDEXExchange(dex_config)
            self.exchange_manager.add_exchange(dex_exchange, is_primary=True)
            
            mexc_config = {"mock_mode": True}
            mexc_exchange = MockMEXCExchange(mexc_config)
            self.exchange_manager.add_exchange(mexc_exchange, is_primary=False)
        else:
            # Use real exchanges for production
            from core.exchanges.dex_exchange import DEXExchange
            from core.exchanges.mexc_exchange import MEXCExchange
            
            # DEX configuration
            dex_config = {
                "network": "ethereum",
                "rpc_url": settings.eth_rpc_url,
                "private_key": settings.private_key,
                "mock_mode": False
            }
            dex_exchange = DEXExchange(dex_config)
            self.exchange_manager.add_exchange(dex_exchange, is_primary=True)
            
            # MEXC configuration
            mexc_config = {
                "api_key": settings.mexc_api_key,
                "secret_key": settings.mexc_secret_key,
                "mock_mode": False
            }
            mexc_exchange = MEXCExchange(mexc_config)
            self.exchange_manager.add_exchange(mexc_exchange, is_primary=False)
        
        # Connect to all exchanges
        await self.exchange_manager.connect_all()
        
        # Enable paper trading in development/test
        if settings.environment in ["test", "development"]:
            self.exchange_manager.set_paper_trading(True)
        
        log.info("Orchestrator Agent initialized with exchange manager")
    
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
            # Process accumulated signals for each ticker
            for ticker in settings.supported_assets:
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
            portfolio = await self.get_portfolio()
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
        
        portfolio = await self.get_portfolio()
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
        """Execute a trade via the exchange manager."""
        try:
            if not self.exchange_manager:
                log.error("Exchange manager not initialized")
                return
            
            # Convert action to exchange format
            order_side = OrderSide.BUY if decision.action == TradeAction.BUY else OrderSide.SELL
            order_type = OrderType.MARKET  # Use market orders for now
            
            # Determine exchange based on ticker or configuration
            exchange_type = None
            if decision.ticker in ["BTC", "ETH", "SOL"]:  # Major cryptos - use DEX
                exchange_type = ExchangeType.DEX
            else:  # Other assets - use MEXC
                exchange_type = ExchangeType.MEXC
            
            # Convert ticker format for exchange
            if exchange_type == ExchangeType.DEX:
                symbol = f"{decision.ticker}USDC"
            else:  # MEXC
                symbol = f"{decision.ticker}USDT"
            
            # Execute trade
            order = await self.exchange_manager.place_order(
                symbol=symbol,
                side=order_side,
                order_type=order_type,
                amount=decision.quantity,
                price=decision.expected_price,
                exchange_type=exchange_type
            )
            
            # Create execution record
            execution = TradeExecution(
                decision_id=str(uuid.uuid4()),
                ticker=decision.ticker,
                action=decision.action,
                quantity=decision.quantity,
                executed_price=order.average_price or decision.expected_price,
                total_cost=decision.quantity * (order.average_price or decision.expected_price),
                fee=order.fee,
                success=order.is_filled
            )
            
            # Notify other agents
            await self._notify_trade_executed(execution, decision)
            
            # Log decision with special marker for trading_decisions.log
            log.bind(TRADE_DECISION=True).info(
                f"TRADE EXECUTED: {decision.ticker} {decision.action.value} "
                f"Qty: {decision.quantity:.4f} Price: ${execution.executed_price:.2f} "
                f"Confidence: {decision.confidence:.2f} | {decision.reasoning}"
            )
            
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

