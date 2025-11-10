"""
DQN Decision Agent - Interfaces with Forecasting API for DQN predictions and forecasts.
"""
import httpx
from typing import Optional, Dict, List
from datetime import datetime
from core.models import (
    AgentType, AgentMessage, MessageType, DQNPrediction, 
    TradeAction, AgentSignal, SignalType
)
from core.config import settings
from core.logging import log
from core.forecasting_client import ForecastingAPIError, AssetNotEnabledError
from agents.base_agent import BaseAgent
from core import asset_registry
from core.camel_runtime.registries import toolkit_registry


class DQNAgent(BaseAgent):
    """Agent responsible for getting DQN predictions from Forecasting API."""
    
    def __init__(self, redis_client):
        super().__init__(AgentType.DQN, redis_client)
        self._supported_assets: List[str] = list(settings.supported_assets)
        
    async def initialize(self):
        """Initialize Forecasting API client."""
        await toolkit_registry.ensure_clients()
        log.info("DQN Agent initialized with CAMEL toolkit registry")
        await self._refresh_supported_assets()
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages."""
        if message.message_type == MessageType.MARKET_DATA_UPDATE:
            # Market data updated, trigger analysis
            ticker = message.payload.get("ticker")
            if ticker:
                await self._analyze_ticker(ticker)
        
        return None
    
    async def run_cycle(self):
        """Run periodic analysis on all supported assets."""
        log.debug("DQN Agent running cycle...")
        
        for ticker in asset_registry.get_assets():
            try:
                await self._analyze_ticker(ticker)
            except Exception as e:
                log.error(f"DQN Agent error analyzing {ticker}: {e}")
    
    def get_cycle_interval(self) -> int:
        return settings.get_agent_cycle_seconds(self.agent_type)
    
    async def _analyze_ticker(self, ticker: str):
        """Analyze a ticker using MCP API predictions."""
        try:
            if ticker not in asset_registry.get_assets():
                log.debug("Skipping unsupported ticker %s", ticker)
                return

            # Get predictions for multiple intervals
            predictions = await self._get_multi_interval_predictions(ticker)
            
            if not predictions:
                log.warning(f"No predictions available for {ticker}")
                return
            
            # Aggregate predictions
            aggregated = self._aggregate_predictions(predictions)
            
            # Create and send signal
            signal = AgentSignal(
                agent_type=self.agent_type,
                signal_type=SignalType.DQN_PREDICTION,
                ticker=ticker,
                action=aggregated["action"],
                confidence=aggregated["confidence"],
                data={
                    "predictions": [p.dict() for p in predictions],
                    "forecast_price": aggregated.get("forecast_price"),
                    "forecast_horizon": aggregated.get("forecast_horizon"),
                    "intervals_analyzed": aggregated["intervals_analyzed"]
                },
                reasoning=aggregated["reasoning"]
            )
            log.bind(agent="DQN", instance=settings.agent_instance_id).info(
                "DQN aggregated decision for %s: %s (confidence=%.2f)",
                ticker,
                aggregated["action"].value if isinstance(aggregated["action"], TradeAction) else aggregated["action"],
                aggregated["confidence"],
            )
            
            await self.send_signal(signal.dict())
            
            # Cache prediction
            cached_signal = signal.dict()
            cached_signal["generated_at"] = datetime.utcnow().isoformat()
            await self.redis.set_json(
                f"dqn:prediction:{ticker}",
                cached_signal,
                expire=300  # 5 minutes
            )
            
        except Exception as e:
            log.error(f"Error analyzing {ticker}: {e}")
    
    async def _get_multi_interval_predictions(self, ticker: str) -> List[DQNPrediction]:
        """Get predictions for multiple time intervals."""
        predictions = []
        
        # Prefer the intervals advertised by the Forecasting API, fallback to hours/days
        intervals = [interval for interval in asset_registry.get_intervals(ticker) if interval in {"hours", "days"}]
        if not intervals:
            intervals = ["hours", "days"]
        
        for interval in intervals:
            try:
                prediction = await self._get_action_recommendation(ticker, interval)
                if prediction:
                    predictions.append(prediction)
            except Exception as e:
                log.error(f"Error getting {interval} prediction for {ticker}: {e}")
        
        return predictions
    
    async def _get_action_recommendation(self, ticker: str, interval: str) -> Optional[DQNPrediction]:
        """Get action recommendation from Forecasting API."""
        try:
            api_ticker = asset_registry.get_symbol(ticker)
            data = await toolkit_registry.get_action_recommendation(api_ticker, interval)
            data = data.get("action", {})
            
            # Parse action (0=SELL, 1=HOLD, 2=BUY)
            action_map = {0: TradeAction.SELL, 1: TradeAction.HOLD, 2: TradeAction.BUY}
            action_value = data.get("action")
            
            if action_value is None:
                return None
            
            action = action_map.get(action_value, TradeAction.HOLD)
            confidence = data.get("action_confidence", 0.5)
            
            # Get forecast if available
            forecast_price = data.get("forecast_price")
            
            return DQNPrediction(
                ticker=ticker,
                action=action,
                confidence=confidence,
                forecast_price=forecast_price,
                forecast_horizon=f"T+14{interval}",
                timestamp=datetime.utcnow()
            )
            
        except AssetNotEnabledError as e:
            disabled_ticker = e.ticker.split("-")[0] if getattr(e, "ticker", None) else ticker
            disabled_ticker = disabled_ticker.upper()
            log.warning(
                "Forecasting API reports %s is not enabled; disabling asset until registry refresh",
                disabled_ticker,
            )
            await asset_registry.disable_asset(disabled_ticker)
            if disabled_ticker != ticker.upper():
                await asset_registry.disable_asset(ticker)
            return None
        except ForecastingAPIError as e:
            log.error(f"Forecasting API error getting action for {ticker}/{interval}: {e}")
            return None
        except Exception as e:
            log.error(f"Error getting action for {ticker}/{interval}: {e}")
            return None
    
    async def get_stock_forecast(self, ticker: str, interval: str) -> Optional[Dict]:
        """Get detailed stock forecast from Forecasting API."""
        try:
            # Convert ticker format (BTC -> BTC-USD)
            api_ticker = asset_registry.get_symbol(ticker)
            
            data = await toolkit_registry.get_stock_forecast(api_ticker, interval)
            return data.get("forecast")
        except Exception as e:
            log.error(f"Error getting forecast for {ticker}/{interval}: {e}")
            return None
    
    def _aggregate_predictions(self, predictions: List[DQNPrediction]) -> Dict:
        """Aggregate multiple interval predictions into a single decision."""
        if not predictions:
            return {
                "action": TradeAction.HOLD,
                "confidence": 0.0,
                "reasoning": "No predictions available",
                "intervals_analyzed": []
            }
        
        # Weight predictions by confidence
        action_scores = {TradeAction.BUY: 0.0, TradeAction.SELL: 0.0, TradeAction.HOLD: 0.0}
        total_confidence = 0.0
        
        for pred in predictions:
            action_scores[pred.action] += pred.confidence
            total_confidence += pred.confidence
        
        # Normalize scores
        if total_confidence > 0:
            for action in action_scores:
                action_scores[action] /= total_confidence
        
        # Determine final action
        final_action = max(action_scores, key=action_scores.get)
        final_confidence = action_scores[final_action]
        
        # Get forecast price from days prediction if available
        forecast_price = None
        forecast_horizon = None
        for pred in predictions:
            if pred.forecast_price and "days" in pred.forecast_horizon:
                forecast_price = pred.forecast_price
                forecast_horizon = pred.forecast_horizon
                break
        
        # Build reasoning
        intervals = [p.forecast_horizon for p in predictions]
        action_summary = ", ".join([f"{p.forecast_horizon}: {p.action.value} ({p.confidence:.2f})" for p in predictions])
        
        reasoning = f"Multi-interval DQN analysis: {action_summary}. "
        reasoning += f"Aggregated decision: {final_action.value} with confidence {final_confidence:.2f}"
        
        if forecast_price:
            reasoning += f". Forecast price: ${forecast_price:.2f} at {forecast_horizon}"
        
        return {
            "action": final_action,
            "confidence": final_confidence,
            "forecast_price": forecast_price,
            "forecast_horizon": forecast_horizon,
            "reasoning": reasoning,
            "intervals_analyzed": intervals
        }
    
    async def get_available_tickers(self) -> List[str]:
        """Get list of available tickers from Forecasting API."""
        try:
            payload = await toolkit_registry.list_supported_assets()
            return [symbol.replace("-USD", "") for symbol in payload.get("assets", [])]
        except Exception as e:
            log.error(f"Error getting available tickers: {e}")
            return []

    async def _refresh_supported_assets(self) -> None:
        """Fetch available assets from the Forecasting API and update the shared registry."""
        try:
            available_payload = await toolkit_registry.list_supported_assets()
            available = available_payload.get("assets", [])
            enabled = available
            await asset_registry.update_assets(available, enabled)
            self._supported_assets = asset_registry.get_assets()
            log.info("DQN Agent using %d dynamically enabled assets", len(self._supported_assets))
        except Exception as exc:
            log.warning("Unable to refresh asset catalogue, falling back to static list: %s", exc)
            await asset_registry.use_fallback_assets()
            self._supported_assets = asset_registry.get_assets()

