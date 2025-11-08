"""
Chart Analysis Agent - Technical analysis using indicators like RSI, MACD, Bollinger Bands.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Any, List
from datetime import datetime
from core.models import (
    AgentType, AgentMessage, MessageType, TechnicalSignal,
    TradeAction, AgentSignal, SignalType
)
from core.config import settings
from core.logging import log
from agents.base_agent import BaseAgent
from core import asset_registry
from core.forecasting_client import ForecastingClient, ForecastingAPIError


class ChartAgent(BaseAgent):
    """Agent responsible for technical analysis of price charts."""
    
    def __init__(self, redis_client):
        super().__init__(AgentType.CHART, redis_client)
        self.forecasting_client: Optional[ForecastingClient] = None
        
    async def initialize(self):
        """Initialize forecasting API client for fetching market data."""
        config = {
            "base_url": settings.mcp_api_url,
            "api_key": settings.mcp_api_key,
            "mock_mode": settings.environment == "test" or settings.use_mock_services,
        }
        self.forecasting_client = ForecastingClient(config)
        await self.forecasting_client.connect()
        log.info("Chart Agent initialized with forecasting client")
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages."""
        if message.message_type == MessageType.MARKET_DATA_UPDATE:
            ticker = message.payload.get("ticker")
            if ticker:
                await self._analyze_ticker(ticker)
        
        return None
    
    async def stop(self):
        if self.forecasting_client:
            await self.forecasting_client.disconnect()
            self.forecasting_client = None
        await super().stop()
    
    async def run_cycle(self):
        """Run periodic technical analysis on all supported assets."""
        log.debug("Chart Agent running cycle...")
        
        for ticker in asset_registry.get_assets():
            try:
                await self._analyze_ticker(ticker)
            except Exception as e:
                log.error(f"Chart Agent error analyzing {ticker}: {e}")
    
    def get_cycle_interval(self) -> int:
        return settings.get_agent_cycle_seconds(self.agent_type)
    
    async def _analyze_ticker(self, ticker: str):
        """Perform technical analysis on a ticker."""
        try:
            interval = self._resolve_interval(ticker)
            df = await self._fetch_price_data(ticker, interval)
            
            if df is None or len(df) < 50:
                log.warning(f"Insufficient data for technical analysis of {ticker}")
                return
            
            # Calculate indicators
            indicators = self._calculate_indicators(df)
            
            # Generate signal
            symbol = asset_registry.get_symbol(ticker)
            distribution, metrics, forecast_price = await self._fetch_forecasting_context(symbol, interval)
            signal = self._generate_signal(ticker, indicators, distribution, forecast_price, metrics, interval)
            
            # Send signal to orchestrator
            agent_signal = AgentSignal(
                agent_type=self.agent_type,
                signal_type=SignalType.TECHNICAL_ANALYSIS,
                ticker=ticker,
                action=signal.recommendation,
                confidence=signal.strength,
                data={
                    "rsi": signal.rsi,
                    "macd": signal.macd,
                    "macd_signal": signal.macd_signal,
                    "bollinger_upper": signal.bollinger_upper,
                    "bollinger_lower": signal.bollinger_lower,
                    "volume_sma": signal.volume_sma,
                    "current_price": float(df['close'].iloc[-1]),
                    "forecast_price": forecast_price,
                    "forecast_action_distribution": distribution,
                    "forecast_metrics": metrics,
                    "interval": interval,
                },
                reasoning=self._build_reasoning(signal, indicators)
            )
            
            await self.send_signal(agent_signal.dict())
            
            # Cache signal
            cached_signal = agent_signal.dict()
            cached_signal["generated_at"] = datetime.utcnow().isoformat()
            await self.redis.set_json(
                f"chart:signal:{ticker}",
                cached_signal,
                expire=300
            )
            
        except Exception as e:
            log.error(f"Error in technical analysis for {ticker}: {e}")
    
    async def _fetch_price_data(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch historical price data for a ticker."""
        if not self.forecasting_client:
            return None
        symbol = asset_registry.get_symbol(ticker)
        try:
            candles = await self.forecasting_client.get_ohlc(symbol, interval, limit=200)
        except ForecastingAPIError as exc:
            log.warning("Forecasting API error fetching OHLC for %s/%s: %s", symbol, interval, exc)
            return None
        except Exception as exc:
            log.error("Unexpected error fetching OHLC for %s/%s: %s", symbol, interval, exc)
            return None

        if not candles:
            return None

        df = pd.DataFrame(candles)
        if df.empty:
            return None

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"])
        df = df.sort_values("timestamp")

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"])
        return df
    
    def _resolve_interval(self, ticker: str) -> str:
        intervals = asset_registry.get_intervals(ticker)
        preferred = [
            settings.decision_interval,
            settings.observation_interval,
            "hours",
            "days",
            "minutes",
        ]
        for candidate in preferred:
            if candidate and candidate in intervals:
                return candidate
        return intervals[0] if intervals else "hours"
    
    async def _fetch_forecasting_context(
        self,
        symbol: str,
        interval: str,
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, Any]], Optional[float]]:
        if not self.forecasting_client:
            return None, None, None

        distribution: Optional[Dict[str, float]] = None
        metrics: Optional[Dict[str, Any]] = None
        forecast_price: Optional[float] = None

        try:
            action_payload = await self.forecasting_client.get_action_recommendation(symbol, interval)
            distribution = self._normalise_distribution(action_payload)
        except ForecastingAPIError as exc:
            log.debug("Unable to fetch action distribution for %s/%s: %s", symbol, interval, exc)
        except Exception as exc:
            log.error("Unexpected error fetching action distribution for %s/%s: %s", symbol, interval, exc)

        try:
            metrics = await self.forecasting_client.get_model_metrics(symbol, interval)
        except ForecastingAPIError as exc:
            log.debug("Unable to fetch model metrics for %s/%s: %s", symbol, interval, exc)
        except Exception as exc:
            log.error("Unexpected error fetching metrics for %s/%s: %s", symbol, interval, exc)

        try:
            forecast_payload = await self.forecasting_client.get_stock_forecast(symbol, interval)
            forecast_price = forecast_payload.get("forecast_price")
            if forecast_price is None:
                forecast_data = forecast_payload.get("forecast_data")
                if forecast_data:
                    forecast_price = forecast_data[0].get("price")
        except ForecastingAPIError as exc:
            log.debug("Unable to fetch stock forecast for %s/%s: %s", symbol, interval, exc)
        except Exception as exc:
            log.error("Unexpected error fetching stock forecast for %s/%s: %s", symbol, interval, exc)

        return distribution, metrics, forecast_price
    
    def _normalise_distribution(self, action_payload: Dict[str, Any]) -> Dict[str, float]:
        q_values = action_payload.get("q_values")
        if isinstance(q_values, (list, tuple)) and len(q_values) == 3:
            total = float(sum(q_values)) or 1.0
            return {
                "sell": float(q_values[0]) / total,
                "hold": float(q_values[1]) / total,
                "buy": float(q_values[2]) / total,
            }

        distribution = {
            "sell": float(action_payload.get("sell_probability", 0.0)),
            "hold": float(action_payload.get("hold_probability", 0.0)),
            "buy": float(action_payload.get("buy_probability", 0.0)),
        }
        total = sum(distribution.values())
        if total <= 0:
            action = action_payload.get("action")
            distribution = {"sell": 0.0, "hold": 0.0, "buy": 0.0}
            if action == 0:
                distribution["sell"] = 1.0
            elif action == 1:
                distribution["hold"] = 1.0
            elif action == 2:
                distribution["buy"] = 1.0
        else:
            distribution = {k: v / total for k, v in distribution.items()}
        return distribution
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators."""
        indicators = {}
        
        # RSI (Relative Strength Index)
        indicators['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD (Moving Average Convergence Divergence)
        macd, signal, histogram = self._calculate_macd(df['close'])
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_histogram'] = histogram
        
        # Bollinger Bands
        upper, middle, lower = self._calculate_bollinger_bands(df['close'])
        indicators['bollinger_upper'] = upper
        indicators['bollinger_middle'] = middle
        indicators['bollinger_lower'] = lower
        
        # Volume SMA
        indicators['volume_sma'] = df['volume'].rolling(window=20).mean().iloc[-1]
        indicators['current_volume'] = df['volume'].iloc[-1]
        
        # Price SMAs
        indicators['sma_20'] = df['close'].rolling(window=20).mean().iloc[-1]
        indicators['sma_50'] = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
        
        indicators['current_price'] = df['close'].iloc[-1]
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1])
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return float(macd.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])
    
    def _generate_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        distribution: Optional[Dict[str, float]],
        forecast_price: Optional[float],
        metrics: Optional[Dict[str, Any]],
        interval: str,
    ) -> TechnicalSignal:
        """Generate trading signal based on technical indicators and forecasting distribution."""
        signals: List[TradeAction] = []
        weights: List[float] = []

        current_price = indicators["current_price"]

        # RSI contribution
        rsi = indicators["rsi"]
        if rsi < 30:
            signals.append(TradeAction.BUY)
            weights.append(0.3)
        elif rsi > 70:
            signals.append(TradeAction.SELL)
            weights.append(0.3)
        else:
            signals.append(TradeAction.HOLD)
            weights.append(0.1)

        # MACD contribution
        macd = indicators["macd"]
        macd_signal = indicators["macd_signal"]
        if macd > macd_signal:
            signals.append(TradeAction.BUY)
            weights.append(0.25)
        elif macd < macd_signal:
            signals.append(TradeAction.SELL)
            weights.append(0.25)
        else:
            signals.append(TradeAction.HOLD)
            weights.append(0.1)

        # Bollinger bands contribution
        bb_upper = indicators["bollinger_upper"]
        bb_lower = indicators["bollinger_lower"]
        if current_price < bb_lower:
            signals.append(TradeAction.BUY)
            weights.append(0.25)
        elif current_price > bb_upper:
            signals.append(TradeAction.SELL)
            weights.append(0.25)
        else:
            signals.append(TradeAction.HOLD)
            weights.append(0.1)

        # SMA crossover contribution
        sma_20 = indicators["sma_20"]
        sma_50 = indicators.get("sma_50")
        if sma_50:
            if sma_20 > sma_50:
                signals.append(TradeAction.BUY)
                weights.append(0.2)
            elif sma_20 < sma_50:
                signals.append(TradeAction.SELL)
                weights.append(0.2)
            else:
                signals.append(TradeAction.HOLD)
                weights.append(0.1)

        # Forecasting distribution contribution
        if distribution:
            buy_prob = float(distribution.get("buy", 0.0))
            sell_prob = float(distribution.get("sell", 0.0))
            hold_prob = float(distribution.get("hold", 0.0))
            bias = buy_prob - sell_prob
            if abs(bias) > 0.05:
                signals.append(TradeAction.BUY if bias > 0 else TradeAction.SELL)
                weights.append(abs(bias))
            else:
                signals.append(TradeAction.HOLD)
                weights.append(max(hold_prob, 0.1))

        action_scores = {TradeAction.BUY: 0.0, TradeAction.SELL: 0.0, TradeAction.HOLD: 0.0}
        for action, weight in zip(signals, weights):
            action_scores[action] += weight

        recommendation = max(action_scores, key=action_scores.get)
        total_weight = sum(weights) or 1.0
        strength = action_scores[recommendation] / total_weight

        return TechnicalSignal(
            ticker=ticker,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            bollinger_upper=bb_upper,
            bollinger_lower=bb_lower,
            volume_sma=indicators["volume_sma"],
            recommendation=recommendation,
            strength=strength,
            timestamp=datetime.utcnow(),
        )
    
    def _build_reasoning(self, signal: TechnicalSignal, indicators: Dict) -> str:
        """Build human-readable reasoning for the signal."""
        reasons = []
        
        # RSI reasoning
        if signal.rsi < 30:
            reasons.append(f"RSI at {signal.rsi:.1f} indicates oversold conditions")
        elif signal.rsi > 70:
            reasons.append(f"RSI at {signal.rsi:.1f} indicates overbought conditions")
        
        # MACD reasoning
        if signal.macd > signal.macd_signal:
            reasons.append("MACD shows bullish momentum")
        elif signal.macd < signal.macd_signal:
            reasons.append("MACD shows bearish momentum")
        
        # Bollinger Bands reasoning
        current_price = indicators['current_price']
        if current_price < signal.bollinger_lower:
            reasons.append("Price below lower Bollinger Band suggests potential bounce")
        elif current_price > signal.bollinger_upper:
            reasons.append("Price above upper Bollinger Band suggests potential pullback")
        
        reasoning = f"Technical analysis: {', '.join(reasons)}. "
        reasoning += f"Overall signal: {signal.recommendation.value} with {signal.strength:.0%} strength"
        
        return reasoning

