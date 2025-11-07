"""
Chart Analysis Agent - Technical analysis using indicators like RSI, MACD, Bollinger Bands.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime, timedelta
import httpx
from core.models import (
    AgentType, AgentMessage, MessageType, TechnicalSignal,
    TradeAction, AgentSignal, SignalType
)
from core.config import settings
from core.logging import log
from agents.base_agent import BaseAgent
from core import asset_registry


class ChartAgent(BaseAgent):
    """Agent responsible for technical analysis of price charts."""
    
    def __init__(self, redis_client):
        super().__init__(AgentType.CHART, redis_client)
        self.client: Optional[httpx.AsyncClient] = None
        
    async def initialize(self):
        """Initialize HTTP client for fetching market data."""
        self.client = httpx.AsyncClient(timeout=30.0)
        log.info("Chart Agent initialized")
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages."""
        if message.message_type == MessageType.MARKET_DATA_UPDATE:
            ticker = message.payload.get("ticker")
            if ticker:
                await self._analyze_ticker(ticker)
        
        return None
    
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
            # Get historical price data
            df = await self._fetch_price_data(ticker)
            
            if df is None or len(df) < 50:
                log.warning(f"Insufficient data for technical analysis of {ticker}")
                return
            
            # Calculate indicators
            indicators = self._calculate_indicators(df)
            
            # Generate signal
            signal = self._generate_signal(ticker, indicators)
            
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
                    "current_price": float(df['close'].iloc[-1])
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
    
    async def _fetch_price_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch historical price data for a ticker."""
        try:
            # Try to get from Kraken API (used by DEX simulator)
            url = f"https://api.kraken.com/0/public/OHLC?pair={ticker}USDC&interval=60"
            response = await self.client.get(url)
            
            if response.status_code != 200:
                log.warning(f"Failed to fetch data for {ticker} from Kraken")
                return None
            
            data = response.json()
            
            if "error" in data and data["error"]:
                log.warning(f"Kraken API error for {ticker}: {data['error']}")
                return None
            
            # Parse OHLC data
            result_key = list(data["result"].keys())[0] if "result" in data else None
            if not result_key or result_key == "last":
                return None
            
            ohlc_data = data["result"][result_key]
            
            df = pd.DataFrame(ohlc_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
            ])
            
            # Convert to numeric
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['volume'] = pd.to_numeric(df['volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            return df
            
        except Exception as e:
            log.error(f"Error fetching price data for {ticker}: {e}")
            return None
    
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
    
    def _generate_signal(self, ticker: str, indicators: Dict) -> TechnicalSignal:
        """Generate trading signal based on technical indicators."""
        signals = []
        weights = []
        
        current_price = indicators['current_price']
        
        # RSI Signal
        rsi = indicators['rsi']
        if rsi < 30:  # Oversold
            signals.append(TradeAction.BUY)
            weights.append(0.3)
        elif rsi > 70:  # Overbought
            signals.append(TradeAction.SELL)
            weights.append(0.3)
        else:
            signals.append(TradeAction.HOLD)
            weights.append(0.1)
        
        # MACD Signal
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        if macd > macd_signal:  # Bullish
            signals.append(TradeAction.BUY)
            weights.append(0.25)
        elif macd < macd_signal:  # Bearish
            signals.append(TradeAction.SELL)
            weights.append(0.25)
        else:
            signals.append(TradeAction.HOLD)
            weights.append(0.1)
        
        # Bollinger Bands Signal
        bb_upper = indicators['bollinger_upper']
        bb_lower = indicators['bollinger_lower']
        if current_price < bb_lower:  # Below lower band
            signals.append(TradeAction.BUY)
            weights.append(0.25)
        elif current_price > bb_upper:  # Above upper band
            signals.append(TradeAction.SELL)
            weights.append(0.25)
        else:
            signals.append(TradeAction.HOLD)
            weights.append(0.1)
        
        # SMA Crossover Signal
        sma_20 = indicators['sma_20']
        sma_50 = indicators.get('sma_50')
        if sma_50:
            if sma_20 > sma_50:  # Golden cross
                signals.append(TradeAction.BUY)
                weights.append(0.2)
            elif sma_20 < sma_50:  # Death cross
                signals.append(TradeAction.SELL)
                weights.append(0.2)
            else:
                signals.append(TradeAction.HOLD)
                weights.append(0.1)
        
        # Aggregate signals
        action_scores = {TradeAction.BUY: 0.0, TradeAction.SELL: 0.0, TradeAction.HOLD: 0.0}
        for signal, weight in zip(signals, weights):
            action_scores[signal] += weight
        
        recommendation = max(action_scores, key=action_scores.get)
        strength = action_scores[recommendation] / sum(weights)
        
        return TechnicalSignal(
            ticker=ticker,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            bollinger_upper=bb_upper,
            bollinger_lower=bb_lower,
            volume_sma=indicators['volume_sma'],
            recommendation=recommendation,
            strength=strength,
            timestamp=datetime.utcnow()
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

