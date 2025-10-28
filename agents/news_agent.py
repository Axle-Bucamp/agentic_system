"""
News Feed Agent - Monitors crypto news and analyzes sentiment using LLM.
"""
import httpx
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from openai import AsyncOpenAI
from core.models import (
    AgentType, AgentMessage, MessageType, NewsSentiment,
    AgentSignal, SignalType
)
from core.config import settings
from core.logging import log
from agents.base_agent import BaseAgent
from core.mocks.mock_llm_service import get_mock_llm_service


class NewsAgent(BaseAgent):
    """Agent responsible for monitoring news and analyzing sentiment."""
    
    def __init__(self, redis_client):
        super().__init__(AgentType.NEWS, redis_client)
        self.llm_client: Optional[AsyncOpenAI] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.mock_llm_service = None
        self.use_mock = settings.use_mock_services
        
    async def initialize(self):
        """Initialize LLM client and HTTP client."""
        if self.use_mock:
            # Initialize mock LLM service
            self.mock_llm_service = await get_mock_llm_service()
            log.info("News agent initialized with mock LLM service")
        else:
            # Initialize OpenAI client (works with VLLM endpoints too)
            if settings.vllm_endpoint:
                self.llm_client = AsyncOpenAI(
                    api_key="dummy",  # VLLM doesn't require real key
                    base_url=settings.vllm_endpoint
                )
            elif settings.openai_api_key:
                self.llm_client = AsyncOpenAI(api_key=settings.openai_api_key)
            else:
                log.warning("No LLM endpoint configured, News Agent will have limited functionality")
            
            self.http_client = httpx.AsyncClient(timeout=30.0)
            log.info("News agent initialized with real LLM service")
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages."""
        # News agent primarily operates on its own cycle
        return None
    
    async def run_cycle(self):
        """Run periodic news monitoring and sentiment analysis."""
        log.debug("News Agent running cycle...")
        
        try:
            # Fetch recent crypto news
            news_items = await self._fetch_crypto_news()
            
            if not news_items:
                log.debug("No new news items found")
                return
            
            # Analyze sentiment for each news item
            for item in news_items[:5]:  # Analyze top 5 news items
                await self._analyze_news_sentiment(item)
            
            # Generate overall market sentiment
            await self._generate_market_sentiment(news_items)
            
        except Exception as e:
            log.error(f"News Agent cycle error: {e}")
    
    def get_cycle_interval(self) -> int:
        """Run every 15 minutes."""
        return 900
    
    async def _fetch_crypto_news(self) -> List[Dict]:
        """Fetch recent crypto news from various sources."""
        news_items = []
        
        try:
            # Check cache first
            cached_news = await self.redis.get_json("news:latest")
            if cached_news:
                cache_time = datetime.fromisoformat(cached_news.get("timestamp", "2000-01-01"))
                if datetime.utcnow() - cache_time < timedelta(minutes=10):
                    return cached_news.get("items", [])
            
            # Fetch from CryptoPanic API (free tier)
            # Note: In production, use actual API key
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                "auth_token": "free",  # Use proper API key in production
                "public": "true",
                "kind": "news",
                "filter": "hot"
            }
            
            response = await self.http_client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                for item in results[:10]:  # Get top 10 news
                    news_items.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "source": item.get("source", {}).get("title", "Unknown"),
                        "published_at": item.get("published_at", ""),
                        "currencies": [c.get("code") for c in item.get("currencies", [])]
                    })
            
            # Cache news
            await self.redis.set_json(
                "news:latest",
                {
                    "items": news_items,
                    "timestamp": datetime.utcnow().isoformat()
                },
                expire=600  # 10 minutes
            )
            
        except Exception as e:
            log.error(f"Error fetching crypto news: {e}")
        
        return news_items
    
    async def _analyze_news_sentiment(self, news_item: Dict):
        """Analyze sentiment of a news item using LLM."""
        if self.use_mock and self.mock_llm_service:
            # Use mock LLM service
            title = news_item.get("title", "")
            currencies = news_item.get("currencies", [])
            
            # Analyze sentiment using mock service
            sentiment_result = await self.mock_llm_service.analyze_sentiment(title)
            
            # Convert to our format
            sentiment_score = sentiment_result["positive_score"] - sentiment_result["negative_score"]
            confidence = sentiment_result["confidence"]
            explanation = f"Mock analysis: {sentiment_result['sentiment']} sentiment"
            
            # Process each currency
            for ticker in currencies:
                sentiment = NewsSentiment(
                    ticker=ticker,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    summary=explanation,
                    sources=[news_item.get("source", "Mock News")]
                )
                
                # Cache sentiment
                await self.redis_client.set_json(
                    f"news:sentiment:{ticker}",
                    sentiment.dict(),
                    ttl=3600  # 1 hour
                )
                
                # Send signal if sentiment is strong
                if abs(sentiment_score) > 0.5 and confidence > 0.7:
                    await self._send_sentiment_signal(sentiment)
            
            return
        
        if not self.llm_client:
            return
        
        try:
            title = news_item.get("title", "")
            currencies = news_item.get("currencies", [])
            
            # Create prompt for sentiment analysis
            prompt = f"""Analyze the sentiment of this crypto news headline and provide a sentiment score.

Headline: {title}
Related cryptocurrencies: {', '.join(currencies) if currencies else 'General market'}

Provide:
1. Sentiment score from -1 (very negative) to +1 (very positive)
2. Confidence level from 0 to 1
3. Brief explanation (max 50 words)

Respond in JSON format:
{{"sentiment_score": <float>, "confidence": <float>, "explanation": "<string>"}}"""
            
            # Call LLM
            response = await self.llm_client.chat.completions.create(
                model="gpt-4.1-mini",  # Will use VLLM model if configured
                messages=[
                    {"role": "system", "content": "You are a crypto market sentiment analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Try to extract JSON
            import json
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                sentiment_data = json.loads(json_match.group())
                
                sentiment_score = sentiment_data.get("sentiment_score", 0.0)
                confidence = sentiment_data.get("confidence", 0.5)
                explanation = sentiment_data.get("explanation", "")
                
                # Determine affected tickers
                affected_tickers = []
                for currency in currencies:
                    # Map currency codes to our supported tickers
                    ticker_map = {
                        "BTC": "BTC", "ETH": "ETH", "SOL": "SOL",
                        "ADA": "ADA", "DOGE": "DOGE", "MATIC": "MATIC"
                    }
                    if currency in ticker_map:
                        affected_tickers.append(ticker_map[currency])
                
                # Create sentiment record
                for ticker in affected_tickers:
                    sentiment = NewsSentiment(
                        ticker=ticker,
                        sentiment_score=sentiment_score,
                        confidence=confidence,
                        summary=f"{title} - {explanation}",
                        sources=[news_item.get("source", "Unknown")],
                        timestamp=datetime.utcnow()
                    )
                    
                    # Cache sentiment
                    await self.redis.set_json(
                        f"news:sentiment:{ticker}",
                        sentiment.dict(),
                        expire=3600  # 1 hour
                    )
                    
                    # Send signal if sentiment is strong
                    if abs(sentiment_score) > 0.5 and confidence > 0.7:
                        await self._send_sentiment_signal(sentiment)
            
        except Exception as e:
            log.error(f"Error analyzing news sentiment: {e}")
    
    async def _generate_market_sentiment(self, news_items: List[Dict]):
        """Generate overall market sentiment from multiple news items."""
        if self.use_mock and self.mock_llm_service:
            # Use mock LLM service for market sentiment
            headlines = [item.get("title", "") for item in news_items[:10]]
            combined_text = " ".join(headlines)
            
            # Analyze overall sentiment
            sentiment_result = await self.mock_llm_service.analyze_sentiment(combined_text)
            
            # Generate summary
            summary = await self.mock_llm_service.generate_summary(combined_text, max_length=100)
            
            # Create market sentiment
            market_sentiment = NewsSentiment(
                ticker=None,  # General market sentiment
                sentiment_score=sentiment_result["positive_score"] - sentiment_result["negative_score"],
                confidence=sentiment_result["confidence"],
                summary=summary,
                sources=["Mock News Analysis"]
            )
            
            # Cache market sentiment
            await self.redis_client.set_json(
                "news:market_sentiment",
                market_sentiment.dict(),
                ttl=1800  # 30 minutes
            )
            
            log.info(f"Market sentiment: {market_sentiment.sentiment_score:.2f} (confidence: {market_sentiment.confidence:.2f})")
            return
        
        if not self.llm_client or not news_items:
            return
        
        try:
            # Aggregate headlines
            headlines = [item.get("title", "") for item in news_items[:10]]
            headlines_text = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines)])
            
            prompt = f"""Analyze the overall crypto market sentiment based on these recent headlines:

{headlines_text}

Provide:
1. Overall market sentiment score from -1 (very bearish) to +1 (very bullish)
2. Confidence level from 0 to 1
3. Brief market summary (max 100 words)

Respond in JSON format:
{{"sentiment_score": <float>, "confidence": <float>, "summary": "<string>"}}"""
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a crypto market analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            
            import json
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                sentiment_data = json.loads(json_match.group())
                
                market_sentiment = NewsSentiment(
                    ticker=None,  # General market sentiment
                    sentiment_score=sentiment_data.get("sentiment_score", 0.0),
                    confidence=sentiment_data.get("confidence", 0.5),
                    summary=sentiment_data.get("summary", ""),
                    sources=[item.get("source", "Unknown") for item in news_items[:5]],
                    timestamp=datetime.utcnow()
                )
                
                # Cache market sentiment
                await self.redis.set_json(
                    "news:market_sentiment",
                    market_sentiment.dict(),
                    expire=3600
                )
                
                log.info(f"Market sentiment: {market_sentiment.sentiment_score:.2f} (confidence: {market_sentiment.confidence:.2f})")
            
        except Exception as e:
            log.error(f"Error generating market sentiment: {e}")
    
    async def _send_sentiment_signal(self, sentiment: NewsSentiment):
        """Send sentiment signal to orchestrator."""
        try:
            # Determine action based on sentiment
            action = None
            if sentiment.sentiment_score > 0.5:
                action = "BUY"
            elif sentiment.sentiment_score < -0.5:
                action = "SELL"
            else:
                action = "HOLD"
            
            signal = AgentSignal(
                agent_type=self.agent_type,
                signal_type=SignalType.NEWS_SENTIMENT,
                ticker=sentiment.ticker,
                action=action,
                confidence=sentiment.confidence,
                data={
                    "sentiment_score": sentiment.sentiment_score,
                    "summary": sentiment.summary,
                    "sources": sentiment.sources
                },
                reasoning=f"News sentiment analysis: {sentiment.summary}"
            )
            
            await self.send_signal(signal.dict())
            log.info(f"Sent sentiment signal for {sentiment.ticker}: {action} (score: {sentiment.sentiment_score:.2f})")
            
        except Exception as e:
            log.error(f"Error sending sentiment signal: {e}")

