""" 
News Feed Agent - Monitors crypto news and analyzes sentiment using LLM.
"""
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
from openai import AsyncOpenAI

from agents.base_agent import BaseAgent
from core.config import settings
from core.logging import log
from core.models import (
    AgentType,
    AgentMessage,
    AgentSignal,
    MessageType,
    NewsSentiment,
    SignalType,
)
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
            log.bind(agent="NEWS").info("News agent initialized with mock LLM service")
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
            
            log.bind(agent="NEWS").info("News agent initialized with real LLM service")

        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages."""
        # News agent primarily operates on its own cycle
        return None
    
    async def run_cycle(self):
        """Run periodic news monitoring and sentiment analysis."""
        log.bind(agent="NEWS").debug("News Agent running cycle...")
        
        try:
            # Fetch recent crypto news
            news_items = await self._fetch_crypto_news()
            
            if not news_items:
                log.bind(agent="NEWS").debug("No new news items found")
                return
            
            # Analyze sentiment for each news item
            for item in news_items[:5]:  # Analyze top 5 news items
                await self._analyze_news_sentiment(item)
            
            # Generate overall market sentiment
            await self._generate_market_sentiment(news_items)
            
        except Exception as e:
            log.error(f"News Agent cycle error: {e}")
    
    def get_cycle_interval(self) -> int:
        return settings.get_agent_cycle_seconds(self.agent_type)

    async def stop(self):
        if self.http_client:
            try:
                await self.http_client.aclose()
            except Exception:
                pass
            self.http_client = None
        await super().stop()
    
    async def _fetch_crypto_news(self) -> List[Dict]:
        """Fetch recent crypto news from multiple sources."""
        news_items: List[Dict] = []

        # Serve from cache if fresh
        cached_news = await self.redis.get_json("news:latest")
        if cached_news:
            timestamp = cached_news.get("timestamp")
            if timestamp:
                try:
                    cache_time = datetime.fromisoformat(timestamp)
                    if datetime.utcnow() - cache_time < timedelta(minutes=10):
                        return cached_news.get("items", [])
                except Exception:
                    pass

        primary = await self._fetch_primary_headlines()
        if primary:
            news_items.extend(primary)

        deep = await self._fetch_deep_search_news()
        if deep:
            news_items.extend(deep)

        arxiv_items = await self._fetch_arxiv_insights()
        if arxiv_items:
            news_items.extend(arxiv_items)

        await self.redis.set_json(
            "news:latest",
            {"items": news_items, "timestamp": datetime.utcnow().isoformat()},
            expire=600,
        )
        log.bind(agent="NEWS").info(
            "Fetched %d news items (primary=%d deep=%d arxiv=%d)",
            len(news_items),
            len(primary),
            len(deep),
            len(arxiv_items),
        )
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
            log.bind(agent="NEWS").info(
                "Mock sentiment for '%s' score=%.2f confidence=%.2f",
                title[:80],
                sentiment_score,
                confidence,
            )
            
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
                await self.redis.set_json(
                    f"news:sentiment:{ticker}",
                    {**sentiment.dict(), "generated_at": datetime.utcnow().isoformat()},
                    ttl=3600  # 1 hour
                )
                
                # Send signal if sentiment is strong
                if abs(sentiment_score) > 0.5 and confidence > 0.7:
                    await self._send_sentiment_signal(sentiment)

                await self._broadcast_news_event(
                    ticker,
                    sentiment_score,
                    confidence,
                    sentiment.summary,
                    sentiment.sources,
                )
            
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
            
            content = response.choices[0].message.content
            sentiment_data = self._extract_json_response(content)
            if not sentiment_data:
                return

            sentiment_score = float(sentiment_data.get("sentiment_score", 0.0))
            confidence = float(sentiment_data.get("confidence", 0.5))
            explanation = sentiment_data.get("explanation", "")

            affected_tickers = currencies or self._infer_currencies(title)
            log.bind(agent="NEWS").info(
                "Sentiment for '%s' score=%.2f confidence=%.2f tickers=%s",
                title[:80],
                sentiment_score,
                confidence,
                affected_tickers,
            )

            for ticker in affected_tickers:
                sentiment = NewsSentiment(
                    ticker=ticker,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    summary=f"{title} - {explanation}",
                    sources=[news_item.get("source", "Unknown")],
                    timestamp=datetime.utcnow()
                )

                await self.redis.set_json(
                    f"news:sentiment:{ticker}",
                    {**sentiment.dict(), "generated_at": datetime.utcnow().isoformat()},
                    expire=3600  # 1 hour
                )

                if abs(sentiment_score) > 0.5 and confidence > 0.7:
                    await self._send_sentiment_signal(sentiment)

                await self._broadcast_news_event(
                    ticker,
                    sentiment_score,
                    confidence,
                    sentiment.summary,
                    sentiment.sources,
                )
            
        except Exception as e:
            log.error(f"Error analyzing news sentiment: {e}")

    async def _fetch_primary_headlines(self) -> List[Dict[str, Any]]:
        if not self.http_client:
            return []
        params = {
            "auth_token": "free",
            "public": "true",
            "kind": "news",
            "filter": "hot",
        }
        try:
            response = await self.http_client.get("https://cryptopanic.com/api/v1/posts/", params=params)
            response.raise_for_status()
        except Exception as exc:
            log.bind(agent="NEWS").debug("Primary headline fetch failed: %s", exc)
            return []

        payload = response.json()
        results = payload.get("results", [])
        headlines: List[Dict[str, Any]] = []
        for item in results[:10]:
            headlines.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", {}).get("title", "Unknown"),
                    "published_at": item.get("published_at", ""),
                    "currencies": [c.get("code") for c in item.get("currencies", [])],
                }
            )
        return headlines

    async def _fetch_deep_search_news(self) -> List[Dict[str, Any]]:
        if not settings.deep_search_api_url or not self.http_client:
            return []
        params = {
            "q": "crypto market OR blockchain adoption",
            "page_size": 8,
        }
        headers = {}
        if settings.deep_search_api_key:
            headers["Authorization"] = settings.deep_search_api_key

        try:
            response = await self.http_client.get(
                settings.deep_search_api_url, params=params, headers=headers
            )
            response.raise_for_status()
        except Exception as exc:
            log.bind(agent="NEWS").debug("Deep search provider failed: %s", exc)
            return []

        data = response.json()
        articles = data.get("articles") or data.get("data") or []
        allowed_sources = {source.lower() for source in settings.deep_search_sources}
        items: List[Dict[str, Any]] = []
        for article in articles:
            source = ""
            article_source = article.get("source")
            if isinstance(article_source, dict):
                source = article_source.get("name", "")
            elif isinstance(article_source, str):
                source = article_source
            if allowed_sources and source and source.lower() not in allowed_sources:
                continue
            title = article.get("title") or article.get("headline") or ""
            items.append(
                {
                    "title": title,
                    "url": article.get("url") or article.get("link", ""),
                    "source": source or "DeepSearch",
                    "published_at": article.get("published_at") or article.get("date"),
                    "currencies": self._infer_currencies(title),
                }
            )
        return items

    async def _fetch_arxiv_insights(self) -> List[Dict[str, Any]]:
        if not settings.arxiv_enabled or not self.http_client:
            return []
        query = urlencode(
            {
                "search_query": 'all:"cryptocurrency" OR all:"blockchain"',
                "sortBy": "submittedDate",
                "max_results": 5,
            }
        )
        try:
            response = await self.http_client.get(f"https://export.arxiv.org/api/query?{query}")
            response.raise_for_status()
        except Exception as exc:
            log.bind(agent="NEWS").debug("ArXiv request failed: %s", exc)
            return []

        items: List[Dict[str, Any]] = []
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(response.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title = entry.findtext("atom:title", default="", namespaces=ns).strip()
                link_el = entry.find("atom:link", ns)
                link = link_el.attrib.get("href") if link_el is not None else ""
                published = entry.findtext("atom:published", default="", namespaces=ns)
                items.append(
                    {
                        "title": title,
                        "url": link,
                        "source": "arXiv",
                        "published_at": published,
                        "currencies": self._infer_currencies(title),
                    }
                )
        except Exception as exc:
            log.bind(agent="NEWS").debug("Failed to parse arXiv feed: %s", exc)
        return items
    
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
            await self.redis.set_json(
                "news:market_sentiment",
                {**market_sentiment.dict(), "generated_at": datetime.utcnow().isoformat()},
                ttl=1800  # 30 minutes
            )
            
            log.bind(agent="NEWS").info(
                "Market sentiment (mock) score=%.2f confidence=%.2f",
                market_sentiment.sentiment_score,
                market_sentiment.confidence,
            )
            await self._broadcast_news_event(
                None,
                market_sentiment.sentiment_score,
                market_sentiment.confidence,
                market_sentiment.summary,
                market_sentiment.sources,
            )
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
            sentiment_data = self._extract_json_response(content)
            if not sentiment_data:
                return

            market_sentiment = NewsSentiment(
                ticker=None,
                sentiment_score=float(sentiment_data.get("sentiment_score", 0.0)),
                confidence=float(sentiment_data.get("confidence", 0.5)),
                summary=sentiment_data.get("summary", ""),
                sources=[item.get("source", "Unknown") for item in news_items[:5]],
                timestamp=datetime.utcnow()
            )

            await self.redis.set_json(
                "news:market_sentiment",
                {**market_sentiment.dict(), "generated_at": datetime.utcnow().isoformat()},
                expire=3600
            )

            log.bind(agent="NEWS").info(
                "Market sentiment score=%.2f confidence=%.2f",
                market_sentiment.sentiment_score,
                market_sentiment.confidence,
            )
            await self._broadcast_news_event(
                None,
                market_sentiment.sentiment_score,
                market_sentiment.confidence,
                market_sentiment.summary,
                market_sentiment.sources,
            )
            
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
            log.bind(agent="NEWS").info(
                "Sent sentiment signal for %s: %s (score=%.2f)",
                sentiment.ticker,
                action,
                sentiment.sentiment_score,
            )
            
        except Exception as e:
            log.error(f"Error sending sentiment signal: {e}")

    async def _broadcast_news_event(
        self,
        ticker: Optional[str],
        sentiment_score: float,
        confidence: float,
        summary: str,
        sources: List[str],
    ):
        try:
            news_id_seed = f"{ticker}:{summary}"
            news_id = hashlib.sha1(news_id_seed.encode("utf-8")).hexdigest()
            payload = {
                "news_id": news_id,
                "ticker": ticker,
                "sentiment_score": sentiment_score,
                "confidence": confidence,
                "summary": summary,
                "sources": sources,
                "timestamp": datetime.utcnow().isoformat(),
            }
            message = AgentMessage(
                message_type=MessageType.NEWS_EVENT,
                sender=self.agent_type,
                payload=payload,
            )
            await self.send_message(message)
        except Exception as exc:
            log.error(f"Failed to broadcast news event: {exc}")

    def _infer_currencies(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = text.upper()
        inferred = [asset for asset in settings.supported_assets if asset.upper() in tokens]
        return inferred[:5]

    def _extract_json_response(self, content: Optional[str]) -> Optional[Dict[str, Any]]:
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None

