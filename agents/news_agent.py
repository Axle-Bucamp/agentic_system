"""News Feed Agent - Monitors crypto news and analyzes sentiment using LLM."""
import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import httpx
from openai import AsyncOpenAI

from agents.base_agent import BaseAgent
from core.config import settings
from core.logging import log
from core.models import (
    AgentType,
    AgentMessage,
    AgentSignal,
    FactInsight,
    MessageType,
    NewsSentiment,
    SignalType,
)
from core.mocks.mock_llm_service import get_mock_llm_service
from core.research import (
    fetch_arxiv_entries,
    fetch_coin_bureau_updates,
    fetch_google_scholar_entries,
    fetch_yahoo_finance_headlines,
)
from core.pipelines.storage import get_fact_insight


class NewsAgent(BaseAgent):
    """Agent responsible for monitoring news and analyzing sentiment."""
    
    def __init__(self, redis_client):
        super().__init__(AgentType.NEWS, redis_client)
        self.llm_client: Optional[AsyncOpenAI] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.mock_llm_service = None
        self.use_mock = settings.use_mock_services
        self._use_gemini = False
        self.llm_model_name = settings.news_llm_model
        self.source_weights: Dict[str, float] = settings.news_source_weights.copy()
        self._mock_fallback_warned = False
        
    async def initialize(self):
        """Initialize LLM client and HTTP client."""
        if self.use_mock:
            # Initialize mock LLM service
            self.mock_llm_service = await get_mock_llm_service()
            log.bind(agent="NEWS").info("News agent initialized with mock LLM service")
        else:
            # Initialize OpenAI-compatible client (OpenRouter/OpenAI/VLLM)
            if settings.vllm_endpoint:
                self.llm_client = AsyncOpenAI(
                    api_key="dummy",  # VLLM doesn't require real key
                    base_url=settings.vllm_endpoint
                )
            elif settings.openai_api_key:
                client_kwargs = settings.openai_client_kwargs
                client_kwargs.setdefault("timeout", 30.0)
                client_kwargs.setdefault("max_retries", 2)
                self.llm_client = AsyncOpenAI(
                    api_key=settings.openai_api_key,
                    **client_kwargs,
                )
            elif settings.gemini_api_key:
                self._use_gemini = True
            else:
                log.warning("No LLM endpoint configured, News Agent will have limited functionality")
            
            if self.llm_client:
                log.bind(agent="NEWS").info("News agent initialized with real LLM service")
            elif self._use_gemini:
                log.bind(agent="NEWS").info("News agent configured to use Gemini service")

        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def _ensure_mock_service(self) -> bool:
        """Initialise the mock LLM service if available."""
        if not self.use_mock:
            return False
        if self.mock_llm_service:
            return True

        try:
            self.mock_llm_service = await get_mock_llm_service()
            if not self._mock_fallback_warned:
                log.bind(agent="NEWS").warning("Falling back to mock LLM service for news sentiment analysis")
                self._mock_fallback_warned = True
            return True
        except Exception as exc:
            log.bind(agent="NEWS").error("Unable to initialise mock LLM service: %s", exc)
            return False

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages."""
        # News agent primarily operates on its own cycle
        return None
    
    async def _gemini_generate(self, prompt: str, max_tokens: int = 256) -> Optional[str]:
        """Generate text using Gemini when configured."""
        if not settings.gemini_api_key:
            return None

        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=30.0)

        model_name = settings.camel_primary_model or "gemini-1.5-pro"
        if not model_name.lower().startswith("gemini"):
            model_name = "gemini-1.5-pro"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": max(1, min(max_tokens, 1024)),
            },
        }

        try:
            response = await self.http_client.post(
                url,
                params={"key": settings.gemini_api_key},
                json=payload,
            )
            response.raise_for_status()
        except Exception as exc:
            log.bind(agent="NEWS").error("Gemini request failed: %s", exc)
            return None

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        return text.strip() if text else None
    
    async def _execute_json_prompt(self, system_prompt: str, user_prompt: str, max_tokens: int) -> Optional[str]:
        """Execute an LLM completion with lightweight retry logic."""
        attempts = 3
        delay = 0.7
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                if self._use_gemini:
                    merged_prompt = f"{system_prompt}\n\n{user_prompt}"
                    content = await self._gemini_generate(merged_prompt, max_tokens=max_tokens)
                else:
                    if not self.llm_client:
                        return None
                    response = await self.llm_client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.3,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                    message = response.choices[0].message
                    content = getattr(message, "content", None)
                    if not content and hasattr(message, "parsed"):
                        try:
                            return json.dumps(message.parsed)
                        except TypeError:
                            content = json.dumps(message.parsed, default=str)

                if content:
                    return content
                last_error = RuntimeError("Empty completion response")
            except Exception as exc:
                last_error = exc
                log.bind(agent="NEWS", attempt=attempt, total_attempts=attempts).warning(
                    "LLM completion failed: {}", exc
                )
                await asyncio.sleep(delay)
                delay *= 1.6

        if last_error:
            log.bind(agent="NEWS").error("LLM completion exhausted retries: {}", last_error)
            raise RuntimeError(str(last_error))
        return None
    
    async def run_cycle(self):
        """Run periodic news monitoring and sentiment analysis."""
        log.bind(agent="NEWS").debug("News Agent running cycle...")
        
        try:
            await self._refresh_source_weights()
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
        """Fetch recent crypto news from weighted sources."""
        news_items: List[Dict] = []

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

        source_map = await self._gather_weighted_sources()
        breakdown: Dict[str, int] = {}
        for source_key, items in source_map.items():
            enriched = self._enrich_source_items(items, source_key)
            if not enriched:
                continue
            news_items.extend(enriched)
            breakdown[source_key] = len(enriched)

        if news_items:
            # Prioritize the highest weighted sources first
            news_items.sort(
                key=lambda item: (
                    item.get("source_weight", 0.1),
                    item.get("published_at") or "",
                ),
                reverse=True,
            )

        await self.redis.set_json(
            "news:latest",
            {"items": news_items, "timestamp": datetime.utcnow().isoformat()},
            expire=600,
        )

        breakdown_text = ", ".join(f"{name}={count}" for name, count in breakdown.items())
        log.bind(agent="NEWS").info(
            "Fetched %d weighted news items (%s)",
            len(news_items),
            breakdown_text or "no sources",
        )
        return news_items

    async def _refresh_source_weights(self) -> None:
        try:
            dashboard = await self.redis.get_json("dashboard:settings") or {}
            weights = dashboard.get("news_source_weights")
            if isinstance(weights, dict):
                parsed = {key: float(value) for key, value in weights.items()}
                if parsed:
                    self.source_weights = parsed
        except Exception as exc:
            log.bind(agent="NEWS").debug("Failed to refresh source weights: %s", exc)

    async def _gather_weighted_sources(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.http_client:
            return {}

        client = self.http_client
        tasks: List[tuple[str, Any]] = [
            ("yahoo_finance", fetch_yahoo_finance_headlines(client)),
            ("coin_bureau", fetch_coin_bureau_updates(client)),
        ]
        if settings.arxiv_enabled:
            tasks.append(
                (
                    "arxiv",
                    fetch_arxiv_entries(
                        client,
                        'all:"cryptocurrency" OR all:"blockchain" OR all:"digital assets"',
                        limit=6,
                    ),
                )
            )
        tasks.append(
            (
                "google_scholar",
                fetch_google_scholar_entries(
                    client,
                    "cryptocurrency OR blockchain adoption OR decentralized finance",
                    limit=6,
                ),
            )
        )

        results = await asyncio.gather(
            *(coro for _, coro in tasks), return_exceptions=True
        )

        sources: Dict[str, List[Dict[str, Any]]] = {}
        for (source_key, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                log.bind(agent="NEWS").debug(
                    "Source fetch failed: %s error=%s", source_key, result
                )
                continue
            items = list(result)
            if not items:
                log.bind(agent="NEWS").debug(
                    "Source %s returned no items; skipping for this cycle", source_key
                )
                continue
            sources[source_key] = items

        # Retain CryptoPanic as a lightweight general feed
        primary = await self._fetch_primary_headlines()
        if primary:
            sources["cryptopanic"] = primary

        return sources

    def _enrich_source_items(
        self, items: List[Dict[str, Any]], source_key: str
    ) -> List[Dict[str, Any]]:
        if not items:
            return []

        weight = self._resolve_source_weight(source_key)
        if weight <= 0:
            log.bind(agent="NEWS").debug(
                "Source %s disabled via weight %.2f; skipping items", source_key, weight
            )
            return []

        label = self._resolve_source_label(source_key)

        enriched: List[Dict[str, Any]] = []
        for item in items:
            entry = dict(item)
            entry.setdefault("source_key", source_key)
            entry.setdefault("source", label)
            entry["source_weight"] = weight
            if not entry.get("currencies"):
                entry["currencies"] = self._infer_currencies(entry.get("title", ""))
            enriched.append(entry)
        return enriched

    def _resolve_source_weight(self, source_key: Optional[str]) -> float:
        if not source_key:
            return 0.1
        if source_key in self.source_weights:
            return max(self.source_weights[source_key], 0.05)
        # Default fallback weights for legacy feeds
        fallback_weights = {
            "cryptopanic": 0.15,
        }
        return fallback_weights.get(source_key, 0.1)

    def _resolve_source_label(self, source_key: Optional[str]) -> str:
        mapping = {
            "yahoo_finance": "Yahoo Finance",
            "coin_bureau": "Coin Bureau",
            "arxiv": "arXiv",
            "google_scholar": "Google Scholar",
            "cryptopanic": "CryptoPanic",
        }
        if not source_key:
            return "Unknown"
        return mapping.get(source_key, source_key.replace("_", " ").title())

    def _average_source_weight(self, items: List[Dict[str, Any]], limit: int = 5) -> float:
        if not items:
            return 0.1
        weights: List[float] = []
        for item in items[:limit]:
            if "source_weight" in item:
                weights.append(float(item["source_weight"]))
            else:
                weights.append(self._resolve_source_weight(item.get("source_key")))
        return sum(weights) / len(weights) if weights else 0.1
    
    async def _analyze_news_sentiment_mock(self, news_item: Dict) -> None:
        """Fallback sentiment analysis using the mock LLM service."""
        if not await self._ensure_mock_service():
            return

        title = news_item.get("title", "")
        currencies = news_item.get("currencies", [])

        sentiment_result = await self.mock_llm_service.analyze_sentiment(title)
        source_weight = self._resolve_source_weight(news_item.get("source_key"))
        base_confidence = float(sentiment_result.get("confidence", 0.5))
        confidence = min(1.0, max(base_confidence, 0.2) * (0.6 + source_weight))

        sentiment_score = sentiment_result.get("positive_score", 0.0) - sentiment_result.get("negative_score", 0.0)
        explanation = f"Mock analysis [{source_weight:.2f} weight]: {sentiment_result.get('sentiment', 'neutral')} sentiment"

        log.bind(agent="NEWS").info(
            "Mock sentiment for '%s' score=%.2f confidence=%.2f",
            title[:80],
            sentiment_score,
            confidence,
        )

        targets = currencies or self._infer_currencies(title)
        for ticker in targets:
            sentiment = NewsSentiment(
                ticker=ticker,
                sentiment_score=sentiment_score,
                confidence=confidence,
                summary=explanation,
                sources=[news_item.get("source", "Mock News")],
                timestamp=datetime.utcnow(),
            )

            await self.redis.set_json(
                f"news:sentiment:{ticker}",
                {**sentiment.dict(), "generated_at": datetime.utcnow().isoformat()},
                expire=3600,
            )

            if abs(sentiment_score) > 0.5 and confidence > 0.7:
                await self._send_sentiment_signal(sentiment)

            await self._broadcast_news_event(
                ticker,
                sentiment_score,
                confidence,
                sentiment.summary,
                sentiment.sources,
                source_weight,
            )
    
    async def _analyze_news_sentiment(self, news_item: Dict):
        """Analyze sentiment of a news item using LLM."""
        if self.use_mock:
            await self._analyze_news_sentiment_mock(news_item)
            return

        if not self.llm_client and not self._use_gemini:
            log.bind(agent="NEWS").error("No LLM client configured; skipping sentiment analysis.")
            await self._apply_fact_guardrail_for_tickers(news_item, [])
            return

        try:
            title = news_item.get("title", "")
            currencies = news_item.get("currencies", [])
            source_weight = self._resolve_source_weight(news_item.get("source_key"))
            
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
            
            content = await self._execute_json_prompt(
                system_prompt="You are a crypto market sentiment analyst. Respond only with valid JSON.",
                user_prompt=prompt,
                max_tokens=200,
            )
            sentiment_data = self._extract_json_response(content)
            if not sentiment_data:
                raise RuntimeError("Unable to parse LLM sentiment response")

            sentiment_score = float(sentiment_data.get("sentiment_score", 0.0))
            base_confidence = float(sentiment_data.get("confidence", 0.5))
            confidence = min(1.0, max(base_confidence, 0.2) * (0.6 + source_weight))
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
                    summary=f"{title} [{source_weight:.2f} weight] - {explanation}",
                    sources=[news_item.get("source", "Unknown")],
                    timestamp=datetime.utcnow()
                )

                await self._persist_ticker_sentiment(sentiment, source_weight)

        except Exception as e:
            log.error(f"Error analyzing news sentiment: {e}")
            affected_tickers = news_item.get("currencies") or self._infer_currencies(news_item.get("title", ""))
            news_item["error_reason"] = str(e)
            await self._apply_fact_guardrail_for_tickers(news_item, affected_tickers)

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

    async def _generate_market_sentiment(self, news_items: List[Dict]):
        """Generate overall market sentiment from multiple news items."""
        if self.use_mock:
            await self._generate_market_sentiment_mock(news_items)
            return

        if not news_items:
            return

        if not self.llm_client and not self._use_gemini:
            log.bind(agent="NEWS").error("No LLM client configured; skipping market sentiment.")
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
            
            content = await self._execute_json_prompt(
                system_prompt="You are a crypto market analyst. Respond only with valid JSON.",
                user_prompt=prompt,
                max_tokens=300,
            )
            if not content:
                raise RuntimeError("Empty response for market sentiment prompt")
            sentiment_data = self._extract_json_response(content)
            if not sentiment_data:
                return

            avg_weight = self._average_source_weight(news_items, limit=8)
            market_sentiment = NewsSentiment(
                ticker=None,
                sentiment_score=float(sentiment_data.get("sentiment_score", 0.0)),
                confidence=min(
                    1.0,
                    max(float(sentiment_data.get("confidence", 0.5)), 0.2)
                    * (0.6 + avg_weight),
                ),
                summary=f"[avg weight {avg_weight:.2f}] {sentiment_data.get('summary', '')}",
                sources=[item.get("source", "Unknown") for item in news_items[:5]],
                timestamp=datetime.utcnow()
            )
            await self._persist_market_sentiment(market_sentiment, avg_weight)
        except Exception as e:
            log.error(f"Error generating market sentiment: {e}")
            guardrail = await self._fact_guardrail(None, f"Market overview fallback ({e})")
            if guardrail:
                await self._persist_market_sentiment(guardrail, 0.0)
            else:
                fallback = NewsSentiment(
                    ticker=None,
                    sentiment_score=0.0,
                    confidence=0.1,
                    summary=f"[degraded] Unable to compute market sentiment (reason: {e})",
                    sources=["NewsAgent"],
                    timestamp=datetime.utcnow(),
                )
                await self._persist_market_sentiment(fallback, 0.0)

    async def _generate_market_sentiment_mock(self, news_items: List[Dict]) -> None:
        """Generate market sentiment using the mock LLM service."""
        if not await self._ensure_mock_service():
            return

        headlines = [item.get("title", "") for item in news_items[:10]]
        combined_text = " ".join(headlines)

        sentiment_result = await self.mock_llm_service.analyze_sentiment(combined_text)
        summary = await self.mock_llm_service.generate_summary(combined_text, max_length=100)
        avg_weight = self._average_source_weight(news_items, limit=8)
        confidence = min(
            1.0,
            max(sentiment_result.get("confidence", 0.5), 0.2) * (0.6 + avg_weight),
        )

        market_sentiment = NewsSentiment(
            ticker=None,
            sentiment_score=sentiment_result.get("positive_score", 0.0)
            - sentiment_result.get("negative_score", 0.0),
            confidence=confidence,
            summary=f"[avg weight {avg_weight:.2f}] {summary}",
            sources=["Mock News Analysis"],
            timestamp=datetime.utcnow(),
        )

        await self.redis.set_json(
            "news:market_sentiment",
            {**market_sentiment.dict(), "generated_at": datetime.utcnow().isoformat()},
            expire=1800,
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
            avg_weight,
        )
    
    async def _persist_market_sentiment(self, sentiment: NewsSentiment, avg_weight: float) -> None:
        await self.redis.set_json(
            "news:market_sentiment",
            {**sentiment.dict(), "generated_at": datetime.utcnow().isoformat()},
            expire=3600,
        )

        log.bind(agent="NEWS").info(
            "Market sentiment score=%.2f confidence=%.2f",
            sentiment.sentiment_score,
            sentiment.confidence,
        )
        await self._broadcast_news_event(
            None,
            sentiment.sentiment_score,
            sentiment.confidence,
            sentiment.summary,
            sentiment.sources,
            avg_weight,
        )

    async def _persist_ticker_sentiment(self, sentiment: NewsSentiment, source_weight: float) -> None:
        if not sentiment.ticker:
            return

        await self.redis.set_json(
            f"news:sentiment:{sentiment.ticker}",
            {**sentiment.dict(), "generated_at": datetime.utcnow().isoformat()},
            expire=3600,
        )

        if abs(sentiment.sentiment_score) > 0.5 and sentiment.confidence > 0.7:
            await self._send_sentiment_signal(sentiment)

        await self._broadcast_news_event(
            sentiment.ticker,
            sentiment.sentiment_score,
            sentiment.confidence,
            sentiment.summary,
            sentiment.sources,
            source_weight,
        )

    async def _fact_guardrail(self, ticker: Optional[str], fallback_summary: str) -> Optional[NewsSentiment]:
        insight: Optional[FactInsight] = await get_fact_insight(self.redis, ticker)
        if not insight:
            return None

        resolved_ticker = insight.ticker or ticker
        summary = insight.thesis or fallback_summary
        confidence = max(min(insight.confidence, 1.0), 0.2)

        return NewsSentiment(
            ticker=resolved_ticker,
            sentiment_score=insight.sentiment_score,
            confidence=confidence,
            summary=f"[fact guardrail] {summary}",
            sources=["FactPipeline"],
            timestamp=datetime.utcnow(),
        )

    async def _apply_fact_guardrail_for_tickers(self, news_item: Dict[str, Any], tickers: List[str]) -> None:
        title = news_item.get("title", "News item")
        error_reason = news_item.get("error_reason")
        source_weight = self._resolve_source_weight(news_item.get("source_key"))

        candidates = tickers or [None]
        applied_guardrail = False
        for ticker in candidates:
            guardrail = await self._fact_guardrail(ticker, title)
            if not guardrail:
                continue
            log.bind(agent="NEWS", guardrail=True, ticker=guardrail.ticker or "market").info(
                "Fact guardrail applied with sentiment %.3f (confidence %.2f)",
                guardrail.sentiment_score,
                guardrail.confidence,
            )
            if guardrail.ticker:
                await self._persist_ticker_sentiment(guardrail, source_weight)
            else:
                await self._persist_market_sentiment(guardrail, source_weight)

            applied_guardrail = True
            break

        if not applied_guardrail:
            fallback_ticker = candidates[0] if candidates else None
            summary_suffix = ""
            if error_reason:
                summary_suffix = f" (reason: {error_reason})"
            fallback_sentiment = NewsSentiment(
                ticker=fallback_ticker,
                sentiment_score=0.0,
                confidence=0.1,
                summary=f"[degraded] Unable to analyze sentiment for '{title}'. Using neutral placeholder{summary_suffix}.",
                sources=[news_item.get("source", "Unknown")],
                timestamp=datetime.utcnow(),
            )
            if fallback_ticker:
                await self._persist_ticker_sentiment(fallback_sentiment, source_weight)
            else:
                await self._persist_market_sentiment(fallback_sentiment, source_weight or 0.0)
            log.bind(agent="NEWS", guardrail=True).warning(
                "No fact guardrail available; recorded neutral fallback sentiment for {}", fallback_ticker or "market"
            )

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
        source_weight: Optional[float] = None,
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
                "source_weight": source_weight
                if source_weight is not None
                else self._resolve_source_weight(
                    sources[0].lower().replace(" ", "_") if sources else None
                ),
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
            log.bind(agent="NEWS").warning("Raw LLM response not pure JSON: {}", content[:500])
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                log.bind(agent="NEWS").debug("Failed to parse JSON fragment from response: {}", match.group()[:500])
                return None
        return None

