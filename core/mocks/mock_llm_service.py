"""
Mock LLM Service

Provides a comprehensive mock implementation of LLM services
for testing and development when OpenAI/VLLM is not available.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from core.logging import log


class SentimentType(Enum):
    """Sentiment types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class NewsArticle:
    """News article data structure."""
    title: str
    content: str
    source: str
    published_at: datetime
    sentiment: SentimentType
    confidence: float
    keywords: List[str]
    url: str


class MockLLMService:
    """Mock LLM service that simulates OpenAI/VLLM behavior."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.api_key = self.config.get("api_key", "mock_api_key")
        self.base_url = self.config.get("base_url", "https://api.openai.com/v1")
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.max_tokens = self.config.get("max_tokens", 1000)
        self.temperature = self.config.get("temperature", 0.7)
        
        # Mock data
        self.news_articles = []
        self.sentiment_cache = {}
        self.response_cache = {}
        
        # Initialize with sample news
        self._initialize_sample_news()
        
        log.info("Mock LLM Service initialized")
    
    def _initialize_sample_news(self):
        """Initialize with sample news articles."""
        sample_news = [
            {
                "title": "Bitcoin Reaches New All-Time High Amid Institutional Adoption",
                "content": "Bitcoin has reached a new all-time high of $70,000 as major institutions continue to adopt cryptocurrency. The surge is driven by increased demand from corporate treasuries and growing acceptance in traditional finance.",
                "source": "CoinDesk",
                "sentiment": SentimentType.POSITIVE,
                "keywords": ["bitcoin", "cryptocurrency", "institutional", "adoption", "price"]
            },
            {
                "title": "Ethereum Network Upgrade Improves Scalability and Reduces Fees",
                "content": "The latest Ethereum network upgrade has successfully improved transaction throughput and reduced gas fees. Developers report significant improvements in network performance and user experience.",
                "source": "Ethereum Foundation",
                "sentiment": SentimentType.POSITIVE,
                "keywords": ["ethereum", "upgrade", "scalability", "fees", "network"]
            },
            {
                "title": "Regulatory Concerns Continue to Impact Crypto Markets",
                "content": "Recent regulatory announcements from various governments have created uncertainty in cryptocurrency markets. Traders are cautious as new compliance requirements are being discussed.",
                "source": "Financial Times",
                "sentiment": SentimentType.NEGATIVE,
                "keywords": ["regulation", "compliance", "uncertainty", "government", "markets"]
            },
            {
                "title": "DeFi Protocol Suffers Smart Contract Exploit",
                "content": "A popular DeFi protocol has been exploited, resulting in the loss of millions of dollars. Security researchers are investigating the vulnerability and working on a fix.",
                "source": "The Block",
                "sentiment": SentimentType.NEGATIVE,
                "keywords": ["defi", "exploit", "security", "vulnerability", "loss"]
            },
            {
                "title": "Major Bank Announces Cryptocurrency Custody Services",
                "content": "A leading traditional bank has announced plans to offer cryptocurrency custody services to institutional clients. This represents a significant step towards mainstream adoption.",
                "source": "Reuters",
                "sentiment": SentimentType.POSITIVE,
                "keywords": ["bank", "custody", "institutional", "adoption", "services"]
            }
        ]
        
        for i, news in enumerate(sample_news):
            article = NewsArticle(
                title=news["title"],
                content=news["content"],
                source=news["source"],
                published_at=datetime.utcnow() - timedelta(hours=random.randint(1, 24)),
                sentiment=news["sentiment"],
                confidence=random.uniform(0.7, 0.95),
                keywords=news["keywords"],
                url=f"https://example.com/news/{i+1}"
            )
            self.news_articles.append(article)
    
    async def _simulate_api_delay(self):
        """Simulate API response delay."""
        delay = random.uniform(0.1, 0.5)  # 100-500ms delay
        await asyncio.sleep(delay)
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        await self._simulate_api_delay()
        
        # Check cache first
        cache_key = f"sentiment_{hash(text)}"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # Simple sentiment analysis based on keywords
        positive_words = ["good", "great", "excellent", "positive", "bullish", "up", "rise", "gain", "profit", "success"]
        negative_words = ["bad", "terrible", "awful", "negative", "bearish", "down", "fall", "loss", "crash", "exploit"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = SentimentType.POSITIVE
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = SentimentType.NEGATIVE
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = SentimentType.NEUTRAL
            confidence = 0.5
        
        result = {
            "sentiment": sentiment.value,
            "confidence": round(confidence, 2),
            "positive_score": positive_count / len(text.split()),
            "negative_score": negative_count / len(text.split()),
            "neutral_score": 1 - (positive_count + negative_count) / len(text.split())
        }
        
        # Cache result
        self.sentiment_cache[cache_key] = result
        return result
    
    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of text."""
        await self._simulate_api_delay()
        
        # Simple extractive summarization
        sentences = text.split('. ')
        if len(sentences) <= 2:
            return text
        
        # Take first and last sentences
        summary = sentences[0] + '. ' + sentences[-1] + '.'
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + '...'
        
        return summary
    
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        await self._simulate_api_delay()
        
        # Simple keyword extraction
        words = text.lower().split()
        
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those"}
        
        # Count word frequency
        word_count = {}
        for word in words:
            word = word.strip('.,!?;:"()[]{}')
            if len(word) > 3 and word not in stop_words:
                word_count[word] = word_count.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        keywords = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in keywords[:max_keywords]]
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Dict[str, Any]:
        """Generate chat completion response."""
        await self._simulate_api_delay()
        
        model = model or self.model
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        # Check cache first
        cache_key = f"chat_{hash(str(messages))}_{model}_{temperature}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Generate response based on the last message
        last_message = messages[-1]["content"] if messages else ""
        
        # Simple response generation based on keywords
        if "sentiment" in last_message.lower():
            response = "Based on the analysis, the market sentiment appears to be mixed with both positive and negative indicators. The overall confidence level is moderate."
        elif "price" in last_message.lower() or "forecast" in last_message.lower():
            response = "The price analysis suggests a potential upward trend with moderate volatility. Technical indicators show mixed signals, and fundamental factors remain supportive."
        elif "news" in last_message.lower():
            response = "Recent news coverage has been generally positive, with several major developments supporting market optimism. However, some regulatory concerns persist."
        elif "risk" in last_message.lower():
            response = "Risk assessment indicates moderate levels of market risk. Key factors include regulatory uncertainty, market volatility, and potential liquidity concerns."
        else:
            response = "I understand your query about the trading system. Based on current market conditions and available data, I recommend a cautious approach with careful risk management."
        
        # Truncate response if too long
        if len(response) > max_tokens:
            response = response[:max_tokens-3] + "..."
        
        result = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(str(messages)),
                "completion_tokens": len(response.split()),
                "total_tokens": len(str(messages)) + len(response.split())
            }
        }
        
        # Cache result
        self.response_cache[cache_key] = result
        return result
    
    async def get_news_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Get news sentiment for a specific ticker."""
        await self._simulate_api_delay()
        
        # Filter news for the ticker
        ticker_news = [
            article for article in self.news_articles
            if ticker.lower() in article.title.lower() or ticker.lower() in article.content.lower()
        ]
        
        if not ticker_news:
            # Generate some mock news if none found
            ticker_news = self._generate_mock_news_for_ticker(ticker)
        
        # Calculate overall sentiment
        sentiments = [article.sentiment for article in ticker_news]
        confidences = [article.confidence for article in ticker_news]
        
        positive_count = sentiments.count(SentimentType.POSITIVE)
        negative_count = sentiments.count(SentimentType.NEGATIVE)
        neutral_count = sentiments.count(SentimentType.NEUTRAL)
        
        total_articles = len(ticker_news)
        
        if positive_count > negative_count:
            overall_sentiment = "positive"
            sentiment_score = positive_count / total_articles
        elif negative_count > positive_count:
            overall_sentiment = "negative"
            sentiment_score = negative_count / total_articles
        else:
            overall_sentiment = "neutral"
            sentiment_score = 0.5
        
        return {
            "ticker": ticker,
            "overall_sentiment": overall_sentiment,
            "sentiment_score": round(sentiment_score, 2),
            "confidence": round(sum(confidences) / len(confidences), 2),
            "article_count": total_articles,
            "positive_articles": positive_count,
            "negative_articles": negative_count,
            "neutral_articles": neutral_count,
            "recent_articles": [
                {
                    "title": article.title,
                    "source": article.source,
                    "sentiment": article.sentiment.value,
                    "confidence": article.confidence,
                    "published_at": article.published_at.isoformat()
                }
                for article in ticker_news[-5:]  # Last 5 articles
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_mock_news_for_ticker(self, ticker: str) -> List[NewsArticle]:
        """Generate mock news articles for a ticker."""
        mock_articles = [
            {
                "title": f"{ticker} Shows Strong Performance in Recent Trading",
                "content": f"The {ticker} token has demonstrated strong performance in recent trading sessions, with increased volume and positive price action.",
                "source": "CryptoNews",
                "sentiment": SentimentType.POSITIVE,
                "keywords": [ticker.lower(), "performance", "trading", "volume", "price"]
            },
            {
                "title": f"Technical Analysis Suggests Bullish Outlook for {ticker}",
                "content": f"Technical indicators for {ticker} suggest a bullish outlook with potential for further upside movement.",
                "source": "TechnicalAnalysis",
                "sentiment": SentimentType.POSITIVE,
                "keywords": [ticker.lower(), "technical", "analysis", "bullish", "outlook"]
            }
        ]
        
        articles = []
        for i, news in enumerate(mock_articles):
            article = NewsArticle(
                title=news["title"],
                content=news["content"],
                source=news["source"],
                published_at=datetime.utcnow() - timedelta(hours=random.randint(1, 12)),
                sentiment=news["sentiment"],
                confidence=random.uniform(0.7, 0.9),
                keywords=news["keywords"],
                url=f"https://example.com/news/{ticker.lower()}/{i+1}"
            )
            articles.append(article)
        
        return articles
    
    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        await self._simulate_api_delay()
        return True
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get detailed service status."""
        await self._simulate_api_delay()
        
        return {
            "status": "healthy",
            "model": self.model,
            "api_key": self.api_key[:8] + "..." if self.api_key else "mock",
            "cache_size": {
                "sentiment": len(self.sentiment_cache),
                "responses": len(self.response_cache)
            },
            "news_articles": len(self.news_articles),
            "uptime": random.randint(1000, 10000),  # seconds
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def reset_cache(self):
        """Reset all caches."""
        self.sentiment_cache.clear()
        self.response_cache.clear()
        log.info("Mock LLM service cache reset")
    
    async def simulate_market_news(self, sentiment: str):
        """Simulate market news with specific sentiment."""
        if sentiment == "positive":
            new_article = NewsArticle(
                title="Breaking: Major Positive Development in Crypto Markets",
                content="A significant positive development has occurred in cryptocurrency markets, driving optimism among traders and investors.",
                source="MarketWatch",
                published_at=datetime.utcnow(),
                sentiment=SentimentType.POSITIVE,
                confidence=0.9,
                keywords=["positive", "development", "crypto", "markets", "optimism"],
                url="https://example.com/breaking/positive"
            )
        elif sentiment == "negative":
            new_article = NewsArticle(
                title="Breaking: Market Concerns Rise as Uncertainty Grows",
                content="Growing uncertainty in cryptocurrency markets has led to increased concerns among traders and investors.",
                source="MarketWatch",
                published_at=datetime.utcnow(),
                sentiment=SentimentType.NEGATIVE,
                confidence=0.9,
                keywords=["concerns", "uncertainty", "crypto", "markets", "traders"],
                url="https://example.com/breaking/negative"
            )
        else:
            return
        
        self.news_articles.append(new_article)
        log.info(f"Simulated {sentiment} market news")


# Global mock LLM service instance
_mock_llm_service: Optional[MockLLMService] = None


async def get_mock_llm_service() -> MockLLMService:
    """Get global mock LLM service instance."""
    global _mock_llm_service
    if _mock_llm_service is None:
        _mock_llm_service = MockLLMService()
    return _mock_llm_service
