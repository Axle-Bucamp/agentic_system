"""Fact pipeline that fuses news, sentiment, and research insights."""
from __future__ import annotations

import asyncio
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urlparse
from uuid import uuid4

import httpx

from core.config import settings
from core.logging import log
from core.models import FactInsight, NewsMemoryEntry
from core.pipelines.storage import set_fact_insight
from core.redis_client import RedisClient


@dataclass
class SentimentSnapshot:
    weighted_score: float
    total_weight: float
    last_updated: Optional[str]


RESEARCH_CACHE_KEY = "pipeline:fact:research:{ticker}"
RESEARCH_CACHE_VERSION = 2


class FactPipeline:
    """Generate long-horizon factual insights using news, research, and sentiment."""

    def __init__(
        self,
        redis: RedisClient,
        ttl_seconds: int = 3600,
        research_ttl_seconds: int = 86400,
    ):
        self.redis = redis
        self.ttl_seconds = ttl_seconds
        self.research_ttl_seconds = research_ttl_seconds
        self._client: Optional[httpx.AsyncClient] = None

    async def _client_instance(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def run_for_ticker(self, ticker: str) -> Optional[FactInsight]:
        """Produce and persist a fact insight for a ticker."""
        try:
            sentiment = await self._load_sentiment(ticker)
            if not sentiment:
                log.debug("Fact pipeline skipped for %s due to missing sentiment", ticker)
                return None

            headlines = await self._load_recent_headlines(ticker, limit=3)
            research_refs = await self._load_research(ticker)

            references = self._combine_references(headlines, research_refs)

            anomalies = []
            if sentiment.weighted_score < -0.35:
                anomalies.append("Sustained negative sentiment observed across recent coverage.")
            elif sentiment.weighted_score > 0.35:
                anomalies.append("Elevated positive sentiment detected; monitor for exuberance.")

            thesis = self._compose_thesis(ticker, sentiment.weighted_score, headlines, research_refs)

            confidence = self._derive_confidence(sentiment.total_weight, len(references))

            insight = FactInsight(
                ticker=ticker,
                sentiment_score=sentiment.weighted_score,
                confidence=confidence,
                thesis=thesis,
                references=references,
                anomalies=anomalies,
            )

            await set_fact_insight(self.redis, insight, ttl_seconds=self.ttl_seconds)
            return insight
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("Fact pipeline failed for %s: %s", ticker, exc)
            return None

    async def _load_sentiment(self, ticker: str) -> Optional[SentimentSnapshot]:
        payload = await self.redis.get_json(f"memory:news:weighted:{ticker}")
        if not payload:
            return None
        try:
            return SentimentSnapshot(
                weighted_score=float(payload.get("weighted_score", 0.0)),
                total_weight=float(payload.get("total_weight", 0.0)),
                last_updated=payload.get("last_updated"),
            )
        except Exception:
            return None

    async def _load_recent_headlines(self, ticker: str, limit: int = 3) -> List[Dict[str, Any]]:
        headlines: List[Dict[str, Any]] = []
        news_raw = await self.redis.lrange(f"memory:news:{ticker}", -limit, -1)
        for raw in reversed(news_raw):
            try:
                entry = NewsMemoryEntry.parse_raw(raw)
            except Exception:
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                entry = NewsMemoryEntry(**data)

            metadata = entry.metadata.get("raw") if isinstance(entry.metadata, dict) else {}
            headlines.append(
                {
                    "title": metadata.get("title", entry.summary[:120]),
                    "url": metadata.get("url"),
                    "source": (metadata.get("source") or (entry.sources[0] if entry.sources else "Unknown")),
                    "sentiment": entry.sentiment_score,
                    "timestamp": entry.timestamp.isoformat(),
                }
            )
        return headlines

    async def _load_research(self, ticker: str) -> List[Dict[str, Any]]:
        cache_key = RESEARCH_CACHE_KEY.format(ticker=ticker)
        cached = await self.redis.get_json(cache_key)
        if cached and cached.get("version") == RESEARCH_CACHE_VERSION:
            return cached.get("entries", [])

        entries: List[Dict[str, Any]] = []
        arxiv_entries = await self._fetch_arxiv_research(ticker)
        if arxiv_entries:
            entries.extend(arxiv_entries)

        deep_research_entries = await self._fetch_deep_research(ticker)
        if deep_research_entries:
            entries.extend(deep_research_entries)

        if entries:
            await self.redis.set_json(
                cache_key,
                {"version": RESEARCH_CACHE_VERSION, "entries": entries},
                expire=self.research_ttl_seconds,
            )

        return entries

    async def _fetch_arxiv_research(self, ticker: str) -> List[Dict[str, Any]]:
        try:
            client = await self._client_instance()
            query = quote_plus(f'all:"{ticker}" AND (ti:crypto OR ti:blockchain)')
            url = f"https://export.arxiv.org/api/query?search_query={query}&sortBy=submittedDate&max_results=3"
            response = await client.get(url)
            response.raise_for_status()
            return self._parse_arxiv_feed(response.text)
        except Exception as exc:  # pragma: no cover - external dependency best-effort
            log.debug("Unable to fetch arXiv research for %s: %s", ticker, exc)
            return []

    async def _fetch_deep_research(self, ticker: str) -> List[Dict[str, Any]]:
        if not settings.deep_research_mcp_url:
            return []

        try:
            client = await self._client_instance()
            payload = {
                "jsonrpc": "2.0",
                "id": str(uuid4()),
                "method": "tools/call",
                "params": {
                    "name": "deep-research",
                    "arguments": {
                        "query": f"Latest developments affecting {ticker} cryptocurrency and blockchain adoption",
                        "depth": max(1, min(5, settings.deep_research_depth)),
                        "breadth": max(1, min(5, settings.deep_research_breadth)),
                    },
                },
            }
            if settings.deep_research_model:
                payload["params"]["arguments"]["model"] = settings.deep_research_model
            if settings.deep_research_source_preferences:
                payload["params"]["arguments"]["sourcePreferences"] = settings.deep_research_source_preferences

            timeout = settings.deep_research_timeout_seconds or 120
            response = await client.post(
                settings.deep_research_mcp_url,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - external dependency best-effort
            log.debug("Deep research MCP call failed for %s: %s", ticker, exc)
            return []

        try:
            data = response.json()
        except ValueError:
            log.debug("Deep research MCP response was not JSON for %s", ticker)
            return []

        result = data.get("result") or {}
        content = result.get("content") or []
        summary_text = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                summary_text = item.get("text", "")
                break
        metadata = result.get("metadata") or {}
        visited = metadata.get("visitedUrls") or []
        learnings = metadata.get("learnings") or []

        if not summary_text and not visited:
            return []

        entries: List[Dict[str, Any]] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        first_url = next((u for u in visited if isinstance(u, str) and u.startswith("http")), None)

        if summary_text:
            entries.append(
                {
                    "title": f"Deep research findings for {ticker}",
                    "url": first_url,
                    "source": "DeepResearch",
                    "summary": self._truncate(summary_text, 600),
                    "published_at": timestamp,
                }
            )

        for idx, url in enumerate(visited[:5]):
            if not isinstance(url, str):
                continue
            domain = urlparse(url).netloc if url.startswith("http") else ""
            title = None
            if isinstance(learnings, list) and idx < len(learnings):
                learning = learnings[idx]
                if isinstance(learning, str):
                    title = learning[:140]
            entries.append(
                {
                    "title": title or f"Research source #{idx + 1}",
                    "url": url,
                    "source": domain or "DeepResearch",
                    "summary": None,
                    "published_at": timestamp,
                }
            )

        return entries

    def _parse_arxiv_feed(self, feed_text: str) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(feed_text)
        except ET.ParseError:
            return entries

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", default="", namespaces=ns).strip()
            summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
            link_el = entry.find("atom:link", ns)
            link = link_el.attrib.get("href") if link_el is not None else None
            published = entry.findtext("atom:published", default="", namespaces=ns)

            entries.append(
                {
                    "title": title,
                    "url": link,
                    "source": "arXiv",
                    "summary": summary[:280],
                    "published_at": published,
                }
            )
        return entries

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        truncated = text[:limit].rsplit(" ", 1)[0]
        return truncated + "…"

    def _derive_confidence(self, weight: float, reference_count: int) -> float:
        base = min(max(weight / 5.0, 0.1), 1.0)  # up to 5 aggregated weights for max
        bonus = min(reference_count / 5.0, 0.3)
        return round(min(base + bonus, 1.0), 3)

    def _compose_thesis(
        self,
        ticker: str,
        sentiment: float,
        headlines: List[Dict[str, Any]],
        research_refs: List[Dict[str, Any]],
    ) -> str:
        tone = "neutral"
        if sentiment > 0.25:
            tone = "bullish"
        elif sentiment < -0.25:
            tone = "bearish"

        headline_titles = ", ".join(h["title"] for h in headlines[:2]) if headlines else "no major coverage"
        research_title = research_refs[0]["title"] if research_refs else "limited new academic commentary"

        return (
            f"{ticker} exhibits a {tone} narrative driven by recent coverage ({headline_titles}). "
            f"Latest research spotlight: {research_title}. "
            "Sentiment-derived thesis suggests aligning position sizing with prevailing tone while respecting risk bounds."
        )

    def _combine_references(
        self,
        headlines: List[Dict[str, Any]],
        research_refs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        references: List[Dict[str, Any]] = []
        references.extend(headlines)
        references.extend(research_refs)
        return references

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

