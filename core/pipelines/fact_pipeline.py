"""Fact pipeline that fuses news, sentiment, and research insights."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from core.config import settings
from core.indicators.cmc_client import get_indicator_snapshot
from core.logging import log
from core.models import FactInsight, NewsMemoryEntry
from core.pipelines.storage import set_fact_insight
from core.redis_client import RedisClient
from core.research import (
    fetch_arxiv_entries,
    fetch_coin_bureau_updates,
    fetch_google_scholar_entries,
    fetch_yahoo_finance_headlines,
)
from core.memory.camel_memory_manager import CamelMemoryManager

try:
    from camel.messages import BaseMessage
except ImportError:  # pragma: no cover
    BaseMessage = None


@dataclass
class SentimentSnapshot:
    weighted_score: float
    total_weight: float
    last_updated: Optional[str]


RESEARCH_CACHE_KEY = "pipeline:fact:research:{ticker}"
RESEARCH_CACHE_VERSION = 3

CMC_REFERENCE_LINKS: List[Tuple[str, str]] = [
    ("CoinMarketCap Fear & Greed Index", "https://coinmarketcap.com/charts/fear-and-greed-index/"),
    ("CoinMarketCap Altcoin Season Index", "https://coinmarketcap.com/charts/altcoin-season-index/"),
    ("CoinMarketCap Market Cycle Indicators", "https://coinmarketcap.com/charts/crypto-market-cycle-indicators/"),
    ("CoinMarketCap Bitcoin Dominance", "https://coinmarketcap.com/charts/bitcoin-dominance/"),
    ("CoinMarketCap CMC20 Index", "https://coinmarketcap.com/charts/cmc20/"),
    ("CoinMarketCap CMC100 Index", "https://coinmarketcap.com/charts/cmc100/"),
]


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
        try:
            self._memory_manager = CamelMemoryManager(agent_id="pipeline_fact")
        except Exception:
            self._memory_manager = None

    async def _client_instance(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def run_for_ticker(self, ticker: str) -> Optional[FactInsight]:
        """Produce and persist a fact insight for a ticker."""
        try:
            news_sentiment = await self._load_sentiment(ticker)
            cmc_snapshot = await self._load_market_indicators()

            sentiment_score, breakdown, combined_weight = self._combine_sentiment_sources(
                ticker, news_sentiment, cmc_snapshot
            )
            if sentiment_score is None:
                log.debug(
                    "Fact pipeline skipped for %s due to missing sentiment and market indicators",
                    ticker,
                )
                return None

            headlines = await self._load_recent_headlines(ticker, limit=3)
            research_refs = await self._load_research(ticker)

            references = self._combine_references(headlines, research_refs)
            if cmc_snapshot:
                references.extend(self._indicator_references())

            anomalies = self._detect_anomalies(sentiment_score, breakdown, cmc_snapshot)

            thesis = self._compose_thesis(ticker, sentiment_score, headlines, research_refs)

            confidence = self._derive_confidence(combined_weight, len(references))

            insight = FactInsight(
                ticker=ticker,
                sentiment_score=sentiment_score,
                confidence=confidence,
                thesis=thesis,
                references=references,
                anomalies=anomalies,
                sentiment_breakdown=breakdown,
                market_indicators=cmc_snapshot or {},
            )
            insight.agentic = True
            insight.ai_explanation = self._build_explanation(sentiment_score, breakdown)

            components = breakdown.get("components", {}) if isinstance(breakdown, dict) else {}
            news_component = components.get("news_sentiment", {}) if isinstance(components, dict) else {}
            regime_component = components.get("market_regime", {}) if isinstance(components, dict) else {}
            log.bind(pipeline="FACT", ticker=ticker).info(
                "Fact pipeline blended sentiment %.3f (news_weight=%.2f, regime_score=%s, references=%d)",
                sentiment_score,
                float(news_component.get("weight", 0.0)) if isinstance(news_component, dict) else 0.0,
                (
                    f"{regime_component.get('score'):.3f}"
                    if isinstance(regime_component, dict) and isinstance(regime_component.get("score"), (int, float))
                    else "n/a"
                ),
                len(references),
            )

            await set_fact_insight(self.redis, insight, ttl_seconds=self.ttl_seconds)
            self._record_memory(insight)
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

    async def _load_market_indicators(self) -> Optional[Dict[str, Any]]:
        if not settings.cmc_api_key:
            return None
        try:
            snapshot = await get_indicator_snapshot()
        except RuntimeError as exc:
            log.debug("CMC indicators unavailable: %s", exc)
            return None
        return snapshot

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

        scholar_entries = await self._fetch_google_scholar_research(ticker)
        if scholar_entries:
            entries.extend(scholar_entries)

        coin_bureau_entries = await self._fetch_coin_bureau_research(ticker)
        if coin_bureau_entries:
            entries.extend(coin_bureau_entries)

        yahoo_entries = await self._fetch_yahoo_finance_research(ticker)
        if yahoo_entries:
            entries.extend(yahoo_entries)

        if entries:
            await self.redis.set_json(
                cache_key,
                {"version": RESEARCH_CACHE_VERSION, "entries": entries},
                expire=self.research_ttl_seconds,
            )

        return entries

    def _record_memory(self, insight: FactInsight) -> None:
        if not self._memory_manager or BaseMessage is None:
            return
        try:
            summary = (
                f"Fact insight for {insight.ticker}: sentiment {insight.sentiment_score:.2f}, "
                f"confidence {insight.confidence:.2f}. Thesis: {insight.thesis[:240]}"
            )
            message = BaseMessage.make_assistant_message(
                role_name="FactPipeline",
                content=summary,
            )
            self._memory_manager.write_record(message)
        except Exception as exc:
            log.debug("Unable to record fact pipeline memory: %s", exc)

    async def _fetch_arxiv_research(self, ticker: str) -> List[Dict[str, Any]]:
        try:
            client = await self._client_instance()
            query = f'all:"{ticker}" AND (ti:crypto OR ti:blockchain OR abs:"{ticker}")'
            results = await fetch_arxiv_entries(client, query, limit=5)
        except Exception as exc:  # pragma: no cover
            log.debug("Unable to fetch arXiv research for %s: %s", ticker, exc)
            return []

        ticker_upper = ticker.upper()
        filtered = []
        for entry in results:
            content = f"{entry.get('title', '')} {entry.get('summary', '')}".upper()
            if ticker_upper in content:
                filtered.append(entry)
        return filtered or results

    async def _fetch_google_scholar_research(self, ticker: str) -> List[Dict[str, Any]]:
        try:
            client = await self._client_instance()
            query = f'"{ticker}" cryptocurrency OR "{ticker}" blockchain adoption'
            results = await fetch_google_scholar_entries(client, query, limit=5)
        except Exception as exc:  # pragma: no cover
            log.debug("Unable to fetch Google Scholar research for %s: %s", ticker, exc)
            return []

        ticker_upper = ticker.upper()
        filtered: List[Dict[str, Any]] = []
        for entry in results:
            content = f"{entry.get('title', '')} {entry.get('summary', '')}".upper()
            if ticker_upper in content:
                filtered.append(entry)
        return filtered or results[:3]

    async def _fetch_coin_bureau_research(self, ticker: str) -> List[Dict[str, Any]]:
        try:
            client = await self._client_instance()
            updates = await fetch_coin_bureau_updates(client, limit=12)
        except Exception as exc:  # pragma: no cover
            log.debug("Unable to fetch Coin Bureau updates for %s: %s", ticker, exc)
            return []

        ticker_upper = ticker.upper()
        entries: List[Dict[str, Any]] = []
        for item in updates:
            title = item.get("title", "")
            if ticker_upper not in title.upper():
                continue
            weight = settings.news_source_weights.get("coin_bureau", 0.1)
            entries.append(
                {
                    "title": title,
                    "url": item.get("url"),
                    "source": "Coin Bureau (Video)",
                    "summary": "Coin Bureau coverage relevant to the asset.",
                    "published_at": item.get("published_at"),
                    "source_key": "coin_bureau",
                    "source_weight": weight,
                }
            )
        return entries[:3]

    async def _fetch_yahoo_finance_research(self, ticker: str) -> List[Dict[str, Any]]:
        try:
            client = await self._client_instance()
            results = await fetch_yahoo_finance_headlines(
                client, tickers=[f"{ticker}-USD", ticker], limit=6
            )
        except Exception as exc:  # pragma: no cover
            log.debug("Unable to fetch Yahoo Finance headlines for %s: %s", ticker, exc)
            return []

        ticker_upper = ticker.upper()
        entries: List[Dict[str, Any]] = []
        for item in results:
            title = item.get("title", "")
            if ticker_upper not in title.upper():
                continue
            weight = settings.news_source_weights.get("yahoo_finance", 0.1)
            entries.append(
                {
                    "title": title,
                    "url": item.get("url"),
                    "source": "Yahoo Finance",
                    "summary": item.get("summary"),
                    "published_at": item.get("published_at"),
                    "source_key": "yahoo_finance",
                    "source_weight": weight,
                }
            )
        return entries[:3]

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

    def _combine_sentiment_sources(
        self,
        ticker: str,
        news_sentiment: Optional[SentimentSnapshot],
        cmc_snapshot: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[float], Dict[str, Any], float]:
        components: Dict[str, Any] = {}
        aggregate_weight = 0.0
        weighted_sum = 0.0

        if news_sentiment and news_sentiment.total_weight > 0:
            news_weight = max(news_sentiment.total_weight, 0.5)
            components["news_sentiment"] = {
                "score": news_sentiment.weighted_score,
                "weight": news_weight,
                "last_updated": news_sentiment.last_updated,
            }
            aggregate_weight += news_weight
            weighted_sum += news_sentiment.weighted_score * news_weight

        indicator_weight = 0.0
        if cmc_snapshot:
            indicator_score, indicator_details = self._score_indicators(cmc_snapshot)
            if indicator_score is not None and indicator_details.get("weight", 0.0) > 0:
                indicator_weight = indicator_details["weight"]
                aggregate_weight += indicator_weight
                weighted_sum += indicator_score * indicator_weight
                components["market_regime"] = {
                    "score": indicator_score,
                    "weight": indicator_weight,
                    "normalized": indicator_details.get("normalized", {}),
                    "source": "coinmarketcap",
                }

        if aggregate_weight == 0:
            breakdown = {"aggregate_score": None, "total_weight": 0.0, "components": components}
            return None, breakdown, 0.0

        aggregate_score = max(min(weighted_sum / aggregate_weight, 1.0), -1.0)
        breakdown = {
            "aggregate_score": aggregate_score,
            "total_weight": aggregate_weight,
            "components": components,
        }
        return aggregate_score, breakdown, aggregate_weight

    def _score_indicators(self, snapshot: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Optional[float]]] = {}
        weight_total = 0.0
        score_total = 0.0

        fear_greed_entry = snapshot.get("fear_greed_index")
        fear_greed_value = self._extract_indicator_value(fear_greed_entry, "value")
        if fear_greed_value is not None:
            fg_score = max(min((fear_greed_value - 50.0) / 50.0, 1.0), -1.0)
            classification = None
            if isinstance(fear_greed_entry, dict):
                classification = fear_greed_entry.get("value_classification")
            normalized["fear_greed_index"] = {
                "value": fear_greed_value,
                "score": fg_score,
                "classification": classification,
            }
            weight_total += 1.2
            score_total += fg_score * 1.2

        altcoin_entry = snapshot.get("altcoin_season_index")
        altcoin_value = self._extract_indicator_value(altcoin_entry, "value")
        if altcoin_value is not None:
            alt_score = max(min((altcoin_value - 50.0) / 50.0, 1.0), -1.0)
            classification = None
            if isinstance(altcoin_entry, dict):
                classification = altcoin_entry.get("value_classification")
            normalized["altcoin_season_index"] = {
                "value": altcoin_value,
                "score": alt_score,
                "classification": classification,
            }
            weight_total += 0.8
            score_total += alt_score * 0.8

        dominance_entry = snapshot.get("bitcoin_dominance")
        dominance_value = self._extract_indicator_value(dominance_entry, "value")
        if dominance_value is not None:
            dom_score = max(min((50.0 - dominance_value) / 50.0, 1.0), -1.0)
            normalized["bitcoin_dominance"] = {
                "value": dominance_value,
                "score": dom_score,
            }
            weight_total += 0.6
            score_total += dom_score * 0.6

        cmc20_change = self._extract_indicator_change(snapshot.get("cmc20_index"))
        if cmc20_change is not None:
            cmc20_score = max(min(cmc20_change / 10.0, 1.0), -1.0)
            normalized["cmc20_index"] = {
                "change_percent": cmc20_change,
                "score": cmc20_score,
            }
            weight_total += 0.5
            score_total += cmc20_score * 0.5

        cmc100_change = self._extract_indicator_change(snapshot.get("cmc100_index"))
        if cmc100_change is not None:
            cmc100_score = max(min(cmc100_change / 10.0, 1.0), -1.0)
            normalized["cmc100_index"] = {
                "change_percent": cmc100_change,
                "score": cmc100_score,
            }
            weight_total += 0.5
            score_total += cmc100_score * 0.5

        market_cycle_entry = snapshot.get("market_cycle_indicator")
        market_cycle_value = self._extract_indicator_value(market_cycle_entry, "score")
        if market_cycle_value is None:
            market_cycle_value = self._extract_indicator_value(market_cycle_entry, "value")
        if market_cycle_value is not None:
            cycle_score = max(min(market_cycle_value, 1.0), -1.0)
            normalized["market_cycle_indicator"] = {
                "score": cycle_score,
                "raw": market_cycle_value,
            }
            weight_total += 0.4
            score_total += cycle_score * 0.4

        if weight_total == 0:
            return None, {"weight": 0.0, "normalized": normalized}

        aggregate = score_total / weight_total
        return aggregate, {"weight": weight_total, "normalized": normalized}

    @staticmethod
    def _extract_indicator_value(entry: Any, key: str) -> Optional[float]:
        if entry is None:
            return None
        if isinstance(entry, dict):
            candidate = entry.get(key)
        else:
            candidate = entry
        if candidate is None:
            return None
        try:
            return float(candidate)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_indicator_change(entry: Any) -> Optional[float]:
        if not isinstance(entry, dict):
            return None
        for key in ("percent_change_24h", "percent_change_7d", "percent_change_30d"):
            value = entry.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    def _indicator_references(self) -> List[Dict[str, Any]]:
        references: List[Dict[str, Any]] = []
        for title, url in CMC_REFERENCE_LINKS:
            references.append(
                {
                    "title": title,
                    "url": url,
                    "source": "CoinMarketCap",
                    "category": "market_indicator",
                }
            )
        return references

    def _detect_anomalies(
        self,
        sentiment: float,
        breakdown: Dict[str, Any],
        cmc_snapshot: Optional[Dict[str, Any]],
    ) -> List[str]:
        anomalies: List[str] = []
        if sentiment < -0.35:
            anomalies.append("Sustained negative blended sentiment observed; monitor for capitulation events.")
        elif sentiment > 0.35:
            anomalies.append("Elevated positive blended sentiment detected; watch for exuberant market conditions.")

        market_regime = breakdown.get("components", {}).get("market_regime", {})
        normalized = market_regime.get("normalized", {}) if isinstance(market_regime, dict) else {}

        fear_greed = normalized.get("fear_greed_index", {})
        fg_value = fear_greed.get("value") if isinstance(fear_greed, dict) else None
        if fg_value is not None:
            try:
                fg_float = float(fg_value)
            except (TypeError, ValueError):
                fg_float = None
            if fg_float is not None:
                if fg_float <= 20:
                    anomalies.append("CMC Fear & Greed indicates extreme fear; sentiment headwinds likely.")
                elif fg_float >= 80:
                    anomalies.append("CMC Fear & Greed shows extreme greed; risk of corrections increases.")

        dominance = normalized.get("bitcoin_dominance", {})
        dom_value = dominance.get("value") if isinstance(dominance, dict) else None
        if dom_value is not None:
            try:
                dom_float = float(dom_value)
            except (TypeError, ValueError):
                dom_float = None
            if dom_float and dom_float >= 60:
                anomalies.append("Bitcoin dominance >60%; altcoin liquidity may be weakening.")

        return anomalies

    def _build_explanation(self, aggregate_score: float, breakdown: Dict[str, Any]) -> str:
        components = breakdown.get("components", {})
        parts: List[str] = [f"Blended sentiment score {aggregate_score:.2f}."]

        news_component = components.get("news_sentiment")
        if isinstance(news_component, dict):
            parts.append(
                f"News sentiment contribution {news_component.get('score', 0.0):.2f} (weight {news_component.get('weight', 0.0):.2f})."
            )

        regime_component = components.get("market_regime")
        if isinstance(regime_component, dict):
            fg = regime_component.get("normalized", {}).get("fear_greed_index", {})
            fg_value = fg.get("value")
            if fg_value is not None:
                parts.append(f"CMC Fear & Greed currently {fg_value}.")
            alt = regime_component.get("normalized", {}).get("altcoin_season_index", {})
            alt_value = alt.get("value") if isinstance(alt, dict) else None
            if alt_value is not None:
                parts.append(f"Altcoin Season index at {alt_value}.")

        return " ".join(parts)

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

