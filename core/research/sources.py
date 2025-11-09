"""
Shared research/news source fetchers.

These helpers are designed to be resilient best-effort integrations that degrade gracefully
when external services throttle or fail. All functions accept an httpx.AsyncClient to reuse
connection pools across agents.
"""
from __future__ import annotations

import asyncio
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote_plus

import httpx

from core.config import settings
from core.logging import log

USER_AGENT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    )
}


async def fetch_yahoo_finance_headlines(
    client: httpx.AsyncClient,
    tickers: Optional[Iterable[str]] = None,
    limit: int = 12,
) -> List[Dict[str, str]]:
    """Fetch recent crypto-related Yahoo Finance headlines via RSS."""
    symbols = list(tickers or ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"])
    joined = ",".join(symbols)
    url = (
        "https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={quote_plus(joined)}&region=US&lang=en-US"
    )
    try:
        response = await client.get(url, headers=USER_AGENT_HEADERS, timeout=20.0)
        response.raise_for_status()
    except Exception as exc:
        log.debug("Yahoo Finance feed fetch failed: %s", exc)
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        log.debug("Yahoo Finance feed parse error: %s", exc)
        return []

    items: List[Dict[str, str]] = []
    for item in root.findall("./channel/item")[:limit]:
        title = _text(item.find("title"))
        link = _text(item.find("link"))
        pub_date = _text(item.find("pubDate"))
        description = _text(item.find("description"))
        items.append(
            {
                "title": title,
                "url": link,
                "source": "Yahoo Finance",
                "published_at": pub_date,
                "summary": description,
                "source_key": "yahoo_finance",
                "source_weight": settings.news_source_weights.get("yahoo_finance", 0.1),
            }
        )
    return items


async def fetch_coin_bureau_updates(
    client: httpx.AsyncClient,
    feed_url: str = "https://www.youtube.com/feeds/videos.xml?channel_id=UCqK_GSMbpiV8spgD3ZGloSw",
    limit: int = 10,
) -> List[Dict[str, str]]:
    """Fetch Coin Bureau updates from the YouTube RSS feed."""
    try:
        response = await client.get(feed_url, headers=USER_AGENT_HEADERS, timeout=20.0)
        response.raise_for_status()
    except Exception as exc:
        log.debug("Coin Bureau feed fetch failed: %s", exc)
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        log.debug("Coin Bureau feed parse error: %s", exc)
        return []

    ns = {"yt": "http://www.youtube.com/xml/schemas/2015", "atom": "http://www.w3.org/2005/Atom"}
    items: List[Dict[str, str]] = []
    for entry in root.findall("atom:entry", ns)[:limit]:
        title = _text(entry.find("atom:title", ns))
        link_el = entry.find("atom:link", ns)
        link = link_el.attrib.get("href") if link_el is not None else ""
        published = _text(entry.find("atom:published", ns))
        video_id = _text(entry.find("yt:videoId", ns))
        items.append(
            {
                "title": title,
                "url": link,
                "source": "Coin Bureau",
                "published_at": published,
                "summary": "",
                "video_id": video_id,
                "source_key": "coin_bureau",
                "source_weight": settings.news_source_weights.get("coin_bureau", 0.1),
            }
        )
    return items


async def fetch_arxiv_entries(
    client: httpx.AsyncClient,
    query: str,
    limit: int = 5,
) -> List[Dict[str, str]]:
    """Fetch arXiv entries for a query."""
    encoded = quote_plus(query)
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query={encoded}&sortBy=submittedDate&max_results={limit}"
    )
    try:
        response = await client.get(url, timeout=20.0, headers=USER_AGENT_HEADERS)
        response.raise_for_status()
    except Exception as exc:
        log.debug("arXiv feed fetch failed: %s", exc)
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        log.debug("arXiv feed parse error: %s", exc)
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items: List[Dict[str, str]] = []
    for entry in root.findall("atom:entry", ns)[:limit]:
        title = _text(entry.find("atom:title", ns)).strip()
        summary = _text(entry.find("atom:summary", ns)).strip()
        link_el = entry.find("atom:link", ns)
        link = link_el.attrib.get("href") if link_el is not None else ""
        published = _text(entry.find("atom:published", ns))
        items.append(
            {
                "title": title,
                "url": link,
                "source": "arXiv",
                "published_at": published,
                "summary": summary[:400],
                "source_key": "arxiv",
                "source_weight": settings.news_source_weights.get("arxiv", 0.1),
            }
        )
    return items


async def fetch_google_scholar_entries(
    client: httpx.AsyncClient,
    query: str,
    limit: int = 5,
) -> List[Dict[str, str]]:
    """Fetch Google Scholar results via the jina.ai readability proxy."""
    encoded = quote_plus(query)
    url = f"https://r.jina.ai/https://scholar.google.com/scholar?hl=en&q={encoded}"
    try:
        response = await client.get(url, headers=USER_AGENT_HEADERS, timeout=20.0)
        response.raise_for_status()
    except Exception as exc:
        log.debug("Google Scholar proxy fetch failed: %s", exc)
        return []

    return _parse_google_scholar_html(response.text, limit)


def _parse_google_scholar_html(html_text: str, limit: int) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    if not html_text:
        log.debug("Google Scholar proxy returned empty body")
        return entries

    title_pattern = re.compile(
        r'<h3[^>]*class="gs_rt"[^>]*>.*?(?:<a href="(?P<url>[^"]+)".*?>)?(?P<title>.*?)</a>?.*?</h3>',
        re.DOTALL | re.IGNORECASE,
    )
    snippet_pattern = re.compile(r'<div class="gs_rs">(.*?)</div>', re.DOTALL | re.IGNORECASE)

    titles = list(title_pattern.finditer(html_text))
    snippets = list(snippet_pattern.finditer(html_text))

    for idx, match in enumerate(titles[:limit]):
        url = html.unescape(match.group("url") or "")
        raw_title = match.group("title") or ""
        title = _strip_tags(raw_title).strip()
        snippet = ""
        if idx < len(snippets):
            snippet = _strip_tags(snippets[idx].group(1)).strip()

        if not title:
            log.debug("Google Scholar entry missing title; skipping.")
            continue

        entries.append(
            {
                "title": title,
                "url": url,
                "source": "Google Scholar",
                "published_at": datetime.utcnow().isoformat(),
                "summary": snippet[:400],
                "source_key": "google_scholar",
                "source_weight": settings.news_source_weights.get("google_scholar", 0.1),
            }
        )
    return entries


def _strip_tags(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", text)
    return html.unescape(cleaned)


def _text(element: Optional[ET.Element]) -> str:
    return element.text if element is not None and element.text else ""


