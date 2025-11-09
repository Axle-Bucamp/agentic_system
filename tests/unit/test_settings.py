import os

import pytest

from core.config import Settings


def test_deep_search_sources_default(monkeypatch):
    monkeypatch.delenv("DEEP_SEARCH_SOURCES", raising=False)

    settings = Settings()
    assert settings.deep_search_sources == ["coindesk", "cointelegraph", "decrypt"]


def test_news_source_weights_default(monkeypatch):
    monkeypatch.delenv("NEWS_SOURCE_WEIGHTS", raising=False)

    settings = Settings()
    assert settings.news_source_weights["yahoo_finance"] == pytest.approx(0.35)
    assert settings.news_source_weights["coin_bureau"] == pytest.approx(0.25)


def test_news_source_weights_env_override(monkeypatch):
    monkeypatch.setenv(
        "NEWS_SOURCE_WEIGHTS",
        '{"yahoo_finance": 0.5, "coin_bureau": 0.3, "arxiv": 0.1, "google_scholar": 0.1}',
    )
    settings = Settings()
    assert settings.news_source_weights["yahoo_finance"] == pytest.approx(0.5)
    assert sum(settings.news_source_weights.values()) == pytest.approx(1.0)
