import os

import pytest

from core.config import Settings


def test_deep_search_sources_default(monkeypatch):
    monkeypatch.delenv("DEEP_SEARCH_SOURCES", raising=False)

    settings = Settings()
    assert settings.deep_search_sources == ["coindesk", "cointelegraph", "decrypt"]

