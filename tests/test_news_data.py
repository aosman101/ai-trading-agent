from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pandas as pd

os.environ.setdefault("TRADING_MODE", "paper")

for module_name in ("feedparser", "httpx"):
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()

from app.data.news_data import NewsDataService


def test_sentiment_time_series_aligns_news_by_date():
    service = NewsDataService()
    service.fetch_alpha_vantage_news = MagicMock(
        return_value=[
            {
                "published_at": "20240103T120000",
                "sentiment_score": "0.6",
            },
            {
                "published_at": "20240104T130000",
                "sentiment_score": "-0.2",
            },
        ]
    )

    index = pd.bdate_range("2024-01-02", periods=4)
    series = service.sentiment_time_series("AAPL", index)

    assert list(series.index) == list(index)
    assert series.iloc[0] == 0.0
    assert series.iloc[1] == 0.6
    assert series.iloc[2] == -0.2


def test_sentiment_time_series_uses_latest_fallback():
    service = NewsDataService()
    service.fetch_alpha_vantage_news = MagicMock(return_value=[])

    index = pd.bdate_range("2024-01-02", periods=3)
    series = service.sentiment_time_series("AAPL", index, fallback_latest=0.75)

    assert series.iloc[0] == 0.0
    assert series.iloc[-1] == 0.75
