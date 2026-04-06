from __future__ import annotations

import pandas as pd

from app.strategies.base import BaseStrategy
from app.strategies.sentiment_strategy import SentimentStrategy


class DummyStrategy(BaseStrategy):
    name = "dummy"

    def __init__(self, value: float):
        self.value = value

    def generate_series(self, frame: pd.DataFrame, sentiment_score: float | pd.Series = 0.0) -> pd.Series:
        return pd.Series(self.value, index=frame.index, dtype=float)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "close": [101.0, 103.0],
            "ema_20": [100.0, 101.0],
            "rsi_14": [55.0, 62.0],
        },
        index=pd.bdate_range("2024-01-02", periods=2),
    )


def test_generate_latest_keeps_fractional_long_signal():
    signal = DummyStrategy(0.6).generate_latest(_frame())
    assert signal.direction == "long"
    assert signal.confidence == 0.6


def test_generate_latest_keeps_fractional_short_signal():
    signal = DummyStrategy(-0.4).generate_latest(_frame())
    assert signal.direction == "short"
    assert signal.confidence == 0.4


def test_sentiment_strategy_supports_series_input():
    frame = _frame()
    sentiment = pd.Series([-0.7, 0.8], index=frame.index)
    series = SentimentStrategy().generate_series(frame, sentiment_score=sentiment)
    assert series.iloc[0] == 0.0
    assert series.iloc[1] > 0.0
