from __future__ import annotations

import pandas as pd

from app.strategies.base import BaseStrategy


class SentimentStrategy(BaseStrategy):
    name = "sentiment"

    def generate_series(self, frame: pd.DataFrame, sentiment_score: float = 0.0) -> pd.Series:
        signal = pd.Series(0.0, index=frame.index)
        if sentiment_score > 0.2:
            long_condition = (frame["close"] > frame["ema_20"]) & (frame["rsi_14"] > 50)
            signal[long_condition] = min(1.0, sentiment_score)
        elif sentiment_score < -0.2:
            short_condition = (frame["close"] < frame["ema_20"]) & (frame["rsi_14"] < 50)
            signal[short_condition] = -min(1.0, abs(sentiment_score))
        return signal.fillna(0.0)
