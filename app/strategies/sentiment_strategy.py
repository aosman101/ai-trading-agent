from __future__ import annotations

import pandas as pd

from app.strategies.base import BaseStrategy


class SentimentStrategy(BaseStrategy):
    name = "sentiment"

    @staticmethod
    def _sentiment_series(frame: pd.DataFrame, sentiment_score: float | pd.Series) -> pd.Series:
        if isinstance(sentiment_score, pd.Series):
            sentiment = sentiment_score.reindex(frame.index).ffill().fillna(0.0)
        else:
            sentiment = pd.Series(float(sentiment_score), index=frame.index)
        return sentiment.astype(float).clip(lower=-1.0, upper=1.0)

    def generate_series(self, frame: pd.DataFrame, sentiment_score: float | pd.Series = 0.0) -> pd.Series:
        sentiment = self._sentiment_series(frame, sentiment_score)
        signal = pd.Series(0.0, index=frame.index, dtype=float)
        long_condition = (sentiment > 0.2) & (frame["close"] > frame["ema_20"]) & (frame["rsi_14"] > 50)
        short_condition = (sentiment < -0.2) & (frame["close"] < frame["ema_20"]) & (frame["rsi_14"] < 50)
        signal[long_condition] = sentiment[long_condition]
        signal[short_condition] = sentiment[short_condition]
        return signal.fillna(0.0)
