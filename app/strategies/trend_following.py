from __future__ import annotations

import pandas as pd

from app.strategies.base import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    name = "trend_following"

    def generate_series(self, frame: pd.DataFrame, sentiment_score: float | pd.Series = 0.0) -> pd.Series:
        long_condition = (
            (frame["ema_50"] > frame["ema_200"])
            & (frame["close"] > frame["ema_50"])
            & (frame["trend_strength"] > 0.02)
        )
        short_condition = (
            (frame["ema_50"] < frame["ema_200"])
            & (frame["close"] < frame["ema_50"])
            & (frame["trend_strength"] < -0.02)
        )
        signal = pd.Series(0.0, index=frame.index)
        strength = (frame["trend_strength"].abs() / 0.10).clip(lower=0.0, upper=1.0)
        signal[long_condition] = strength[long_condition].clip(lower=0.2, upper=1.0)
        signal[short_condition] = -strength[short_condition].clip(lower=0.2, upper=1.0)
        return signal.fillna(0.0)
