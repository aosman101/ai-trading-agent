from __future__ import annotations

import pandas as pd

from app.strategies.base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    name = "breakout"

    def generate_series(self, frame: pd.DataFrame, sentiment_score: float = 0.0) -> pd.Series:
        prior_high = frame["rolling_high_20"].shift(1)
        prior_low = frame["rolling_low_20"].shift(1)

        long_condition = (frame["close"] > prior_high) & (frame["volume_zscore_20"] > 1.0)
        short_condition = (frame["close"] < prior_low) & (frame["volume_zscore_20"] > 1.0)

        signal = pd.Series(0.0, index=frame.index)
        strength = (frame["volume_zscore_20"].abs() / 3.0).clip(lower=0.0, upper=1.0)
        signal[long_condition] = strength[long_condition].clip(lower=0.2, upper=1.0)
        signal[short_condition] = -strength[short_condition].clip(lower=0.2, upper=1.0)
        return signal.fillna(0.0)
