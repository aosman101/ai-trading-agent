from __future__ import annotations

import pandas as pd

from app.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    def generate_series(self, frame: pd.DataFrame, sentiment_score: float | pd.Series = 0.0) -> pd.Series:
        long_condition = (
            (frame["price_zscore_20"] < -2.0)
            & (frame["close"] < frame["bb_lower"])
            & (frame["rsi_14"] < 35)
        )
        short_condition = (
            (frame["price_zscore_20"] > 2.0)
            & (frame["close"] > frame["bb_upper"])
            & (frame["rsi_14"] > 65)
        )
        signal = pd.Series(0.0, index=frame.index)
        z_strength = (frame["price_zscore_20"].abs() / 3.0).clip(lower=0.0, upper=1.0)
        signal[long_condition] = z_strength[long_condition].clip(lower=0.3, upper=1.0)
        signal[short_condition] = -z_strength[short_condition].clip(lower=0.3, upper=1.0)
        return signal.fillna(0.0)
