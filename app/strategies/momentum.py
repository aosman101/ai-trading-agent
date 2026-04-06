from __future__ import annotations

import pandas as pd

from app.strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    name = "momentum"

    def generate_series(self, frame: pd.DataFrame, sentiment_score: float | pd.Series = 0.0) -> pd.Series:
        long_condition = (
            (frame["ema_20"] > frame["ema_50"])
            & (frame["macd"] > frame["macd_signal"])
            & (frame["rsi_14"] > 55)
        )
        short_condition = (
            (frame["ema_20"] < frame["ema_50"])
            & (frame["macd"] < frame["macd_signal"])
            & (frame["rsi_14"] < 45)
        )
        signal = pd.Series(0.0, index=frame.index)
        strength = ((frame["ema_20"] / frame["ema_50"]) - 1.0).abs().clip(lower=0.0, upper=0.05) / 0.05
        signal[long_condition] = strength[long_condition].clip(lower=0.2, upper=1.0)
        signal[short_condition] = -strength[short_condition].clip(lower=0.2, upper=1.0)
        return signal.fillna(0.0)
