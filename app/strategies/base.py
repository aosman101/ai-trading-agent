from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from app.types import StrategySignal


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_series(self, frame: pd.DataFrame, sentiment_score: float = 0.0) -> pd.Series:
        raise NotImplementedError

    def generate_latest(self, frame: pd.DataFrame, sentiment_score: float = 0.0) -> StrategySignal:
        series = self.generate_series(frame, sentiment_score=sentiment_score)
        latest = int(series.iloc[-1]) if not series.empty else 0
        direction = "flat"
        if latest > 0:
            direction = "long"
        elif latest < 0:
            direction = "short"
        confidence = min(1.0, abs(float(series.iloc[-1])) if not series.empty else 0.0)
        return StrategySignal(
            strategy=self.name,
            symbol=str(frame.iloc[-1]["symbol"]) if "symbol" in frame.columns else "UNKNOWN",
            direction=direction,
            confidence=confidence,
            reason=f"{self.name} latest signal",
        )
