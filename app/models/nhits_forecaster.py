from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from app.config import get_settings
from app.data.market_data import MarketDataService
from app.types import ModelSignal
from app.utils.logging import get_logger
from app.utils.safe_model_io import load_model, save_model

logger = get_logger(__name__)


class NHITSForecaster:
    def __init__(self) -> None:
        try:
            from neuralforecast import NeuralForecast  # type: ignore
            from neuralforecast.models import NHITS  # type: ignore
        except Exception as exc:
            raise ImportError("neuralforecast is required for NHITSForecaster") from exc

        self.settings = get_settings()
        self._NeuralForecast = NeuralForecast
        self._NHITS = NHITS
        self.model = None
        self.freq = "D"

    def fit(self, frame: pd.DataFrame) -> None:
        train_df = MarketDataService.to_neuralforecast_frame(frame)
        base_model = self._NHITS(
            h=self.settings.forecast_horizon,
            input_size=self.settings.n_hits_input_size,
            max_steps=self.settings.n_hits_max_steps,
            learning_rate=1e-3,
            scaler_type="robust",
            random_seed=42,
        )
        self.model = self._NeuralForecast(models=[base_model], freq=self.freq)
        self.model.fit(train_df)

    def predict_all(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("NHITS model is not trained")
        forecasts = None
        if frame is not None:
            latest_frame = MarketDataService.to_neuralforecast_frame(frame)
            try:
                forecasts = self.model.predict(df=latest_frame)
            except TypeError:
                try:
                    forecasts = self.model.predict(latest_frame)
                except Exception as exc:
                    logger.warning("NHITS predict(latest_frame) failed, falling back to stored context: %s", exc)
            except Exception as exc:
                logger.warning("NHITS predict(df=...) failed, falling back to stored context: %s", exc)
        if forecasts is None:
            forecasts = self.model.predict()
        return forecasts.sort_values(["unique_id", "ds"])

    def predict_latest(self, frame: pd.DataFrame, symbol: str) -> ModelSignal:
        forecasts = self.predict_all(frame=frame)
        symbol_forecasts = forecasts[forecasts["unique_id"] == symbol].copy()
        if symbol_forecasts.empty:
            raise ValueError(f"No NHITS forecast available for {symbol}")
        value_columns = [column for column in symbol_forecasts.columns if column not in {"unique_id", "ds"}]
        point_column = value_columns[0]
        predicted_prices = symbol_forecasts[point_column].tolist()
        current_price = float(
            MarketDataService.to_neuralforecast_frame(frame)
            .query("unique_id == @symbol")
            .sort_values("ds")
            .iloc[-1]["y"]
        )
        short_return = predicted_prices[0] / current_price - 1.0
        medium_return = predicted_prices[min(len(predicted_prices) - 1, max(1, len(predicted_prices) // 2))] / current_price - 1.0
        long_return = predicted_prices[-1] / current_price - 1.0
        expected_return = (0.5 * short_return) + (0.3 * medium_return) + (0.2 * long_return)

        direction = "flat"
        if expected_return > 0.002:
            direction = "long"
        elif expected_return < -0.002:
            direction = "short"

        return ModelSignal(
            name="nhits",
            symbol=symbol,
            direction=direction,
            score=float(expected_return),
            confidence=min(1.0, abs(expected_return) / 0.03),
            metadata={
                "short_return": float(short_return),
                "medium_return": float(medium_return),
                "long_return": float(long_return),
                "current_price": current_price,
            },
        )

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained NHITS model")
        save_model(self.model, path)

    def load(self, path: str | Path) -> None:
        self.model = load_model(path)
