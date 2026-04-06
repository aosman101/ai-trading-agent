from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.data.market_data import MarketDataService
from app.types import ModelSignal
from app.utils.safe_model_io import load_model, save_model


class ITransformerForecaster:
    def __init__(self) -> None:
        try:
            from neuralforecast import NeuralForecast  # type: ignore
            from neuralforecast.models import iTransformer  # type: ignore
        except Exception as exc:
            raise ImportError("neuralforecast with iTransformer support is required") from exc

        self.settings = get_settings()
        self._NeuralForecast = NeuralForecast
        self._iTransformer = iTransformer
        self.model = None
        self.freq = "D"

    def fit(self, frame: pd.DataFrame) -> None:
        train_df = MarketDataService.to_neuralforecast_frame(frame)
        base_model = self._iTransformer(
            h=self.settings.forecast_horizon,
            input_size=self.settings.n_hits_input_size,
            max_steps=self.settings.n_hits_max_steps,
            learning_rate=1e-3,
            scaler_type="robust",
            random_seed=42,
        )
        self.model = self._NeuralForecast(models=[base_model], freq=self.freq)
        self.model.fit(train_df)

    def predict_latest(self, frame: pd.DataFrame, symbol: str) -> ModelSignal:
        if self.model is None:
            raise RuntimeError("iTransformer model is not trained")
        forecasts = self.model.predict().sort_values(["unique_id", "ds"])
        symbol_forecasts = forecasts[forecasts["unique_id"] == symbol]
        value_columns = [column for column in symbol_forecasts.columns if column not in {"unique_id", "ds"}]
        point_column = value_columns[0]
        predicted_price = float(symbol_forecasts.iloc[0][point_column])
        current_price = float(
            MarketDataService.to_neuralforecast_frame(frame)
            .query("unique_id == @symbol")
            .sort_values("ds")
            .iloc[-1]["y"]
        )
        expected_return = predicted_price / current_price - 1.0
        direction = "flat"
        if expected_return > 0.002:
            direction = "long"
        elif expected_return < -0.002:
            direction = "short"
        return ModelSignal(
            name="itransformer",
            symbol=symbol,
            direction=direction,
            score=expected_return,
            confidence=min(1.0, abs(expected_return) / 0.03),
            metadata={"current_price": current_price, "predicted_price": predicted_price},
        )

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained iTransformer model")
        save_model(self.model, path)

    def load(self, path: str | Path) -> None:
        self.model = load_model(path)
