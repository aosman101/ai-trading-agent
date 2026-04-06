from __future__ import annotations

import os
import sys
from types import ModuleType

import pandas as pd

os.environ.setdefault("TRADING_MODE", "paper")

if "neuralforecast" not in sys.modules:
    neuralforecast = ModuleType("neuralforecast")
    neuralforecast.NeuralForecast = object
    models = ModuleType("neuralforecast.models")
    models.NHITS = object
    models.iTransformer = object
    neuralforecast.models = models
    sys.modules["neuralforecast"] = neuralforecast
    sys.modules["neuralforecast.models"] = models

if "yfinance" not in sys.modules:
    from unittest.mock import MagicMock

    sys.modules["yfinance"] = MagicMock()

from app.models.itransformer_model import ITransformerForecaster
from app.models.nhits_forecaster import NHITSForecaster


class RecordingForecastModel:
    def __init__(self, rows: pd.DataFrame):
        self.rows = rows
        self.last_df = None

    def predict(self, df=None):
        self.last_df = df
        return self.rows.copy()


def _forecast_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=4)
    return pd.DataFrame(
        {
            "ds": dates,
            "symbol": ["AAPL"] * 4,
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )


def test_nhits_predict_latest_uses_latest_frame_context():
    forecaster = NHITSForecaster.__new__(NHITSForecaster)
    forecast_rows = pd.DataFrame(
        {
            "unique_id": ["AAPL", "AAPL"],
            "ds": pd.bdate_range("2024-01-08", periods=2),
            "NHITS": [104.0, 105.0],
        }
    )
    forecaster.model = RecordingForecastModel(forecast_rows)

    signal = NHITSForecaster.predict_latest(forecaster, _forecast_frame(), "AAPL")

    assert forecaster.model.last_df is not None
    assert signal.symbol == "AAPL"
    assert signal.direction == "long"


def test_itransformer_predict_latest_uses_latest_frame_context():
    forecaster = ITransformerForecaster.__new__(ITransformerForecaster)
    forecast_rows = pd.DataFrame(
        {
            "unique_id": ["AAPL"],
            "ds": pd.bdate_range("2024-01-08", periods=1),
            "iTransformer": [104.0],
        }
    )
    forecaster.model = RecordingForecastModel(forecast_rows)

    signal = ITransformerForecaster.predict_latest(forecaster, _forecast_frame(), "AAPL")

    assert forecaster.model.last_df is not None
    assert signal.symbol == "AAPL"
    assert signal.direction == "long"
