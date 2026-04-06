from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.config import get_settings
from app.data.market_data import MarketDataService
from app.types import ModelSignal
from app.utils.logging import get_logger
from app.utils.safe_model_io import load_model, save_model

logger = get_logger(__name__)


class TFTForecaster:
    def __init__(self) -> None:
        try:
            from lightning.pytorch import Trainer  # type: ignore
            from lightning.pytorch.callbacks import EarlyStopping  # type: ignore
            from pytorch_forecasting import (  # type: ignore
                GroupNormalizer,
                TemporalFusionTransformer,
                TimeSeriesDataSet,
            )
            from pytorch_forecasting.metrics import QuantileLoss  # type: ignore
        except Exception as exc:
            raise ImportError("lightning and pytorch-forecasting are required for TFTForecaster") from exc

        self.settings = get_settings()
        self._Trainer = Trainer
        self._EarlyStopping = EarlyStopping
        self._GroupNormalizer = GroupNormalizer
        self._TemporalFusionTransformer = TemporalFusionTransformer
        self._TimeSeriesDataSet = TimeSeriesDataSet
        self._QuantileLoss = QuantileLoss

        self.model = None
        self.trainer = None
        self.dataset_parameters: Dict[str, Any] | None = None
        self.known_reals: list[str] = []

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        tft_frame = MarketDataService.to_tft_frame(frame)
        preferred = [
            "time_idx",
            "day_of_week",
            "day_of_month",
            "month",
            "is_month_end",
            "volume",
            "rsi_14",
            "macd",
            "macd_signal",
            "bb_width",
            "atr_14",
            "realized_vol_20",
            "trend_strength",
        ]
        self.known_reals = [column for column in preferred if column in tft_frame.columns]
        return tft_frame.sort_values(["unique_id", "time_idx"]).reset_index(drop=True)

    def fit(self, frame: pd.DataFrame) -> None:
        tft_frame = self._prepare_frame(frame)
        training_cutoff = int(tft_frame["time_idx"].max()) - self.settings.tft_prediction_length
        if training_cutoff <= self.settings.tft_encoder_length:
            raise ValueError("Not enough rows to train TFT. Increase lookback window.")

        training = self._TimeSeriesDataSet(
            tft_frame[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="y",
            group_ids=["unique_id"],
            min_encoder_length=max(30, self.settings.tft_encoder_length // 2),
            max_encoder_length=self.settings.tft_encoder_length,
            min_prediction_length=self.settings.tft_prediction_length,
            max_prediction_length=self.settings.tft_prediction_length,
            static_categoricals=["unique_id"],
            time_varying_known_reals=self.known_reals,
            time_varying_unknown_reals=["y"],
            target_normalizer=self._GroupNormalizer(groups=["unique_id"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = self._TimeSeriesDataSet.from_dataset(
            training, tft_frame, predict=True, stop_randomization=True
        )
        train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

        self.model = self._TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=1e-3,
            hidden_size=16,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=self._QuantileLoss(),
            log_interval=-1,
            reduce_on_plateau_patience=3,
        )

        early_stopping = self._EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min",
        )
        self.trainer = self._Trainer(
            max_epochs=self.settings.tft_max_epochs,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
            callbacks=[early_stopping],
        )
        self.trainer.fit(self.model, train_loader, val_loader)
        self.dataset_parameters = training.get_parameters()
        logger.info("TFT training complete")

    def _build_future_rows(self, symbol_frame: pd.DataFrame) -> pd.DataFrame:
        latest = symbol_frame.iloc[-1]
        future_dates = pd.bdate_range(
            latest["ds"] + pd.tseries.offsets.BDay(1),
            periods=self.settings.tft_prediction_length,
        )
        rows = []
        for step, future_date in enumerate(future_dates, start=1):
            row = latest.copy()
            row["ds"] = pd.Timestamp(future_date)
            row["time_idx"] = int(latest["time_idx"]) + step
            row["y"] = np.nan
            row["day_of_week"] = future_date.dayofweek
            row["day_of_month"] = future_date.day
            row["month"] = future_date.month
            row["is_month_end"] = int(future_date.is_month_end)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _extract_prediction_tensor(raw_output: Any) -> np.ndarray:
        prediction = None
        if isinstance(raw_output, tuple):
            raw_output = raw_output[0]
        if hasattr(raw_output, "output") and hasattr(raw_output.output, "prediction"):
            prediction = raw_output.output.prediction
        elif hasattr(raw_output, "prediction"):
            prediction = raw_output.prediction
        elif isinstance(raw_output, dict) and "prediction" in raw_output:
            prediction = raw_output["prediction"]
        else:
            prediction = raw_output

        if hasattr(prediction, "detach"):
            prediction = prediction.detach().cpu().numpy()
        return np.asarray(prediction)

    def _interpret(self, raw_output: Any) -> dict[str, float]:
        if self.model is None:
            return {}
        try:
            interpretation = self.model.interpret_output(raw_output, reduction="sum")
        except Exception:
            return {}

        payload: dict[str, float] = {}
        for key in ("encoder_variables", "decoder_variables"):
            values = interpretation.get(key)
            if values is None:
                continue
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            arr = np.asarray(values).reshape(-1)
            for name, score in zip(self.known_reals, arr[: len(self.known_reals)]):
                payload[f"{key}:{name}"] = float(score)
        return payload

    def predict_latest(self, frame: pd.DataFrame, symbol: str) -> ModelSignal:
        if self.model is None or self.dataset_parameters is None:
            raise RuntimeError("TFT model is not trained")

        tft_frame = self._prepare_frame(frame)
        symbol_frame = tft_frame[tft_frame["unique_id"] == symbol].copy()
        if symbol_frame.empty:
            raise ValueError(f"No TFT frame rows found for {symbol}")
        prediction_frame = pd.concat(
            [symbol_frame, self._build_future_rows(symbol_frame)], ignore_index=True
        )
        dataset = self._TimeSeriesDataSet.from_parameters(
            self.dataset_parameters,
            prediction_frame,
            predict=True,
            stop_randomization=True,
        )
        dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        raw_output = self.model.predict(dataloader, mode="raw", return_x=True)
        predictions = self._extract_prediction_tensor(raw_output)

        if predictions.ndim == 3:
            lower = predictions[0, :, 0]
            median = predictions[0, :, predictions.shape[-1] // 2]
            upper = predictions[0, :, -1]
        elif predictions.ndim == 2:
            lower = predictions[0]
            median = predictions[0]
            upper = predictions[0]
        else:
            flat = predictions.reshape(-1)
            lower = flat
            median = flat
            upper = flat

        current_price = float(symbol_frame.iloc[-1]["y"])
        point_forecast = float(median[0])
        expected_return = point_forecast / current_price - 1.0
        interval_width = float(max(upper[0] - lower[0], 0.0))
        volatility_proxy = interval_width / current_price if current_price else 0.0

        direction = "flat"
        if expected_return > 0.002:
            direction = "long"
        elif expected_return < -0.002:
            direction = "short"

        return ModelSignal(
            name="tft",
            symbol=symbol,
            direction=direction,
            score=expected_return,
            confidence=max(0.0, 1.0 - min(volatility_proxy / 0.05, 1.0)),
            metadata={
                "current_price": current_price,
                "expected_price": point_forecast,
                "lower_bound": float(lower[0]),
                "upper_bound": float(upper[0]),
                "interval_width": interval_width,
                "volatility_proxy": volatility_proxy,
                "attention": self._interpret(raw_output[0] if isinstance(raw_output, tuple) else raw_output),
            },
        )

    def save(self, checkpoint_path: str | Path) -> None:
        if self.model is None or self.trainer is None or self.dataset_parameters is None:
            raise RuntimeError("Cannot save an untrained TFT model")
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save_checkpoint(str(checkpoint_path))
        save_model(
            {
                "dataset_parameters": self.dataset_parameters,
                "known_reals": self.known_reals,
            },
            checkpoint_path.with_suffix(".params.pkl"),
        )

    def load(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        self.model = self._TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))
        payload = load_model(checkpoint_path.with_suffix(".params.pkl"))
        self.dataset_parameters = payload["dataset_parameters"]
        self.known_reals = payload["known_reals"]
