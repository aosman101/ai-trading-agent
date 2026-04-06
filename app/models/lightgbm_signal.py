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


class LightGBMSignalModel:
    def __init__(self) -> None:
        try:
            from lightgbm import LGBMClassifier  # type: ignore
        except Exception as exc:
            raise ImportError("lightgbm is required for LightGBMSignalModel") from exc

        self._LGBMClassifier = LGBMClassifier
        self.settings = get_settings()
        self.model = None
        self.calibration_model = None
        self.features: list[str] = []
        self.validation_metrics: Dict[str, float] = {}
        self.target_column = f"target_up_{self.settings.lightgbm_horizon}"

    @staticmethod
    def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if "ds" in frame.columns:
            return frame.sort_values(["ds", "symbol"]).reset_index(drop=True)
        return frame.reset_index(drop=True)

    def _split_train_calibration(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if "ds" in frame.columns:
            ds_values = pd.to_datetime(frame["ds"], errors="coerce")
            unique_dates = sorted(date for date in ds_values.dropna().unique())
            if len(unique_dates) >= 30:
                calibration_window = max(20, len(unique_dates) // 5)
                calibration_start = unique_dates[-calibration_window]
                train_mask = ds_values < calibration_start
                calibration_mask = ~train_mask
                if train_mask.any() and calibration_mask.any():
                    return frame.loc[train_mask].copy(), frame.loc[calibration_mask].copy()

        split = max(1, int(len(frame) * 0.8))
        return frame.iloc[:split].copy(), frame.iloc[split:].copy()

    def _fit_calibration_model(self, raw_probabilities: np.ndarray, labels: pd.Series) -> None:
        self.calibration_model = None
        if len(raw_probabilities) == 0 or labels.nunique() < 2:
            return
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception as exc:
            logger.warning("Calibration model unavailable: %s", exc)
            return

        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(raw_probabilities.reshape(-1, 1), labels.astype(int))
        self.calibration_model = calibrator

    def _apply_calibration(self, raw_probabilities: np.ndarray) -> np.ndarray:
        if self.calibration_model is None:
            return raw_probabilities
        calibrated = self.calibration_model.predict_proba(raw_probabilities.reshape(-1, 1))[:, 1]
        return np.clip(calibrated, 0.0, 1.0)

    def fit(self, frame: pd.DataFrame) -> None:
        train_df = self._sort_frame(frame.dropna().copy())
        self.features = MarketDataService.feature_columns(train_df)
        if self.target_column not in train_df.columns:
            raise ValueError(f"Missing target column {self.target_column}")

        train_slice, calibration_slice = self._split_train_calibration(train_df)
        if train_slice.empty:
            train_slice = train_df.copy()
        if calibration_slice.empty:
            calibration_slice = train_df.iloc[-max(1, len(train_df) // 5) :].copy()

        X_train = train_slice[self.features]
        y_train = train_slice[self.target_column].astype(int)
        X_valid = calibration_slice[self.features]
        y_valid = calibration_slice[self.target_column].astype(int)
        if len(X_train) < 10 or y_train.nunique() < 2:
            X_train = train_df[self.features]
            y_train = train_df[self.target_column].astype(int)
            X_valid = calibration_slice[self.features]
            y_valid = calibration_slice[self.target_column].astype(int)

        self.model = self._LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            class_weight="balanced",
        )
        self.model.fit(X_train, y_train)

        raw_valid_prob = self.model.predict_proba(X_valid)[:, 1] if len(X_valid) else np.array([], dtype=float)
        self._fit_calibration_model(raw_valid_prob, y_valid)
        valid_prob = self._apply_calibration(raw_valid_prob) if len(raw_valid_prob) else raw_valid_prob
        valid_pred = (valid_prob >= 0.5).astype(int) if len(valid_prob) else np.array([], dtype=int)

        valid_acc = float((valid_pred == y_valid.to_numpy()).mean()) if len(y_valid) else 0.0
        valid_brier = (
            float(np.mean((valid_prob - y_valid.to_numpy()) ** 2))
            if len(y_valid)
            else 0.0
        )
        self.validation_metrics = {
            "validation_accuracy": valid_acc,
            "validation_brier": valid_brier,
            "train_rows": float(len(X_train)),
            "validation_rows": float(len(X_valid)),
        }
        logger.info(
            "LightGBM temporal validation accuracy: %.4f | brier: %.4f",
            valid_acc,
            valid_brier,
        )

    def predict_latest(self, latest_row: pd.Series) -> ModelSignal:
        if self.model is None:
            raise RuntimeError("LightGBM model is not trained")
        X = latest_row[self.features].fillna(0.0).to_frame().T
        raw_prob_up = float(self.model.predict_proba(X)[0][1])
        prob_up = float(self._apply_calibration(np.array([raw_prob_up], dtype=float))[0])
        score = 2.0 * prob_up - 1.0
        direction = "flat"
        if score > 0.05:
            direction = "long"
        elif score < -0.05:
            direction = "short"
        return ModelSignal(
            name="lightgbm",
            symbol=str(latest_row.get("symbol", "UNKNOWN")),
            direction=direction,
            score=score,
            confidence=abs(score),
            metadata={
                "prob_up": prob_up,
                "raw_prob_up": raw_prob_up,
                "top_features": self.feature_importance(top_n=10),
                "validation_metrics": self.validation_metrics,
            },
        )

    def feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        if self.model is None or not self.features:
            return {}
        values = self.model.feature_importances_
        pairs = sorted(zip(self.features, values), key=lambda item: item[1], reverse=True)
        return {name: float(score) for name, score in pairs[:top_n]}

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained LightGBM model")
        payload = {
            "model": self.model,
            "calibration_model": self.calibration_model,
            "features": self.features,
            "validation_metrics": self.validation_metrics,
            "target_column": self.target_column,
        }
        save_model(payload, path)

    def load(self, path: str | Path) -> None:
        payload = load_model(path)
        self.model = payload["model"]
        self.calibration_model = payload.get("calibration_model")
        self.features = payload["features"]
        self.validation_metrics = payload.get("validation_metrics", {})
        self.target_column = payload["target_column"]
