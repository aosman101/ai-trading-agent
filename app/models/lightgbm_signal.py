from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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
        self.features: list[str] = []
        self.target_column = f"target_up_{self.settings.lightgbm_horizon}"

    def fit(self, frame: pd.DataFrame) -> None:
        train_df = frame.dropna().copy()
        self.features = MarketDataService.feature_columns(train_df)
        if self.target_column not in train_df.columns:
            raise ValueError(f"Missing target column {self.target_column}")

        X = train_df[self.features]
        y = train_df[self.target_column].astype(int)

        split = int(len(train_df) * 0.8)
        X_train, X_valid = X.iloc[:split], X.iloc[split:]
        y_train, y_valid = y.iloc[:split], y.iloc[split:]

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
        valid_acc = float((self.model.predict(X_valid) == y_valid).mean()) if len(X_valid) else 0.0
        logger.info("LightGBM validation accuracy: %.4f", valid_acc)

    def predict_latest(self, latest_row: pd.Series) -> ModelSignal:
        if self.model is None:
            raise RuntimeError("LightGBM model is not trained")
        X = latest_row[self.features].fillna(0.0).to_frame().T
        probabilities = self.model.predict_proba(X)[0]
        prob_up = float(probabilities[1])
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
                "top_features": self.feature_importance(top_n=10),
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
            "features": self.features,
            "target_column": self.target_column,
        }
        save_model(payload, path)

    def load(self, path: str | Path) -> None:
        payload = load_model(path)
        self.model = payload["model"]
        self.features = payload["features"]
        self.target_column = payload["target_column"]
