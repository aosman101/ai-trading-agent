from __future__ import annotations

import time
from typing import Dict, Optional

import httpx

from app.config import get_settings
from app.types import ModelSignal
from app.utils.logging import get_logger

logger = get_logger(__name__)

_DSI_SIGNAL_MAP = {"BUY": "long", "SELL": "short", "HOLD": "flat"}
_REQUEST_TIMEOUT = 30.0


class DSIClient:
    """Fetches predictions from the Deep Stock Insights platform."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._base_url = self.settings.dsi_base_url.rstrip("/")
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        self.last_fetch_errors: Dict[str, str] = {}

    @property
    def configured(self) -> bool:
        return self.settings.dsi_configured

    def _authenticate(self) -> None:
        if self._token and time.time() < self._token_expiry:
            return
        response = httpx.post(
            f"{self._base_url}/api/auth/login",
            data={"username": self.settings.dsi_email, "password": self.settings.dsi_password},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        self._token = payload["access_token"]
        # Refresh 5 minutes before assumed 60-minute expiry
        self._token_expiry = time.time() + 3300

    def _headers(self) -> Dict[str, str]:
        self._authenticate()
        return {"Authorization": f"Bearer {self._token}"}

    @staticmethod
    def _map_symbol(symbol: str) -> str:
        """Map agent symbol to DSI symbol. Currently 1:1 but centralised for future mapping."""
        return symbol.upper()

    @staticmethod
    def _derive_score(
        change_pct: float,
        current_price: float,
        predicted_close: float,
    ) -> float:
        """Return a fractional expected-return score for DSI predictions.

        DSI has historically been inconsistent about whether ``predicted_change_pct`` is
        a fraction (0.015) or a percentage (1.5). Cross-check against the price delta
        when possible, and clip to a safe range so one bad payload cannot nuke the
        ensemble.
        """
        computed = None
        if current_price and predicted_close:
            try:
                computed = (float(predicted_close) / float(current_price)) - 1.0
            except (TypeError, ZeroDivisionError):
                computed = None
        candidate = float(change_pct or 0.0)
        if abs(candidate) > 1.0:
            candidate = candidate / 100.0
        if computed is not None and abs(candidate) > 0 and abs(candidate) / max(abs(computed), 1e-9) > 10.0:
            # Large disagreement implies the payload unit is off — trust the price-derived value.
            candidate = computed
        score = candidate if candidate != 0.0 else (computed or 0.0)
        return max(-0.5, min(0.5, float(score)))

    @classmethod
    def _to_model_signal(cls, response: Dict, model_key: str, symbol: str) -> ModelSignal:
        prediction = response.get("prediction") or {}
        current_price = float(response.get("current_price") or prediction.get("current_price") or 0.0)
        predicted_close = float(prediction.get("predicted_close") or response.get("predicted_close") or 0.0)
        change_pct = float(prediction.get("predicted_change_pct") or response.get("predicted_change_pct") or 0.0)
        confidence = float(prediction.get("confidence") or response.get("confidence") or 0.0)
        signal_strength = float(prediction.get("signal_strength") or response.get("signal_strength") or 0.0)
        raw_signal = prediction.get("signal") or response.get("signal") or "HOLD"
        direction = _DSI_SIGNAL_MAP.get(raw_signal.upper(), "flat")

        score = cls._derive_score(change_pct, current_price, predicted_close)

        stop_loss = prediction.get("stop_loss") or response.get("stop_loss")
        take_profit = prediction.get("take_profit") or response.get("take_profit")
        horizon = prediction.get("prediction_horizon") or response.get("prediction_horizon") or "1d"

        return ModelSignal(
            name=model_key,
            symbol=symbol,
            direction=direction,
            score=score,
            confidence=confidence,
            metadata={
                "source": "dsi",
                "current_price": current_price,
                "predicted_close": predicted_close,
                "signal_strength": signal_strength,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "prediction_horizon": horizon,
            },
        )

    def fetch_nhits_signal(self, symbol: str) -> ModelSignal:
        dsi_symbol = self._map_symbol(symbol)
        response = httpx.get(
            f"{self._base_url}/api/predictions/{dsi_symbol}",
            params={"horizon": "1d", "model_key": "nhits"},
            headers=self._headers(),
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return self._to_model_signal(response.json(), "nhits", symbol)

    def fetch_tft_signal(self, symbol: str) -> ModelSignal:
        dsi_symbol = self._map_symbol(symbol)
        response = httpx.get(
            f"{self._base_url}/api/predictions/{dsi_symbol}",
            params={"horizon": "1d", "model_key": "tft"},
            headers=self._headers(),
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return self._to_model_signal(response.json(), "tft", symbol)

    def fetch_lightgbm_signal(self, symbol: str) -> ModelSignal:
        dsi_symbol = self._map_symbol(symbol)
        response = httpx.get(
            f"{self._base_url}/api/scanner/predict/{dsi_symbol}",
            headers=self._headers(),
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return self._to_model_signal(response.json(), "lightgbm", symbol)

    def fetch_all_signals(self, symbol: str) -> list[ModelSignal]:
        """Fetch N-HiTS, TFT, and LightGBM signals. Skip any that fail."""
        signals: list[ModelSignal] = []
        self.last_fetch_errors = {}
        fetchers = [
            ("nhits", self.fetch_nhits_signal),
            ("tft", self.fetch_tft_signal),
            ("lightgbm", self.fetch_lightgbm_signal),
        ]
        for model_key, fetcher in fetchers:
            try:
                signals.append(fetcher(symbol))
            except Exception as exc:
                self.last_fetch_errors[model_key] = str(exc)
                logger.warning("DSI %s prediction failed for %s: %s", model_key, symbol, exc)
        return signals
