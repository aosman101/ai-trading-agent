from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("TRADING_MODE", "paper")

for module_name in ("feedparser", "gymnasium", "httpx", "yfinance"):
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()

from app.types import ModelSignal


# ── Sample DSI API responses ───────────────────────────────────

NHITS_RESPONSE = {
    "status": "ok",
    "asset": "AAPL",
    "model_key": "nhits",
    "current_price": 180.0,
    "prediction": {
        "predicted_close": 185.0,
        "predicted_change_pct": 2.78,
        "confidence": 0.72,
        "signal": "BUY",
        "signal_strength": 0.65,
        "prediction_horizon": "1d",
        "stop_loss": 176.0,
        "take_profit": 190.0,
    },
    "predictions": [
        {
            "predicted_close": 185.0,
            "predicted_change_pct": 2.78,
            "confidence": 0.72,
            "signal": "BUY",
            "signal_strength": 0.65,
            "prediction_horizon": "1d",
        },
        {
            "predicted_close": 188.0,
            "predicted_change_pct": 4.44,
            "confidence": 0.60,
            "signal": "BUY",
            "signal_strength": 0.55,
            "prediction_horizon": "3d",
        },
    ],
}

TFT_RESPONSE = {
    "status": "ok",
    "asset": "AAPL",
    "model_key": "tft",
    "current_price": 180.0,
    "prediction": {
        "predicted_close": 183.0,
        "predicted_change_pct": 1.67,
        "confidence": 0.68,
        "signal": "BUY",
        "signal_strength": 0.58,
        "prediction_horizon": "1d",
        "stop_loss": 177.0,
        "take_profit": 189.0,
    },
    "predictions": [
        {
            "predicted_close": 183.0,
            "predicted_change_pct": 1.67,
            "confidence": 0.68,
            "signal": "BUY",
            "signal_strength": 0.58,
            "prediction_horizon": "1d",
        },
    ],
}

LIGHTGBM_SCANNER_RESPONSE = {
    "status": "ok",
    "asset": "AAPL",
    "model_key": "lightgbm",
    "current_price": 180.0,
    "predicted_close": 182.5,
    "predicted_change_pct": 1.39,
    "signal": "BUY",
    "confidence": 0.61,
    "signal_strength": 0.55,
    "prediction_horizon": "1d",
}


# ── Tests ──────────────────────────────────────────────────────


def test_convert_nhits_response_to_model_signal():
    from app.data.dsi_client import DSIClient

    signal = DSIClient._to_model_signal(NHITS_RESPONSE, "nhits", "AAPL")
    assert isinstance(signal, ModelSignal)
    assert signal.name == "nhits"
    assert signal.symbol == "AAPL"
    assert signal.direction == "long"
    assert signal.score == pytest.approx(0.0278, abs=0.001)
    assert signal.confidence == pytest.approx(0.72)
    assert signal.metadata["current_price"] == 180.0
    assert signal.metadata["predicted_close"] == 185.0


def test_convert_tft_response_to_model_signal():
    from app.data.dsi_client import DSIClient

    signal = DSIClient._to_model_signal(TFT_RESPONSE, "tft", "AAPL")
    assert isinstance(signal, ModelSignal)
    assert signal.name == "tft"
    assert signal.direction == "long"
    assert signal.metadata["predicted_close"] == 183.0


def test_convert_lightgbm_response_to_model_signal():
    from app.data.dsi_client import DSIClient

    signal = DSIClient._to_model_signal(LIGHTGBM_SCANNER_RESPONSE, "lightgbm", "AAPL")
    assert isinstance(signal, ModelSignal)
    assert signal.name == "lightgbm"
    assert signal.direction == "long"
    assert signal.confidence == pytest.approx(0.61)


def test_convert_sell_signal_maps_to_short():
    sell_response = {**NHITS_RESPONSE, "prediction": {**NHITS_RESPONSE["prediction"], "signal": "SELL", "predicted_change_pct": -2.5}}
    from app.data.dsi_client import DSIClient

    signal = DSIClient._to_model_signal(sell_response, "nhits", "AAPL")
    assert signal.direction == "short"


def test_convert_hold_signal_maps_to_flat():
    hold_response = {**NHITS_RESPONSE, "prediction": {**NHITS_RESPONSE["prediction"], "signal": "HOLD", "predicted_change_pct": 0.1}}
    from app.data.dsi_client import DSIClient

    signal = DSIClient._to_model_signal(hold_response, "nhits", "AAPL")
    assert signal.direction == "flat"


def test_convert_missing_prediction_returns_flat():
    empty_response = {"status": "ok", "asset": "AAPL", "model_key": "nhits", "current_price": 180.0}
    from app.data.dsi_client import DSIClient

    signal = DSIClient._to_model_signal(empty_response, "nhits", "AAPL")
    assert signal.direction == "flat"
    assert signal.score == 0.0


def test_symbol_mapping_stock_ticker():
    from app.data.dsi_client import DSIClient

    assert DSIClient._map_symbol("AAPL") == "AAPL"
    assert DSIClient._map_symbol("MSFT") == "MSFT"
    assert DSIClient._map_symbol("SPY") == "SPY"


def test_model_bundle_no_longer_requires_nhits_tft_lightgbm():
    from app.training.retrainer import ModelBundle
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(ModelBundle)}
    assert "nhits" not in field_names
    assert "tft" not in field_names
    assert "lightgbm" not in field_names
    assert "finbert" in field_names
    assert "ppo" in field_names
    assert "dqn" in field_names
