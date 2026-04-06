from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TRADING_MODE", "paper")

# Stub yfinance so we can import the module without the full dependency
if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = MagicMock()

from app.data.market_data import (
    add_targets,
    add_technical_indicators,
    atr,
    bollinger,
    ema,
    macd,
    rsi,
    sma,
)


def _sample_ohlcv(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1_000_000, 10_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestIndicators:
    def test_ema_length(self):
        series = pd.Series(range(50), dtype=float)
        result = ema(series, span=10)
        assert len(result) == 50

    def test_sma_length(self):
        series = pd.Series(range(50), dtype=float)
        result = sma(series, window=10)
        assert len(result) == 50
        assert pd.isna(result.iloc[0])
        assert not pd.isna(result.iloc[9])

    def test_rsi_bounded(self):
        df = _sample_ohlcv()
        result = rsi(df["close"]).dropna()
        assert result.min() >= 0
        assert result.max() <= 100

    def test_macd_columns(self):
        df = _sample_ohlcv()
        result = macd(df["close"])
        assert set(result.columns) == {"macd", "macd_signal", "macd_hist"}

    def test_bollinger_columns(self):
        df = _sample_ohlcv()
        result = bollinger(df["close"])
        assert set(result.columns) == {"bb_mid", "bb_upper", "bb_lower", "bb_width"}

    def test_bollinger_upper_above_lower(self):
        df = _sample_ohlcv()
        result = bollinger(df["close"]).dropna()
        assert (result["bb_upper"] >= result["bb_lower"]).all()

    def test_atr_positive(self):
        df = _sample_ohlcv()
        result = atr(df["high"], df["low"], df["close"]).dropna()
        assert (result >= 0).all()


class TestAddTechnicalIndicators:
    def test_adds_expected_columns(self):
        df = _sample_ohlcv()
        result = add_technical_indicators(df)
        expected = {"rsi_14", "macd", "atr_14", "ema_20", "sma_20", "bb_width", "trend_strength"}
        assert expected.issubset(set(result.columns))

    def test_no_infinite_values(self):
        df = _sample_ohlcv()
        result = add_technical_indicators(df).dropna()
        numeric = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()


class TestAddTargets:
    def test_creates_forward_return_columns(self):
        df = _sample_ohlcv()
        result = add_targets(df, horizons=(1, 5))
        assert "forward_return_1" in result.columns
        assert "forward_return_5" in result.columns
        assert "target_up_1" in result.columns
        assert "target_up_5" in result.columns

    def test_target_up_is_binary(self):
        df = _sample_ohlcv()
        result = add_targets(df, horizons=(1,))
        valid = result["target_up_1"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_forward_return_last_rows_are_nan(self):
        df = _sample_ohlcv(50)
        result = add_targets(df, horizons=(5,))
        assert result["forward_return_5"].iloc[-1] != result["forward_return_5"].iloc[-1]  # NaN check

    def test_no_targets_when_flag_false(self):
        df = _sample_ohlcv()
        result = add_technical_indicators(df)
        assert "forward_return_1" not in result.columns
