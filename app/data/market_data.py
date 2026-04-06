from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List

import numpy as np
import pandas as pd
import yfinance as yf

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _flatten_yfinance_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(symbol, axis=1, level=-1)
        except Exception:
            df.columns = ["_".join([str(p) for p in col if p]) for col in df.columns]
    rename_map = {col: str(col).lower().replace(" ", "_") for col in df.columns}
    return df.rename(columns=rename_map)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_line = ema(close, fast)
    slow_line = ema(close, slow)
    macd_line = fast_line - slow_line
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist},
        index=close.index,
    )


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width},
        index=close.index,
    )


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return true_range.rolling(window).mean()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["return_1"] = data["close"].pct_change()
    data["return_5"] = data["close"].pct_change(5)
    data["log_return_1"] = np.log(data["close"]).diff()
    data["realized_vol_20"] = data["log_return_1"].rolling(20).std() * np.sqrt(252)

    data["sma_20"] = sma(data["close"], 20)
    data["sma_50"] = sma(data["close"], 50)
    data["ema_20"] = ema(data["close"], 20)
    data["ema_50"] = ema(data["close"], 50)
    data["ema_200"] = ema(data["close"], 200)

    data["rsi_14"] = rsi(data["close"], 14)

    macd_df = macd(data["close"])
    for column in macd_df.columns:
        data[column] = macd_df[column]

    bollinger_df = bollinger(data["close"])
    for column in bollinger_df.columns:
        data[column] = bollinger_df[column]

    data["atr_14"] = atr(data["high"], data["low"], data["close"], 14)
    data["rolling_high_20"] = data["high"].rolling(20).max()
    data["rolling_low_20"] = data["low"].rolling(20).min()
    data["price_zscore_20"] = (
        (data["close"] - data["close"].rolling(20).mean())
        / data["close"].rolling(20).std(ddof=0)
    )
    data["volume_zscore_20"] = (
        (data["volume"] - data["volume"].rolling(20).mean())
        / data["volume"].rolling(20).std(ddof=0)
    )
    data["trend_strength"] = (data["ema_50"] / data["ema_200"]) - 1.0
    data["drawdown_252"] = data["close"] / data["close"].rolling(252).max() - 1.0

    data["day_of_week"] = data.index.dayofweek
    data["day_of_month"] = data.index.day
    data["month"] = data.index.month
    data["is_month_end"] = data.index.is_month_end.astype(int)

    return data


def add_targets(df: pd.DataFrame, horizons: Iterable[int] = (1, 5, 10)) -> pd.DataFrame:
    data = df.copy()
    for horizon in horizons:
        data[f"forward_return_{horizon}"] = data["close"].shift(-horizon) / data["close"] - 1.0
        data[f"target_up_{horizon}"] = (data[f"forward_return_{horizon}"] > 0).astype(int)
    return data


@dataclass
class MarketFrame:
    symbol: str
    frame: pd.DataFrame


class MarketDataService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def fetch_symbol_history(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str | None = None,
        include_targets: bool = False,
    ) -> pd.DataFrame:
        if end is None:
            end = datetime.utcnow()
        if start is None:
            start = end - timedelta(days=365 * self.settings.lookback_years)
        if interval is None:
            interval = self.settings.bar_interval

        raw = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw.empty:
            raise ValueError(f"No data returned for {symbol}")

        data = _flatten_yfinance_columns(raw, symbol).copy()
        data.index = pd.to_datetime(data.index).tz_localize(None)
        data = data.rename(
            columns={
                "adj_close": "adj_close",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        required = {"open", "high", "low", "close", "volume"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required columns for {symbol}: {sorted(missing)}")

        bad_ohlc = (data["high"] < data["low"]).sum()
        if bad_ohlc > 0:
            logger.warning("%s: %d rows where high < low — dropping them", symbol, bad_ohlc)
            data = data[data["high"] >= data["low"]]
        close_above_high = (data["close"] > data["high"] * 1.001).sum()
        close_below_low = (data["close"] < data["low"] * 0.999).sum()
        if close_above_high > 0 or close_below_low > 0:
            logger.warning(
                "%s: %d rows with close outside high/low range",
                symbol,
                close_above_high + close_below_low,
            )

        data["symbol"] = symbol
        data = add_technical_indicators(data)
        if include_targets:
            data = add_targets(data, horizons=(1, 5, 10))
        return data.dropna().copy()

    def fetch_universe_history(self, symbols: List[str] | None = None, include_targets: bool = True) -> pd.DataFrame:
        symbols = symbols or self.settings.symbols
        frames = []
        failed: list[str] = []
        for symbol in symbols:
            try:
                symbol_df = self.fetch_symbol_history(symbol, include_targets=include_targets)
                frames.append(symbol_df.reset_index(names="ds"))
            except Exception as exc:
                logger.exception("Failed to fetch history for %s: %s", symbol, exc)
                failed.append(symbol)
        if not frames:
            raise RuntimeError("No symbols returned data.")
        success_rate = len(frames) / len(symbols)
        if success_rate < 0.5:
            raise RuntimeError(
                f"Only {len(frames)}/{len(symbols)} symbols returned data "
                f"(failed: {failed}). Aborting to prevent biased training."
            )
        return pd.concat(frames, ignore_index=True).sort_values(["symbol", "ds"])

    def latest_feature_row(self, symbol: str) -> pd.Series:
        history = self.fetch_symbol_history(symbol)
        return history.iloc[-1]

    @staticmethod
    def to_neuralforecast_frame(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        if "ds" not in frame.columns:
            frame = frame.reset_index(names="ds")
        return frame.rename(columns={"symbol": "unique_id", "close": "y"})[
            ["unique_id", "ds", "y"]
        ].dropna()

    @staticmethod
    def to_tft_frame(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        if "ds" not in frame.columns:
            frame = frame.reset_index(names="ds")
        frame = frame.rename(columns={"symbol": "unique_id", "close": "y"})
        frame = frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        frame["time_idx"] = frame.groupby("unique_id").cumcount()
        return frame

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> list[str]:
        excluded = {
            "symbol",
            "ds",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "forward_return_1",
            "forward_return_5",
            "forward_return_10",
            "target_up_1",
            "target_up_5",
            "target_up_10",
        }
        return [column for column in df.columns if column not in excluded]
