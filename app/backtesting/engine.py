from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from app.backtesting.metrics import summarize_performance
from app.config import get_settings
from app.data.market_data import MarketDataService
from app.strategies.breakout import BreakoutStrategy
from app.strategies.mean_reversion import MeanReversionStrategy
from app.strategies.momentum import MomentumStrategy
from app.strategies.sentiment_strategy import SentimentStrategy
from app.strategies.trend_following import TrendFollowingStrategy
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    metrics: dict[str, float]
    equity_curve: pd.Series
    returns: pd.Series
    position: pd.Series


class WalkForwardBacktester:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.market_data = MarketDataService()
        self.strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            TrendFollowingStrategy(),
            BreakoutStrategy(),
            SentimentStrategy(),
        ]

    @staticmethod
    def _estimate_slippage(
        position_change: pd.Series,
        volatility: pd.Series,
        base_slippage_bps: float,
    ) -> pd.Series:
        vol_multiplier = (volatility / volatility.rolling(60).median()).clip(0.5, 3.0).fillna(1.0)
        size_multiplier = position_change.abs().clip(0.0, 1.0)
        return (base_slippage_bps / 10_000.0) * vol_multiplier * size_multiplier

    def _evaluation_slices(self, n_rows: int) -> list[slice]:
        min_history = max(30, self.settings.backtest_min_history_bars)
        test_window = max(10, self.settings.backtest_test_window_bars)
        if n_rows <= min_history:
            return [slice(0, n_rows)]

        windows: list[slice] = []
        start = min_history
        while start < n_rows:
            end = min(start + test_window, n_rows)
            windows.append(slice(start, end))
            if end >= n_rows:
                break
            start += test_window
        return windows or [slice(0, n_rows)]

    def _simulate_strategy(
        self,
        frame: pd.DataFrame,
        signal_series: pd.Series,
    ) -> BacktestResult:
        position = signal_series.shift(1).fillna(0.0).clip(lower=-1.0, upper=1.0)
        asset_returns = frame["close"].pct_change().fillna(0.0)
        position_change = position.diff().fillna(0.0)

        fee_cost = position_change.abs() * (self.settings.fee_bps / 10_000.0)

        volatility = asset_returns.rolling(20).std().fillna(asset_returns.std())
        slippage_cost = self._estimate_slippage(
            position_change, volatility, self.settings.slippage_bps,
        )

        costs = fee_cost + slippage_cost
        strategy_returns = position * asset_returns - costs
        equity_curve = (1.0 + strategy_returns).cumprod()
        metrics = summarize_performance(strategy_returns, position)
        return BacktestResult(
            symbol=str(frame["symbol"].iloc[-1]),
            strategy="",
            metrics=metrics,
            equity_curve=equity_curve,
            returns=strategy_returns,
            position=position,
        )

    def run_for_symbol(
        self,
        symbol: str,
        sentiment_score: float | pd.Series = 0.0,
    ) -> Dict[str, BacktestResult]:
        frame = self.market_data.fetch_symbol_history(symbol)
        evaluation_slices = self._evaluation_slices(len(frame))
        results: dict[str, BacktestResult] = {}
        for strategy in self.strategies:
            signals = strategy.generate_series(frame, sentiment_score=sentiment_score)
            returns_segments: list[pd.Series] = []
            position_segments: list[pd.Series] = []
            for evaluation_slice in evaluation_slices:
                segment_frame = frame.iloc[evaluation_slice].copy()
                segment_signals = signals.iloc[evaluation_slice].copy()
                segment_result = self._simulate_strategy(segment_frame, segment_signals)
                returns_segments.append(segment_result.returns)
                position_segments.append(segment_result.position)

            strategy_returns = pd.concat(returns_segments).sort_index()
            position = pd.concat(position_segments).sort_index()
            equity_curve = (1.0 + strategy_returns).cumprod()
            metrics = summarize_performance(strategy_returns, position)
            result = BacktestResult(
                symbol=str(frame["symbol"].iloc[-1]),
                strategy=strategy.name,
                metrics=metrics,
                equity_curve=equity_curve,
                returns=strategy_returns,
                position=position,
            )
            result.strategy = strategy.name
            results[strategy.name] = result
        return results

    def run_all(self, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        symbols = list(symbols or self.settings.symbols)
        rows: List[dict[str, float | str]] = []
        for symbol in symbols:
            logger.info("Running backtests for %s", symbol)
            symbol_results = self.run_for_symbol(symbol)
            for strategy_name, result in symbol_results.items():
                row = {"symbol": symbol, "strategy": strategy_name}
                row.update(result.metrics)
                rows.append(row)
        return pd.DataFrame(rows)

    def aggregate_strategy_metrics(self, symbols: Iterable[str] | None = None) -> Dict[str, Dict[str, float]]:
        report = self.run_all(symbols=symbols)
        if report.empty:
            return {}
        grouped = report.groupby("strategy").mean(numeric_only=True)
        return grouped.to_dict(orient="index")
