from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtesting.metrics import (
    max_drawdown,
    risk_reward_ratio,
    sharpe_ratio,
    summarize_performance,
    trade_returns_from_position,
    win_rate,
)


class TestSharpeRatio:
    def test_positive_sharpe(self):
        returns = pd.Series([0.01, 0.02, 0.005, 0.01, -0.005] * 50)
        result = sharpe_ratio(returns)
        assert result > 0

    def test_negative_sharpe(self):
        returns = pd.Series([-0.01, -0.02, -0.005, -0.01, 0.005] * 50)
        result = sharpe_ratio(returns)
        assert result < 0

    def test_zero_std_returns_zero(self):
        returns = pd.Series([0.0, 0.0, 0.0])
        assert sharpe_ratio(returns) == 0.0

    def test_empty_returns_zero(self):
        returns = pd.Series([], dtype=float)
        assert sharpe_ratio(returns) == 0.0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        equity = pd.Series([1.0, 1.1, 1.2, 1.3])
        assert max_drawdown(equity) == 0.0

    def test_known_drawdown(self):
        equity = pd.Series([1.0, 1.2, 0.9, 1.1])
        dd = max_drawdown(equity)
        assert abs(dd - 0.25) < 1e-6  # 0.9 / 1.2 - 1 = -0.25

    def test_full_drawdown(self):
        equity = pd.Series([1.0, 0.5, 0.0])
        assert max_drawdown(equity) == 1.0


class TestTradeReturns:
    def test_single_round_trip(self):
        position = pd.Series([0.0, 1.0, 1.0, 0.0])
        returns = pd.Series([0.0, 0.01, 0.02, 0.0])
        trades = trade_returns_from_position(position, returns)
        assert len(trades) == 1
        assert trades[0] > 0

    def test_no_trades(self):
        position = pd.Series([0.0, 0.0, 0.0])
        returns = pd.Series([0.01, 0.02, -0.01])
        trades = trade_returns_from_position(position, returns)
        assert len(trades) == 0

    def test_multiple_trades(self):
        position = pd.Series([0, 1, 1, 0, -1, -1, 0.0])
        returns = pd.Series([0, 0.01, 0.02, 0, -0.01, -0.02, 0.0])
        trades = trade_returns_from_position(position, returns)
        assert len(trades) == 2


class TestWinRate:
    def test_all_winners(self):
        position = pd.Series([0, 1, 1, 0, 1, 1, 0.0])
        returns = pd.Series([0, 0.02, 0.01, 0, 0.03, 0.01, 0.0])
        assert win_rate(position, returns) == 1.0

    def test_no_trades_returns_zero(self):
        position = pd.Series([0, 0, 0.0])
        returns = pd.Series([0.01, 0.02, 0.0])
        assert win_rate(position, returns) == 0.0


class TestRiskRewardRatio:
    def test_no_losses_returns_zero(self):
        position = pd.Series([0, 1, 1, 0.0])
        returns = pd.Series([0, 0.05, 0.03, 0.0])
        assert risk_reward_ratio(position, returns) == 0.0

    def test_symmetric_trades(self):
        position = pd.Series([0, 1, 0, 1, 0.0])
        returns = pd.Series([0, 0.02, 0, -0.02, 0.0])
        ratio = risk_reward_ratio(position, returns)
        assert abs(ratio - 1.0) < 0.1


class TestSummarizePerformance:
    def test_returns_expected_keys(self):
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.005])
        position = pd.Series([1.0, 1.0, -1.0, -1.0, 0.0])
        metrics = summarize_performance(returns, position)
        expected_keys = {"win_rate", "sharpe", "max_drawdown", "risk_reward", "total_return"}
        assert set(metrics.keys()) == expected_keys

    def test_total_return_calculation(self):
        returns = pd.Series([0.10])
        position = pd.Series([1.0])
        metrics = summarize_performance(returns, position)
        assert abs(metrics["total_return"] - 0.10) < 1e-6
