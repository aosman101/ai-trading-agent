from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if returns.empty or returns.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=0))


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(abs(drawdown.min()))


def trade_returns_from_position(position: pd.Series, strategy_returns: pd.Series) -> list[float]:
    trade_returns = []
    active = False
    cumulative = 1.0
    current_sign = 0
    shifted = position.shift(-1).fillna(0.0)
    for idx in position.index:
        pos = float(position.loc[idx])
        ret = float(strategy_returns.loc[idx])
        if not active and pos != 0:
            active = True
            current_sign = int(np.sign(pos))
            cumulative = 1.0
        if active:
            cumulative *= 1.0 + ret
            next_pos = float(shifted.loc[idx])
            if next_pos == 0.0 or int(np.sign(next_pos)) != current_sign:
                trade_returns.append(cumulative - 1.0)
                active = False
                cumulative = 1.0
                current_sign = 0
    return trade_returns


def win_rate(position: pd.Series, strategy_returns: pd.Series) -> float:
    trades = trade_returns_from_position(position, strategy_returns)
    if not trades:
        return 0.0
    winners = sum(1 for item in trades if item > 0)
    return winners / len(trades)


def risk_reward_ratio(position: pd.Series, strategy_returns: pd.Series) -> float:
    trades = trade_returns_from_position(position, strategy_returns)
    wins = [item for item in trades if item > 0]
    losses = [abs(item) for item in trades if item < 0]
    if not wins or not losses:
        return 0.0
    return float(np.mean(wins) / np.mean(losses))


def summarize_performance(strategy_returns: pd.Series, position: pd.Series) -> dict[str, float]:
    returns = strategy_returns.fillna(0.0)
    equity_curve = (1.0 + returns).cumprod()
    total_return = float(equity_curve.iloc[-1] - 1.0) if not equity_curve.empty else 0.0
    return {
        "win_rate": win_rate(position.fillna(0.0), returns.fillna(0.0)),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
        "risk_reward": risk_reward_ratio(position.fillna(0.0), returns.fillna(0.0)),
        "total_return": total_return,
    }
