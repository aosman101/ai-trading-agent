from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, Optional

from app.config import get_settings
from app.types import StrategySignal
from app.utils.logging import get_logger

logger = get_logger(__name__)


class StrategySelector:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.performance: dict[str, deque[dict[str, float]]] = defaultdict(
            lambda: deque(maxlen=self.settings.strategy_performance_window)
        )

    def update_performance(self, strategy_name: str, metrics: dict[str, float]) -> None:
        self.performance[strategy_name].append(metrics)

    def _score(self, metrics: dict[str, float]) -> float:
        sharpe = metrics.get("sharpe", 0.0)
        win_rate = metrics.get("win_rate", 0.0)
        total_return = metrics.get("total_return", 0.0)
        drawdown = metrics.get("max_drawdown", 1.0)
        return 0.40 * sharpe + 0.30 * win_rate + 0.20 * total_return - 0.30 * drawdown

    def select_best(
        self,
        strategy_signals: Iterable[StrategySignal],
        latest_backtest_metrics: Dict[str, Dict[str, float]],
    ) -> Optional[StrategySignal]:
        candidates = []
        for signal in strategy_signals:
            metrics = latest_backtest_metrics.get(signal.strategy, {})
            if not metrics:
                continue
            if metrics.get("sharpe", 0.0) < self.settings.min_sharpe_to_deploy:
                continue
            if metrics.get("max_drawdown", 1.0) > self.settings.max_drawdown_to_deploy:
                continue
            if metrics.get("win_rate", 0.0) < self.settings.min_win_rate_to_deploy:
                continue
            if metrics.get("total_return", 0.0) < self.settings.min_total_return_to_deploy:
                continue
            score = self._score(metrics) * max(signal.confidence, 0.1)
            candidates.append((score, signal))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_signal = candidates[0][1]
        logger.info("Selected strategy %s for %s", best_signal.strategy, best_signal.symbol)
        return best_signal
