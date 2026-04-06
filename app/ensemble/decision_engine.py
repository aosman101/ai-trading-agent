from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable

from app.config import get_settings
from app.types import EnsembleDecision, ModelSignal, StrategySignal
from app.utils.math_utils import clamp, safe_div, sigmoid


class EnsembleDecisionEngine:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.performance: dict[str, deque[dict[str, float]]] = defaultdict(
            lambda: deque(maxlen=self.settings.model_performance_window)
        )

    def update_model_performance(self, model_name: str, metrics: dict[str, float]) -> None:
        self.performance[model_name].append(metrics)

    def current_weights(self, model_names: Iterable[str]) -> Dict[str, float]:
        scores: dict[str, float] = {}
        names = list(model_names)
        for name in names:
            history = list(self.performance.get(name, []))
            if not history:
                scores[name] = 1.0
                continue
            accuracy = sum(item.get("accuracy", 0.5) for item in history) / len(history)
            sharpe = sum(item.get("sharpe", 0.0) for item in history) / len(history)
            calibration = sum(item.get("calibration", 0.5) for item in history) / len(history)
            drawdown = sum(item.get("drawdown", 0.0) for item in history) / len(history)
            score = (
                0.40 * clamp(accuracy, 0.0, 1.0)
                + 0.35 * clamp((sharpe + 1.0) / 3.0, 0.0, 1.0)
                + 0.15 * clamp(1.0 - calibration, 0.0, 1.0)
                + 0.10 * clamp(1.0 - drawdown, 0.0, 1.0)
            )
            scores[name] = max(score, 0.05)
        total = sum(scores.values()) or len(names) or 1.0
        return {name: value / total for name, value in scores.items()}

    @staticmethod
    def _signal_score(signal: ModelSignal | StrategySignal) -> float:
        direction_multiplier = {"long": 1.0, "short": -1.0, "flat": 0.0}.get(signal.direction, 0.0)
        base = direction_multiplier * max(abs(getattr(signal, "score", 0.0)), getattr(signal, "confidence", 0.0))
        if isinstance(signal, StrategySignal):
            base = direction_multiplier * signal.confidence
        return float(base)

    def combine(
        self,
        symbol: str,
        model_signals: list[ModelSignal],
        selected_strategy: StrategySignal | None = None,
    ) -> EnsembleDecision:
        model_names = [signal.name for signal in model_signals]
        weights = self.current_weights(model_names)
        contributions = {
            signal.name: weights.get(signal.name, 0.0) * self._signal_score(signal)
            for signal in model_signals
        }

        if selected_strategy is not None:
            strategy_contribution = 0.10 * self._signal_score(selected_strategy)
            contributions[f"strategy:{selected_strategy.strategy}"] = strategy_contribution

        weighted_score = float(sum(contributions.values()))
        confidence = sigmoid(abs(weighted_score) * 3.0)

        direction = "flat"
        if confidence >= self.settings.min_confidence_to_trade and weighted_score > 0:
            direction = "long"
        elif confidence >= self.settings.min_confidence_to_trade and weighted_score < 0:
            direction = "short"

        most_influential_model = None
        if contributions:
            most_influential_model = max(contributions, key=lambda key: abs(contributions[key]))

        explanation = (
            f"Direction={direction}; score={weighted_score:.4f}; "
            f"influencer={most_influential_model or 'none'}"
        )

        return EnsembleDecision(
            symbol=symbol,
            direction=direction,
            confidence=float(confidence),
            weighted_score=weighted_score,
            weights=weights,
            contributions=contributions,
            selected_strategy=selected_strategy.strategy if selected_strategy else None,
            most_influential_model=most_influential_model,
            explanation=explanation,
        )
