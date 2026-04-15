from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable

from app.config import get_settings
from app.types import EnsembleDecision, ModelSignal, StrategySignal
from app.utils.math_utils import clamp, sigmoid


class EnsembleDecisionEngine:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.performance: dict[str, dict[str, deque[dict[str, float]]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.settings.model_performance_window))
        )

    @staticmethod
    def scope_key(symbol: str | None = None, regime: str | None = None) -> str:
        if symbol and regime:
            return f"symbol:{symbol}|regime:{regime}"
        if symbol:
            return f"symbol:{symbol}"
        if regime:
            return f"regime:{regime}"
        return "global"

    def scope_order(self, symbol: str | None = None, regime: str | None = None) -> list[str]:
        scopes: list[str] = []
        if symbol and regime:
            scopes.append(self.scope_key(symbol=symbol, regime=regime))
        if symbol:
            scopes.append(self.scope_key(symbol=symbol))
        if regime:
            scopes.append(self.scope_key(regime=regime))
        scopes.append("global")
        return scopes

    def prediction_scopes(self, symbol: str, regime: str | None = None) -> list[str]:
        scopes = ["global", self.scope_key(symbol=symbol)]
        if regime:
            scopes.append(self.scope_key(regime=regime))
            scopes.append(self.scope_key(symbol=symbol, regime=regime))
        return scopes

    def update_model_performance(
        self,
        model_name: str,
        metrics: dict[str, float],
        scope: str = "global",
    ) -> None:
        self.performance[scope][model_name].append(metrics)

    def _score_history(self, history: list[dict[str, float]]) -> float:
        if not history:
            return 1.0
        accuracy = sum(item.get("accuracy", 0.5) for item in history) / len(history)
        sharpe = sum(item.get("sharpe", 0.0) for item in history) / len(history)
        calibration = sum(item.get("calibration", 0.5) for item in history) / len(history)
        drawdown = sum(item.get("drawdown", 0.0) for item in history) / len(history)
        avg_edge = sum(item.get("avg_edge", 0.0) for item in history) / len(history)
        samples = sum(item.get("samples", 0) for item in history)

        score = (
            0.35 * clamp(accuracy, 0.0, 1.0)
            + 0.30 * clamp((sharpe + 1.0) / 3.0, 0.0, 1.0)
            + 0.15 * clamp(1.0 - calibration, 0.0, 1.0)
            + 0.10 * clamp(1.0 - drawdown, 0.0, 1.0)
            + 0.10 * clamp((avg_edge + 0.02) / 0.04, 0.0, 1.0)
        )

        # Penalise models with very few samples — don't let them dominate.
        if samples < 20:
            score *= clamp(samples / 20.0, 0.3, 1.0)

        return max(score, 0.05)

    def _resolved_scores(
        self,
        model_names: Iterable[str],
        symbol: str | None = None,
        regime: str | None = None,
    ) -> tuple[dict[str, float], dict[str, str]]:
        scores: dict[str, float] = {}
        resolved_scope: dict[str, str] = {}
        names = list(model_names)
        for name in names:
            resolved = False
            for scope in self.scope_order(symbol=symbol, regime=regime):
                history = list(self.performance.get(scope, {}).get(name, []))
                if not history:
                    continue
                scores[name] = self._score_history(history)
                resolved_scope[name] = scope
                resolved = True
                break
            if not resolved:
                scores[name] = 0.5
                resolved_scope[name] = "uniform"
        return scores, resolved_scope

    def current_weights(
        self,
        model_names: Iterable[str],
        symbol: str | None = None,
        regime: str | None = None,
    ) -> Dict[str, float]:
        names = list(model_names)
        scores, _ = self._resolved_scores(names, symbol=symbol, regime=regime)
        total = sum(scores.values()) or len(names) or 1.0
        return {name: value / total for name, value in scores.items()}

    @staticmethod
    def _signal_score(signal: ModelSignal | StrategySignal) -> float:
        direction_multiplier = {"long": 1.0, "short": -1.0, "flat": 0.0}.get(signal.direction, 0.0)
        if direction_multiplier == 0.0:
            return 0.0
        if isinstance(signal, StrategySignal):
            return float(direction_multiplier * clamp(signal.confidence, 0.0, 1.0))
        raw_score = abs(float(getattr(signal, "score", 0.0) or 0.0))
        confidence = clamp(float(getattr(signal, "confidence", 0.0) or 0.0), 0.0, 1.0)
        # Treat score as a magnitude (edge) and confidence as a reliability weight.
        magnitude = clamp(raw_score if raw_score > 0 else confidence, 0.0, 1.0)
        return float(direction_multiplier * magnitude * confidence)

    def combine(
        self,
        symbol: str,
        model_signals: list[ModelSignal],
        selected_strategy: StrategySignal | None = None,
        regime: str | None = None,
    ) -> EnsembleDecision:
        model_names = [signal.name for signal in model_signals]
        scores, resolved_scope = self._resolved_scores(model_names, symbol=symbol, regime=regime)
        total = sum(scores.values()) or len(model_names) or 1.0
        weights = {name: value / total for name, value in scores.items()}
        contributions = {
            signal.name: weights.get(signal.name, 0.0) * self._signal_score(signal)
            for signal in model_signals
        }

        if selected_strategy is not None:
            strategy_contribution = 0.10 * self._signal_score(selected_strategy)
            contributions[f"strategy:{selected_strategy.strategy}"] = strategy_contribution

        weighted_score = float(sum(contributions.values()))

        # Weighted agreement: each model's vote counts proportional to its ensemble weight.
        long_weight = 0.0
        short_weight = 0.0
        active_weight = 0.0
        for signal in model_signals:
            if signal.direction == "flat":
                continue
            weight = weights.get(signal.name, 0.0)
            active_weight += weight
            if signal.direction == "long":
                long_weight += weight
            elif signal.direction == "short":
                short_weight += weight
        n_active = sum(1 for s in model_signals if s.direction != "flat")
        if active_weight > 0:
            agreement_ratio = max(long_weight, short_weight) / active_weight
        else:
            agreement_ratio = 0.0

        # Scale confidence: boost when models agree, dampen when they conflict.
        raw_confidence = sigmoid(abs(weighted_score) * 3.0)
        if agreement_ratio >= 0.75:
            confidence = raw_confidence * clamp(0.90 + 0.10 * agreement_ratio, 0.90, 1.0)
        elif agreement_ratio <= 0.55 and n_active >= 3:
            # Strong disagreement — shrink confidence toward the flat threshold.
            confidence = raw_confidence * 0.70
        else:
            confidence = raw_confidence
        confidence = clamp(confidence, 0.0, 1.0)

        direction = "flat"
        if confidence >= self.settings.min_confidence_to_trade and weighted_score > 0:
            direction = "long"
        elif confidence >= self.settings.min_confidence_to_trade and weighted_score < 0:
            direction = "short"

        most_influential_model = None
        if contributions:
            most_influential_model = max(contributions, key=lambda key: abs(contributions[key]))

        primary_scope = None
        if resolved_scope:
            scope_preference = {scope: idx for idx, scope in enumerate(self.scope_order(symbol=symbol, regime=regime))}
            primary_scope = min(
                resolved_scope.values(),
                key=lambda scope: scope_preference.get(scope, len(scope_preference) + 1),
            )

        explanation = (
            f"Direction={direction}; score={weighted_score:.4f}; "
            f"influencer={most_influential_model or 'none'}; "
            f"scope={primary_scope or 'uniform'}"
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
            market_regime=regime,
            weight_scope=primary_scope,
            explanation=explanation,
        )
