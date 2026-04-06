from __future__ import annotations

import os

import pytest

os.environ.setdefault("TRADING_MODE", "paper")

from app.ensemble.decision_engine import EnsembleDecisionEngine
from app.types import ModelSignal, StrategySignal


def _model_signal(name: str, direction: str = "long", score: float = 0.5, confidence: float = 0.7) -> ModelSignal:
    return ModelSignal(name=name, symbol="AAPL", direction=direction, score=score, confidence=confidence)


def _strategy_signal(strategy: str = "momentum", direction: str = "long", confidence: float = 0.6) -> StrategySignal:
    return StrategySignal(strategy=strategy, symbol="AAPL", direction=direction, confidence=confidence)


@pytest.fixture
def engine():
    return EnsembleDecisionEngine()


class TestCurrentWeights:
    def test_equal_weights_with_no_history(self, engine):
        weights = engine.current_weights(["nhits", "lightgbm", "tft"])
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(abs(v - 1 / 3) < 1e-6 for v in weights.values())

    def test_weights_sum_to_one_with_history(self, engine):
        engine.update_model_performance("nhits", {"accuracy": 0.8, "sharpe": 1.5})
        engine.update_model_performance("lightgbm", {"accuracy": 0.5, "sharpe": 0.2})
        weights = engine.current_weights(["nhits", "lightgbm"])
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert weights["nhits"] > weights["lightgbm"]

    def test_scoped_weights_prefer_symbol_regime_then_fallback(self, engine):
        engine.update_model_performance(
            "nhits",
            {"accuracy": 0.9, "sharpe": 1.6, "calibration": 0.1, "drawdown": 0.05},
            scope="symbol:AAPL|regime:bull_trend_high_vol",
        )
        engine.update_model_performance(
            "lightgbm",
            {"accuracy": 0.55, "sharpe": 0.2, "calibration": 0.3, "drawdown": 0.20},
            scope="global",
        )
        weights = engine.current_weights(
            ["nhits", "lightgbm"],
            symbol="AAPL",
            regime="bull_trend_high_vol",
        )
        assert weights["nhits"] > weights["lightgbm"]

    def test_fallback_to_symbol_scope_when_regime_scope_missing(self, engine):
        engine.update_model_performance(
            "tft",
            {"accuracy": 0.8, "sharpe": 1.0, "calibration": 0.2, "drawdown": 0.1},
            scope="symbol:AAPL",
        )
        weights = engine.current_weights(["tft", "nhits"], symbol="AAPL", regime="range_low_vol")
        assert weights["tft"] > weights["nhits"]

    def test_minimum_weight_floor(self, engine):
        engine.update_model_performance("bad_model", {"accuracy": 0.0, "sharpe": -2.0, "drawdown": 1.0})
        weights = engine.current_weights(["bad_model", "good_model"])
        assert weights["bad_model"] > 0


class TestCombine:
    def test_long_decision_from_bullish_signals(self, engine):
        signals = [
            _model_signal("nhits", direction="long", score=0.5),
            _model_signal("lightgbm", direction="long", score=0.3),
            _model_signal("tft", direction="long", score=0.4),
        ]
        decision = engine.combine(symbol="AAPL", model_signals=signals)
        assert decision.direction == "long"
        assert decision.confidence > 0
        assert decision.weighted_score > 0

    def test_short_decision_from_bearish_signals(self, engine):
        signals = [
            _model_signal("nhits", direction="short", score=-0.5),
            _model_signal("lightgbm", direction="short", score=-0.3),
            _model_signal("tft", direction="short", score=-0.4),
        ]
        decision = engine.combine(symbol="AAPL", model_signals=signals)
        assert decision.direction == "short"
        assert decision.weighted_score < 0

    def test_flat_when_signals_conflict(self, engine):
        signals = [
            _model_signal("nhits", direction="long", score=0.3),
            _model_signal("lightgbm", direction="short", score=-0.3),
            _model_signal("tft", direction="flat", score=0.0),
        ]
        decision = engine.combine(symbol="AAPL", model_signals=signals)
        assert decision.direction == "flat"

    def test_strategy_contributes_to_decision(self, engine):
        signals = [_model_signal("nhits", direction="long", score=0.1, confidence=0.5)]
        strategy = _strategy_signal(strategy="momentum", direction="long", confidence=0.9)
        decision_with = engine.combine(symbol="AAPL", model_signals=signals, selected_strategy=strategy)
        decision_without = engine.combine(symbol="AAPL", model_signals=signals, selected_strategy=None)
        assert decision_with.weighted_score >= decision_without.weighted_score

    def test_most_influential_model_identified(self, engine):
        signals = [
            _model_signal("nhits", direction="long", score=0.9, confidence=0.9),
            _model_signal("lightgbm", direction="flat", score=0.0, confidence=0.1),
        ]
        decision = engine.combine(symbol="AAPL", model_signals=signals)
        assert decision.most_influential_model == "nhits"

    def test_explanation_populated(self, engine):
        signals = [_model_signal("nhits")]
        decision = engine.combine(symbol="AAPL", model_signals=signals)
        assert "Direction=" in decision.explanation
        assert "score=" in decision.explanation

    def test_combine_records_market_regime_and_scope(self, engine):
        engine.update_model_performance(
            "nhits",
            {"accuracy": 0.9, "sharpe": 1.4, "calibration": 0.1, "drawdown": 0.05},
            scope="regime:bull_trend_low_vol",
        )
        signals = [_model_signal("nhits")]
        decision = engine.combine(symbol="AAPL", model_signals=signals, regime="bull_trend_low_vol")
        assert decision.market_regime == "bull_trend_low_vol"
        assert decision.weight_scope == "regime:bull_trend_low_vol"
