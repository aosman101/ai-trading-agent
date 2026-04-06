from __future__ import annotations

import pytest

from app.types import (
    EnsembleDecision,
    LearningEvent,
    ModelSignal,
    RiskPlan,
    StrategySignal,
    TradeRecord,
)


class TestModelSignal:
    def test_defaults(self):
        sig = ModelSignal(name="test", symbol="AAPL")
        assert sig.direction == "flat"
        assert sig.score == 0.0
        assert sig.confidence == 0.0

    def test_valid_directions(self):
        for d in ("long", "short", "flat"):
            sig = ModelSignal(name="test", symbol="AAPL", direction=d)
            assert sig.direction == d

    def test_invalid_direction_rejected(self):
        with pytest.raises(Exception):
            ModelSignal(name="test", symbol="AAPL", direction="sideways")


class TestRiskPlan:
    def test_defaults(self):
        plan = RiskPlan(symbol="AAPL")
        assert plan.approved is False
        assert plan.quantity == 0
        assert plan.reasons == []

    def test_serialization_roundtrip(self):
        plan = RiskPlan(
            approved=True,
            symbol="AAPL",
            direction="long",
            quantity=10,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
        )
        data = plan.model_dump()
        restored = RiskPlan(**data)
        assert restored.symbol == "AAPL"
        assert restored.quantity == 10


class TestEnsembleDecision:
    def test_model_dump_keys(self):
        decision = EnsembleDecision(symbol="AAPL")
        data = decision.model_dump()
        assert "symbol" in data
        assert "direction" in data
        assert "weights" in data
        assert "contributions" in data
