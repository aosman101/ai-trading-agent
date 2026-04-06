from __future__ import annotations

import os
from unittest.mock import patch

import pytest

os.environ.setdefault("TRADING_MODE", "paper")

from app.risk.risk_manager import RiskManager
from app.types import EnsembleDecision


def _make_decision(**overrides) -> EnsembleDecision:
    defaults = dict(
        symbol="AAPL",
        direction="long",
        confidence=0.75,
        weighted_score=0.5,
        weights={"nhits": 0.2, "lightgbm": 0.2},
        contributions={"nhits": 0.1, "lightgbm": 0.1},
    )
    defaults.update(overrides)
    return EnsembleDecision(**defaults)


@pytest.fixture
def risk_manager():
    return RiskManager()


class TestBuildTradePlan:
    def test_approved_long_trade(self, risk_manager):
        decision = _make_decision(direction="long", confidence=0.75)
        plan = risk_manager.build_trade_plan(
            symbol="AAPL",
            decision=decision,
            price=150.0,
            atr=3.0,
            interval_width=2.0,
            equity=100_000.0,
            current_daily_pnl=0.0,
        )
        assert plan.approved is True
        assert plan.quantity > 0
        assert plan.stop_loss < plan.entry_price
        assert plan.take_profit > plan.entry_price
        assert plan.direction == "long"

    def test_approved_short_trade(self, risk_manager):
        risk_manager.settings.allow_shorting = True
        decision = _make_decision(direction="short", confidence=0.75)
        plan = risk_manager.build_trade_plan(
            symbol="AAPL",
            decision=decision,
            price=150.0,
            atr=3.0,
            interval_width=2.0,
            equity=100_000.0,
            current_daily_pnl=0.0,
        )
        assert plan.approved is True
        assert plan.stop_loss > plan.entry_price
        assert plan.take_profit < plan.entry_price

    def test_rejected_flat_direction(self, risk_manager):
        decision = _make_decision(direction="flat", confidence=0.75)
        plan = risk_manager.build_trade_plan(
            symbol="AAPL",
            decision=decision,
            price=150.0,
            atr=3.0,
            interval_width=2.0,
            equity=100_000.0,
            current_daily_pnl=0.0,
        )
        assert plan.approved is False
        assert "flat" in " ".join(plan.reasons).lower()

    def test_rejected_low_confidence(self, risk_manager):
        decision = _make_decision(direction="long", confidence=0.30)
        plan = risk_manager.build_trade_plan(
            symbol="AAPL",
            decision=decision,
            price=150.0,
            atr=3.0,
            interval_width=2.0,
            equity=100_000.0,
            current_daily_pnl=0.0,
        )
        assert plan.approved is False
        assert any("confidence" in r.lower() for r in plan.reasons)

    def test_rejected_daily_loss_limit(self, risk_manager):
        decision = _make_decision(direction="long", confidence=0.75)
        plan = risk_manager.build_trade_plan(
            symbol="AAPL",
            decision=decision,
            price=150.0,
            atr=3.0,
            interval_width=2.0,
            equity=100_000.0,
            current_daily_pnl=-5000.0,
        )
        assert plan.approved is False
        assert any("daily loss" in r.lower() for r in plan.reasons)

    def test_rejected_kill_switch(self, risk_manager):
        risk_manager.settings.kill_switch = True
        decision = _make_decision(direction="long", confidence=0.75)
        plan = risk_manager.build_trade_plan(
            symbol="AAPL",
            decision=decision,
            price=150.0,
            atr=3.0,
            interval_width=2.0,
            equity=100_000.0,
            current_daily_pnl=0.0,
        )
        assert plan.approved is False
        assert any("kill switch" in r.lower() for r in plan.reasons)
        risk_manager.settings.kill_switch = False

    def test_rejected_max_positions(self, risk_manager):
        decision = _make_decision(direction="long", confidence=0.75)
        open_positions = {f"SYM{i}": 10.0 for i in range(risk_manager.settings.max_open_positions)}
        plan = risk_manager.build_trade_plan(
            symbol="AAPL",
            decision=decision,
            price=150.0,
            atr=3.0,
            interval_width=2.0,
            equity=100_000.0,
            current_daily_pnl=0.0,
            open_positions=open_positions,
        )
        assert plan.approved is False
        assert any("positions" in r.lower() for r in plan.reasons)

    def test_shorting_disabled_rejects_short(self, risk_manager):
        risk_manager.settings.allow_shorting = False
        decision = _make_decision(direction="short", confidence=0.75)
        plan = risk_manager.build_trade_plan(
            symbol="AAPL",
            decision=decision,
            price=150.0,
            atr=3.0,
            interval_width=2.0,
            equity=100_000.0,
            current_daily_pnl=0.0,
        )
        assert plan.approved is False
        assert any("shorting" in r.lower() for r in plan.reasons)

    def test_portfolio_heat_cap(self, risk_manager):
        decision = _make_decision(direction="long", confidence=1.0)
        plan = risk_manager.build_trade_plan(
            symbol="PENNY",
            decision=decision,
            price=0.50,
            atr=0.01,
            interval_width=0.005,
            equity=100_000.0,
            current_daily_pnl=0.0,
        )
        assert plan.notional <= 100_000.0 * risk_manager.settings.max_portfolio_heat

    def test_notional_and_risk_amount_positive(self, risk_manager):
        decision = _make_decision(direction="long", confidence=0.75)
        plan = risk_manager.build_trade_plan(
            symbol="AAPL",
            decision=decision,
            price=150.0,
            atr=3.0,
            interval_width=2.0,
            equity=100_000.0,
            current_daily_pnl=0.0,
        )
        assert plan.notional >= 0
        assert plan.risk_amount >= 0
