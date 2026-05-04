from __future__ import annotations

import pandas as pd
import pytest

from app.config import get_settings
from app.ensemble.decision_memory import DecisionMemory
from app.types import EnsembleDecision, RiskPlan


class FakeMemoryRepository:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def decision_memory_exists(self, symbol: str, trade_date: str) -> bool:
        return any(row.get("symbol") == symbol and row.get("trade_date") == trade_date for row in self.rows)

    def store_decision_memory(self, record: dict) -> None:
        if not self.decision_memory_exists(record["symbol"], record["trade_date"]):
            self.rows.append({**record, "_local_id": len(self.rows) + 1})

    def pending_decision_memory(self, symbol: str | None = None, limit: int = 200):
        rows = [row for row in self.rows if row.get("status") == "pending"]
        if symbol:
            rows = [row for row in rows if row.get("symbol") == symbol]
        return rows[:limit]

    def recent_decision_memory(self, limit: int = 200, symbol: str | None = None, status: str | None = "resolved"):
        rows = list(self.rows)
        if symbol:
            rows = [row for row in rows if row.get("symbol") == symbol]
        if status:
            rows = [row for row in rows if row.get("status") == status]
        rows.sort(key=lambda row: row.get("trade_date", ""), reverse=True)
        return rows[:limit]

    def update_decision_memory_outcome(self, entry: dict, updates: dict) -> None:
        for row in self.rows:
            if row.get("_local_id") == entry.get("_local_id"):
                row.update(updates)
                return


def _frame(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=len(closes))
    return pd.DataFrame({"symbol": symbol, "close": closes}, index=dates)


def test_store_pending_is_idempotent():
    repo = FakeMemoryRepository()
    memory = DecisionMemory(repo)
    decision = EnsembleDecision(symbol="AAPL", direction="long", rating="overweight", confidence=0.7)
    risk_plan = RiskPlan(symbol="AAPL", direction="long")

    for _ in range(2):
        memory.store_pending(
            symbol="AAPL",
            trade_date="2024-01-02",
            decision=decision,
            risk_plan=risk_plan,
            model_signals=[],
            selected_strategy=None,
            entry_price=100.0,
            benchmark_symbol="SPY",
            benchmark_entry_price=100.0,
        )

    assert len(repo.rows) == 1
    assert repo.rows[0]["status"] == "pending"


def test_resolve_pending_writes_reflection_and_returns_alpha(monkeypatch):
    repo = FakeMemoryRepository()
    memory = DecisionMemory(repo)
    monkeypatch.setattr(memory.settings, "decision_memory_holding_bars", 2)
    decision = EnsembleDecision(symbol="AAPL", direction="long", rating="overweight", confidence=0.7)

    memory.store_pending(
        symbol="AAPL",
        trade_date="2024-01-02",
        decision=decision,
        risk_plan=RiskPlan(symbol="AAPL", direction="long"),
        model_signals=[],
        selected_strategy=None,
        entry_price=100.0,
        benchmark_symbol="SPY",
        benchmark_entry_price=100.0,
    )

    resolved = memory.resolve_pending(
        "AAPL",
        _frame("AAPL", [100.0, 99.0, 98.0, 101.0]),
        _frame("SPY", [100.0, 100.0, 100.0, 100.0]),
    )

    assert len(resolved) == 1
    assert repo.rows[0]["status"] == "resolved"
    assert repo.rows[0]["raw_return"] == pytest.approx(-0.02)
    assert repo.rows[0]["alpha_return"] == pytest.approx(-0.02)
    assert repo.rows[0]["decision_alpha"] == pytest.approx(-0.02)
    assert "directional call failed" in repo.rows[0]["reflection"].lower()


def test_negative_memory_can_dampen_future_decision_below_threshold(monkeypatch):
    repo = FakeMemoryRepository()
    settings = get_settings()
    monkeypatch.setattr(settings, "decision_memory_min_samples", 1)
    monkeypatch.setattr(settings, "decision_memory_alpha_threshold", 0.005)
    repo.rows.append(
        {
            "_local_id": 1,
            "status": "resolved",
            "symbol": "AAPL",
            "trade_date": "2024-01-02",
            "direction": "long",
            "rating": "overweight",
            "market_regime": "bull_trend_low_vol",
            "decision_alpha": -0.02,
            "reflection": "Long setup failed.",
        }
    )
    memory = DecisionMemory(repo)
    decision = EnsembleDecision(
        symbol="AAPL",
        direction="long",
        rating="overweight",
        confidence=0.60,
        market_regime="bull_trend_low_vol",
    )

    assessment = memory.assess_decision(decision)
    adjusted = memory.apply_assessment(decision, assessment)

    assert assessment["confidence_multiplier"] < 1.0
    assert adjusted.direction == "flat"
    assert adjusted.rating == "hold"
    assert any("decision memory" in flag.lower() for flag in adjusted.risk_flags)
