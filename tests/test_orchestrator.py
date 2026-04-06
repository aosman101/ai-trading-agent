from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

os.environ.setdefault("TRADING_MODE", "paper")

for module_name in ("feedparser", "gymnasium", "httpx", "yfinance"):
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()

from app.config import get_settings
from app.orchestrator import TradingOrchestrator
from app.types import ModelSignal, StrategySignal


def _history_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=6)
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 102, 104, 106, 108, 110],
            "adj_close": [100, 102, 104, 106, 108, 110],
            "volume": [1_000_000] * 6,
            "atr_14": [2.0] * 6,
            "feature_alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feature_beta": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "forward_return_1": [0.02, 0.0196, 0.0192, 0.0189, 0.0185, 0.0],
            "forward_return_5": [0.0] * 6,
            "forward_return_10": [0.0] * 6,
            "target_up_1": [1, 1, 1, 1, 1, 0],
            "target_up_5": [0] * 6,
            "target_up_10": [0] * 6,
            "symbol": ["AAPL"] * 6,
        },
        index=dates,
    )


class FakeMarketData:
    def fetch_symbol_history(self, symbol: str):
        return _history_frame()

    def latest_feature_row(self, symbol: str):
        return _history_frame().iloc[-1]


class FakeNewsData:
    def collect_text_corpus(self, symbol: str):
        return ["bullish setup"]


class FakeRepository:
    def __init__(self):
        self.runtime_state = {
            "live_state": {
                "reference_equity": 100_000.0,
                "max_equity": 110_000.0,
            }
        }
        self.predictions_logged = []
        self.trades_logged = []
        self.equity_logged = []
        self.learning_events = []
        self.weights_saved = []

    def recent_predictions(self, limit: int = 100):
        return [
            {
                "created_at": "2024-01-04T16:00:00+00:00",
                "symbol": "AAPL",
                "payload": {
                    "model_signals": [
                        {"name": "nhits", "direction": "long", "confidence": 0.8},
                        {"name": "tft", "direction": "short", "confidence": 0.8},
                    ]
                },
            }
        ]

    def recent_trades(self, limit: int = 100):
        return []

    def log_prediction(self, record):
        self.predictions_logged.append(record)

    def save_model_weights(self, record):
        self.weights_saved.append(record)

    def log_equity(self, record):
        self.equity_logged.append(record)

    def log_trade(self, record):
        self.trades_logged.append(record)

    def log_learning_event(self, record):
        self.learning_events.append(record)

    def write_runtime_state(self, state_key: str, payload):
        self.runtime_state[state_key] = payload

    def read_runtime_state(self, state_key: str):
        return self.runtime_state.get(state_key, {})


class FakeBroker:
    def __init__(self):
        self.orders = []

    def account_equity(self):
        return 110_000.0

    def day_pnl(self):
        return 750.0

    def list_open_positions(self):
        return {"AAPL": 10.0}

    def place_bracket_order(self, trade_plan):
        self.orders.append(trade_plan)
        return {"id": "order-1", "status": "submitted"}


class FakeStrategy:
    def __init__(self, name: str):
        self.name = name

    def generate_series(self, frame, sentiment_score: float = 0.0):
        return pd.Series([1.0] * len(frame), index=frame.index)

    def generate_latest(self, frame, sentiment_score: float = 0.0):
        return StrategySignal(
            strategy=self.name,
            symbol="AAPL",
            direction="long",
            confidence=0.9,
        )


class FakeRLAgent:
    def __init__(self, name: str, direction: str):
        self.name = name
        self.direction = direction
        self.last_observation = None

    def predict(self, observation, symbol: str):
        self.last_observation = observation
        score = 1.0 if self.direction == "long" else -1.0 if self.direction == "short" else 0.0
        confidence = 0.65 if score else 0.3
        return ModelSignal(
            name=self.name,
            symbol=symbol,
            direction=self.direction,
            score=score,
            confidence=confidence,
            metadata={"action": 1 if self.direction == "long" else 2 if self.direction == "short" else 0},
        )


class FakePredictor:
    def __init__(self, name: str, direction: str, score: float, confidence: float, metadata=None):
        self.name = name
        self.direction = direction
        self.score = score
        self.confidence = confidence
        self.metadata = metadata or {}

    def predict_latest(self, *args, **kwargs):
        symbol = kwargs.get("symbol")
        if symbol is None:
            for arg in args:
                if isinstance(arg, str):
                    symbol = arg
                    break
        symbol = symbol or "AAPL"
        return ModelSignal(
            name=self.name,
            symbol=symbol,
            direction=self.direction,
            score=self.score,
            confidence=self.confidence,
            metadata=self.metadata,
        )


class FakeBacktester:
    def run_for_symbol(self, symbol: str):
        metrics = {
            "momentum": {
                "sharpe": 1.5,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_return": 0.12,
            }
        }
        return {
            "momentum": SimpleNamespace(metrics=metrics["momentum"]),
        }


def _build_orchestrator() -> TradingOrchestrator:
    orchestrator = TradingOrchestrator.__new__(TradingOrchestrator)
    orchestrator.settings = get_settings()
    orchestrator.market_data = FakeMarketData()
    orchestrator.news_data = FakeNewsData()
    orchestrator.repository = FakeRepository()
    orchestrator.decision_engine = __import__("app.ensemble.decision_engine", fromlist=["EnsembleDecisionEngine"]).EnsembleDecisionEngine()
    orchestrator.strategy_selector = __import__("app.strategies.selector", fromlist=["StrategySelector"]).StrategySelector()
    orchestrator.risk_manager = __import__("app.risk.risk_manager", fromlist=["RiskManager"]).RiskManager()
    orchestrator.broker = FakeBroker()
    orchestrator.rule_strategies = [FakeStrategy("momentum")]
    orchestrator._model_lock = nullcontext()
    orchestrator.models = SimpleNamespace(
        nhits=FakePredictor("nhits", "long", 0.9, 0.9),
        lightgbm=FakePredictor("lightgbm", "long", 0.4, 0.6),
        tft=FakePredictor("tft", "short", -0.2, 0.7, metadata={"interval_width": 1.0}),
        finbert=FakePredictor("finbert", "long", 0.2, 0.6),
        ppo=FakeRLAgent("ppo", "long"),
        dqn=FakeRLAgent("dqn", "long"),
        itransformer=None,
        backtester=FakeBacktester(),
    )
    return orchestrator


def test_run_cycle_for_symbol_uses_live_state_and_updates_runtime_state():
    orchestrator = _build_orchestrator()

    result = orchestrator.run_cycle_for_symbol("AAPL")

    assert result["decision"]["direction"] == "long"
    assert result["decision"]["weights"]["nhits"] > result["decision"]["weights"]["tft"]
    observation = orchestrator.models.ppo.last_observation
    assert observation is not None
    assert observation[-3] == pytest.approx(1.0)
    assert observation[-2] == pytest.approx(1.1)
    assert observation[-1] == pytest.approx(0.0)
    assert "worker_status" not in orchestrator.repository.runtime_state
    assert orchestrator.repository.runtime_state["live_state"]["current_portfolio_heat"] > 0
    assert orchestrator.repository.runtime_state["model_performance"]["nhits"]["accuracy"] > 0
    assert orchestrator.broker.orders
