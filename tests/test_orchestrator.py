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

    def sentiment_time_series(self, symbol: str, index, fallback_latest=None, limit: int = 100):
        series = pd.Series(0.0, index=index, dtype=float)
        if fallback_latest is not None and len(series):
            series.iloc[-1] = float(fallback_latest)
        return series


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
        self.journal_entries = []
        self._external_signals: list[dict] = []
        self._consumed_signal_ids: set = set()

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

    def log_journal_entry(self, record):
        self.journal_entries.append(record)

    def write_runtime_state(self, state_key: str, payload):
        self.runtime_state[state_key] = payload

    def read_runtime_state(self, state_key: str):
        return self.runtime_state.get(state_key, {})

    def pending_external_signals(self, symbol: str, limit: int = 20):
        return [
            s for s in self._external_signals
            if s.get("symbol") == symbol and s.get("id") not in self._consumed_signal_ids
        ][:limit]

    def mark_signals_consumed(self, signal_ids):
        self._consumed_signal_ids.update(str(sid) for sid in signal_ids)


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
    def run_for_symbol(self, symbol: str, sentiment_score=0.0, frame=None):
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


class FakeDSIClient:
    configured = True

    def fetch_all_signals(self, symbol: str):
        return [
            ModelSignal(name="nhits", symbol=symbol, direction="long", score=0.02, confidence=0.7,
                        metadata={"source": "dsi", "current_price": 110.0, "predicted_close": 112.0,
                                  "signal_strength": 0.6, "stop_loss": 107.0, "take_profit": 116.0,
                                  "prediction_horizon": "1d"}),
            ModelSignal(name="tft", symbol=symbol, direction="long", score=0.015, confidence=0.65,
                        metadata={"source": "dsi", "current_price": 110.0, "predicted_close": 111.5,
                                  "signal_strength": 0.55, "stop_loss": 108.0, "take_profit": 115.0,
                                  "prediction_horizon": "1d"}),
            ModelSignal(name="lightgbm", symbol=symbol, direction="long", score=0.1, confidence=0.6,
                        metadata={"source": "dsi", "current_price": 110.0, "predicted_close": 111.0,
                                  "signal_strength": 0.5, "stop_loss": None, "take_profit": None,
                                  "prediction_horizon": "1d"}),
        ]


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
    orchestrator.dsi_client = SimpleNamespace(configured=False)
    orchestrator.models = SimpleNamespace(
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
    assert orchestrator.repository.runtime_state["model_performance"]["global"]["nhits"]["accuracy"] > 0
    assert result["decision"]["market_regime"] is not None
    assert result["decision"]["weight_scope"] is not None
    assert orchestrator.broker.orders


def test_run_cycle_for_symbol_uses_dsi_signals_when_available():
    orchestrator = _build_orchestrator()
    orchestrator.dsi_client = FakeDSIClient()

    result = orchestrator.run_cycle_for_symbol("AAPL")

    assert result["decision"]["direction"] == "long"
    assert result["status_snapshot"]["dsi_status"]["configured"] is True
    assert result["status_snapshot"]["dsi_status"]["missing_models"] == []
    logged = orchestrator.repository.predictions_logged[0]
    dsi_signals = {
        signal["name"]: signal
        for signal in logged["payload"]["model_signals"]
        if signal["name"] in {"nhits", "tft", "lightgbm"}
    }
    assert dsi_signals["nhits"]["metadata"]["source"] == "dsi"
    assert dsi_signals["tft"]["metadata"]["predicted_close"] == pytest.approx(111.5)
    assert dsi_signals["lightgbm"]["confidence"] == pytest.approx(0.6)


def test_orchestrator_continues_when_dsi_unavailable():
    class EmptyDSIClient:
        configured = True

        def fetch_all_signals(self, symbol: str):
            return []

    orchestrator = _build_orchestrator()
    orchestrator.dsi_client = EmptyDSIClient()

    result = orchestrator.run_cycle_for_symbol("AAPL")

    assert result["decision"]["direction"] in {"long", "short", "flat"}
    assert result["status_snapshot"]["dsi_status"]["available"] is False
    logged = orchestrator.repository.predictions_logged[0]
    fallback_signals = {
        signal["name"]: signal
        for signal in logged["payload"]["model_signals"]
        if signal["name"] in {"nhits", "tft", "lightgbm"}
    }
    assert fallback_signals["nhits"]["direction"] == "flat"
    assert fallback_signals["tft"]["confidence"] == pytest.approx(0.0)
    assert fallback_signals["lightgbm"]["score"] == pytest.approx(0.0)


def test_run_cycle_for_symbol_raises_on_empty_history():
    orchestrator = _build_orchestrator()

    class EmptyMarketData:
        def fetch_symbol_history(self, symbol: str):
            return pd.DataFrame()

        def latest_feature_row(self, symbol: str):
            return pd.Series()

    orchestrator.market_data = EmptyMarketData()
    with pytest.raises(ValueError, match="No market data"):
        orchestrator.run_cycle_for_symbol("AAPL")


def test_build_live_rl_observation_raises_on_all_nan():
    orchestrator = _build_orchestrator()
    frame = _history_frame().reset_index(names="ds")
    # Set all feature columns to NaN
    for col in frame.columns:
        if col not in {"symbol", "ds", "open", "high", "low", "close", "adj_close",
                       "volume", "forward_return_1", "forward_return_5", "forward_return_10",
                       "target_up_1", "target_up_5", "target_up_10"}:
            frame[col] = float("nan")
    live_state = {"current_position": 0.0, "portfolio_value": 1.0, "drawdown": 0.0}
    with pytest.raises(ValueError, match="No valid rows"):
        orchestrator._build_live_rl_observation(frame, live_state)


def test_prediction_outcome_multi_horizon():
    orchestrator = _build_orchestrator()
    frame = _history_frame().reset_index(names="ds")
    # Use a date that falls on the 2nd row (2024-01-03)
    result = orchestrator._prediction_outcome(frame, "2024-01-03T16:00:00+00:00")
    assert result is not None
    # Should be a blended return, not just single-bar
    single_bar_return = (106 / 104) - 1.0  # bars 2->3
    assert result != pytest.approx(single_bar_return, abs=1e-6)


def test_market_regime_empty_history():
    orchestrator = _build_orchestrator()
    result = orchestrator._market_regime(pd.DataFrame())
    assert result == "range_low_vol"


def test_run_cycle_logs_error_on_symbol_failure():
    orchestrator = _build_orchestrator()

    class FailingMarketData:
        def fetch_symbol_history(self, symbol: str):
            raise RuntimeError("API down")

        def latest_feature_row(self, symbol: str):
            raise RuntimeError("API down")

    orchestrator.market_data = FailingMarketData()
    results = orchestrator.run_cycle(["AAPL"])
    assert results == []
    assert len(orchestrator.repository.learning_events) == 1
    assert orchestrator.repository.learning_events[0]["event_type"] == "cycle_error"


# ---------------------------------------------------------------------------
# External signals integration
# ---------------------------------------------------------------------------


def test_external_signal_is_consumed_and_included_in_ensemble():
    orchestrator = _build_orchestrator()
    orchestrator.repository._external_signals = [
        {
            "id": 1,
            "symbol": "AAPL",
            "direction": "long",
            "score": 0.8,
            "confidence": 0.9,
            "source": "website",
            "reasoning": "Strong breakout pattern on 4h chart",
        }
    ]

    result = orchestrator.run_cycle_for_symbol("AAPL")

    # External signal should appear in the logged prediction payload
    logged = orchestrator.repository.predictions_logged[0]
    signal_names = [s["name"] for s in logged["payload"]["model_signals"]]
    assert "external:website" in signal_names

    # Signal should be marked as consumed
    assert "1" in orchestrator.repository._consumed_signal_ids

    # Decision should still be valid
    assert result["decision"]["direction"] in {"long", "short", "flat"}


def test_external_signal_with_no_pending_does_not_break_cycle():
    orchestrator = _build_orchestrator()
    # No external signals queued — should run normally
    result = orchestrator.run_cycle_for_symbol("AAPL")
    assert result["decision"]["direction"] == "long"

    # No external signals in the prediction payload
    logged = orchestrator.repository.predictions_logged[0]
    signal_names = [s["name"] for s in logged["payload"]["model_signals"]]
    assert not any(n.startswith("external:") for n in signal_names)


def test_multiple_external_signals_all_consumed():
    orchestrator = _build_orchestrator()
    orchestrator.repository._external_signals = [
        {"id": 10, "symbol": "AAPL", "direction": "long", "score": 0.6, "confidence": 0.7, "source": "site_a", "reasoning": ""},
        {"id": 11, "symbol": "AAPL", "direction": "short", "score": -0.3, "confidence": 0.5, "source": "site_b", "reasoning": ""},
        {"id": 12, "symbol": "MSFT", "direction": "long", "score": 0.5, "confidence": 0.6, "source": "site_a", "reasoning": ""},
    ]

    orchestrator.run_cycle_for_symbol("AAPL")

    # Only AAPL signals consumed (MSFT signal left alone)
    assert "10" in orchestrator.repository._consumed_signal_ids
    assert "11" in orchestrator.repository._consumed_signal_ids
    assert "12" not in orchestrator.repository._consumed_signal_ids


def test_invalid_external_direction_normalized_to_flat():
    orchestrator = _build_orchestrator()
    orchestrator.repository._external_signals = [
        {"id": 99, "symbol": "AAPL", "direction": "YOLO", "score": 1.0, "confidence": 1.0, "source": "test", "reasoning": ""},
    ]

    result = orchestrator.run_cycle_for_symbol("AAPL")
    logged = orchestrator.repository.predictions_logged[0]
    ext_signals = [s for s in logged["payload"]["model_signals"] if s["name"].startswith("external:")]
    assert ext_signals[0]["direction"] == "flat"


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


def test_journal_entry_written_on_trade():
    orchestrator = _build_orchestrator()
    orchestrator.run_cycle_for_symbol("AAPL")

    assert len(orchestrator.repository.journal_entries) == 1
    entry = orchestrator.repository.journal_entries[0]
    assert entry["symbol"] == "AAPL"
    assert entry["event_type"] in {"trade_executed", "trade_skipped"}
    assert "headline" in entry
    assert "body" in entry
    assert "Model signals:" in entry["body"]
    assert "Ensemble decision:" in entry["body"]
    assert "Model weights:" in entry["body"]


def test_journal_entry_includes_external_signal_reasoning():
    orchestrator = _build_orchestrator()
    orchestrator.repository._external_signals = [
        {
            "id": 50,
            "symbol": "AAPL",
            "direction": "long",
            "score": 0.7,
            "confidence": 0.8,
            "source": "website",
            "reasoning": "Cup and handle forming on daily",
        }
    ]

    orchestrator.run_cycle_for_symbol("AAPL")

    entry = orchestrator.repository.journal_entries[0]
    assert "Cup and handle forming on daily" in entry["body"]


def test_journal_skipped_trade_shows_reasons():
    orchestrator = _build_orchestrator()
    # Force a flat decision by making all models flat and removing confident strategies
    orchestrator.models = SimpleNamespace(
        finbert=FakePredictor("finbert", "flat", 0.0, 0.3),
        ppo=FakeRLAgent("ppo", "flat"),
        dqn=FakeRLAgent("dqn", "flat"),
        itransformer=None,
        backtester=FakeBacktester(),
    )

    class FlatStrategy:
        name = "flat_strat"

        def generate_series(self, frame, sentiment_score=0.0):
            return pd.Series([0.0] * len(frame), index=frame.index)

        def generate_latest(self, frame, sentiment_score=0.0):
            return StrategySignal(strategy=self.name, symbol="AAPL", direction="flat", confidence=0.0)

    orchestrator.rule_strategies = [FlatStrategy()]

    orchestrator.run_cycle_for_symbol("AAPL")

    entry = orchestrator.repository.journal_entries[0]
    assert entry["event_type"] == "trade_skipped"
    assert "Trade not placed:" in entry["body"]
