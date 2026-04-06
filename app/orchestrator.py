from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from app.backtesting.metrics import max_drawdown, sharpe_ratio
from app.config import get_settings
from app.data.market_data import MarketDataService
from app.data.news_data import NewsDataService
from app.db.supabase_client import TradeRepository
from app.ensemble.decision_engine import EnsembleDecisionEngine
from app.execution.alpaca_broker import AlpacaBroker
from app.risk.risk_manager import RiskManager
from app.strategies.breakout import BreakoutStrategy
from app.strategies.mean_reversion import MeanReversionStrategy
from app.strategies.momentum import MomentumStrategy
from app.strategies.selector import StrategySelector
from app.strategies.sentiment_strategy import SentimentStrategy
from app.strategies.trend_following import TrendFollowingStrategy
from app.training.retrainer import ModelBundle, ModelTrainer
from app.types import ModelSignal, StrategySignal
from app.utils.logging import get_logger
from app.utils.time import utc_now_iso

logger = get_logger(__name__)


class TradingOrchestrator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.market_data = MarketDataService()
        self.news_data = NewsDataService()
        self.repository = TradeRepository()
        self.decision_engine = EnsembleDecisionEngine()
        self.strategy_selector = StrategySelector()
        self.risk_manager = RiskManager()
        self.broker = AlpacaBroker()
        self.model_trainer = ModelTrainer()
        self.models: ModelBundle = self.model_trainer.load_or_bootstrap()
        self._model_lock = threading.Lock()
        self._hydrate_model_performance()

        self.rule_strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            TrendFollowingStrategy(),
            BreakoutStrategy(),
            SentimentStrategy(),
        ]

    def _hydrate_model_performance(self) -> None:
        snapshot = self.repository.read_runtime_state("model_performance")
        if not snapshot:
            return
        for scope, scoped_metrics in snapshot.items():
            if not isinstance(scoped_metrics, dict):
                continue
            if {"accuracy", "sharpe", "calibration", "drawdown"}.intersection(scoped_metrics.keys()):
                self.decision_engine.update_model_performance(scope, scoped_metrics, scope="global")
                continue
            for model_name, metrics in scoped_metrics.items():
                if isinstance(metrics, dict):
                    self.decision_engine.update_model_performance(model_name, metrics, scope=scope)

    def _build_live_rl_observation(self, frame: pd.DataFrame, live_state: Dict[str, float]) -> np.ndarray:
        work = frame.copy()
        for strategy in self.rule_strategies:
            work[f"signal_{strategy.name}"] = strategy.generate_series(work)
        excluded = {
            "symbol",
            "ds",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "forward_return_1",
            "forward_return_5",
            "forward_return_10",
            "target_up_1",
            "target_up_5",
            "target_up_10",
        }
        feature_columns = [column for column in work.columns if column not in excluded]
        clean = work.dropna()
        if clean.empty:
            raise ValueError("No valid rows available to build RL observation — all rows contain NaN")
        latest_row = clean.iloc[-1]
        observation = latest_row[feature_columns].astype(float).to_numpy(dtype=np.float32)
        extras = np.array(
            [
                float(live_state.get("current_position", 0.0)),
                float(live_state.get("portfolio_value", 1.0)),
                float(live_state.get("drawdown", 0.0)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([observation, extras]).astype(np.float32)

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _prediction_outcome(self, history_frame: pd.DataFrame, created_at: Any) -> float | None:
        """Compute a blended multi-horizon return following a prediction.

        Uses 1-bar, 3-bar, and 5-bar forward returns (weighted 0.5/0.3/0.2) to
        reduce noise from single-bar luck while still rewarding timely signals.
        """
        timestamp = self._parse_timestamp(created_at)
        if timestamp is None or history_frame.empty:
            return None

        ds_values = pd.to_datetime(history_frame["ds"], utc=False)
        search_point = pd.Timestamp(timestamp.date())
        anchor_idx = int(ds_values.searchsorted(search_point, side="right")) - 1
        if anchor_idx < 0 or anchor_idx + 1 >= len(history_frame):
            return None

        entry_price = float(history_frame.iloc[anchor_idx]["close"])
        if entry_price == 0:
            return None

        horizons = [(1, 0.50), (3, 0.30), (5, 0.20)]
        blended = 0.0
        total_weight = 0.0
        for bars, weight in horizons:
            target_idx = anchor_idx + bars
            if target_idx >= len(history_frame):
                continue
            exit_price = float(history_frame.iloc[target_idx]["close"])
            blended += weight * ((exit_price / entry_price) - 1.0)
            total_weight += weight

        if total_weight == 0:
            return None
        return blended / total_weight

    @staticmethod
    def _signal_direction_multiplier(direction: str) -> float:
        return {"long": 1.0, "short": -1.0, "flat": 0.0}.get(direction, 0.0)

    def _market_regime(self, history: pd.DataFrame) -> str:
        if history.empty:
            return "range_low_vol"

        latest = history.iloc[-1]
        recent = history.tail(60)

        trend_strength = float(latest.get("trend_strength", 0.0) or 0.0)
        close_price = float(latest.get("close", 0.0) or 0.0)
        ema_50 = float(latest.get("ema_50", close_price) or close_price or 1.0)
        current_vol = float(latest.get("realized_vol_20", 0.0) or 0.0)
        current_width = float(latest.get("bb_width", 0.0) or 0.0)

        vol_clean = recent.get("realized_vol_20")
        vol_clean = vol_clean.dropna() if vol_clean is not None else None
        width_clean = recent.get("bb_width")
        width_clean = width_clean.dropna() if width_clean is not None else None
        vol_baseline = float(vol_clean.median()) if vol_clean is not None and not vol_clean.empty else max(current_vol, 1e-6)
        width_baseline = float(width_clean.median()) if width_clean is not None and not width_clean.empty else max(current_width, 1e-6)

        if trend_strength >= 0.02 and close_price >= ema_50:
            trend_label = "bull_trend"
        elif trend_strength <= -0.02 and close_price <= ema_50:
            trend_label = "bear_trend"
        else:
            trend_label = "range"

        is_high_vol = (
            current_vol > max(vol_baseline * 1.1, vol_baseline + 1e-6)
            or current_width > max(width_baseline * 1.1, width_baseline + 1e-6)
        )
        vol_label = "high_vol" if is_high_vol else "low_vol"
        return f"{trend_label}_{vol_label}"

    def _refresh_model_performance(self, symbol: str, history_frame: pd.DataFrame) -> None:
        recent_predictions = self.repository.recent_predictions(
            limit=max(self.settings.model_performance_lookback * 3, 300)
        )
        per_scope_returns: dict[str, dict[str, list[float]]] = {}
        per_scope_accuracy: dict[str, dict[str, list[float]]] = {}
        per_scope_calibration: dict[str, dict[str, list[float]]] = {}

        for prediction in recent_predictions:
            prediction_symbol = str(prediction.get("symbol", ""))
            if prediction_symbol != symbol:
                continue
            realized_return = self._prediction_outcome(history_frame, prediction.get("created_at"))
            if realized_return is None:
                continue
            payload = prediction.get("payload") or {}
            market_regime = payload.get("market_regime")
            scopes = self.decision_engine.prediction_scopes(prediction_symbol, regime=market_regime)
            signals = payload.get("model_signals") or []
            for signal in signals:
                model_name = str(signal.get("name", "")).strip()
                if not model_name:
                    continue
                direction = str(signal.get("direction", "flat"))
                direction_multiplier = self._signal_direction_multiplier(direction)
                confidence = float(signal.get("confidence", 0.0) or 0.0)
                signal_return = direction_multiplier * realized_return
                if direction_multiplier == 0.0:
                    accuracy = 1.0 if abs(realized_return) < 0.002 else 0.0
                    signal_return = -abs(realized_return) * 0.1
                else:
                    accuracy = 1.0 if signal_return > 0 else 0.0
                realized_strength = min(abs(realized_return) / 0.02, 1.0)
                calibration_error = abs(confidence - realized_strength)

                for scope in scopes:
                    per_scope_returns.setdefault(scope, {}).setdefault(model_name, []).append(signal_return)
                    per_scope_accuracy.setdefault(scope, {}).setdefault(model_name, []).append(accuracy)
                    per_scope_calibration.setdefault(scope, {}).setdefault(model_name, []).append(calibration_error)

        metrics_snapshot: dict[str, dict[str, dict[str, float]]] = {}
        for scope, model_returns in per_scope_returns.items():
            metrics_snapshot[scope] = {}
            for model_name, returns in model_returns.items():
                clipped_returns = returns[-self.settings.model_performance_lookback :]
                accuracy = per_scope_accuracy.get(scope, {}).get(model_name, [])[-self.settings.model_performance_lookback :]
                calibration = per_scope_calibration.get(scope, {}).get(model_name, [])[-self.settings.model_performance_lookback :]
                returns_series = pd.Series(clipped_returns, dtype=float)
                equity_curve = (1.0 + returns_series).cumprod()
                # Apply exponential decay: recent predictions matter more than stale ones.
                n = len(returns_series)
                if n > 1:
                    decay_weights = np.array([0.97 ** (n - 1 - i) for i in range(n)])
                    decay_weights /= decay_weights.sum()
                    weighted_accuracy = float(np.dot(decay_weights[:len(accuracy)], accuracy)) if accuracy else 0.5
                    weighted_calibration = float(np.dot(decay_weights[:len(calibration)], calibration)) if calibration else 0.5
                else:
                    weighted_accuracy = float(np.mean(accuracy)) if accuracy else 0.5
                    weighted_calibration = float(np.mean(calibration)) if calibration else 0.5
                metrics = {
                    "accuracy": weighted_accuracy,
                    "sharpe": sharpe_ratio(returns_series, periods_per_year=252),
                    "calibration": weighted_calibration,
                    "drawdown": max_drawdown(equity_curve) if not equity_curve.empty else 0.0,
                    "samples": float(len(clipped_returns)),
                    "avg_edge": float(returns_series.mean()) if not returns_series.empty else 0.0,
                }
                self.decision_engine.update_model_performance(model_name, metrics, scope=scope)
                metrics_snapshot[scope][model_name] = metrics

        if metrics_snapshot:
            self.repository.write_runtime_state("model_performance", metrics_snapshot)

    def _portfolio_state_snapshot(
        self,
        symbol: str,
        equity: float,
        open_positions: Dict[str, float],
    ) -> Dict[str, float]:
        live_state = self.repository.read_runtime_state("live_state")
        reference_equity = float(live_state.get("reference_equity") or equity or 1.0)
        max_equity = max(float(live_state.get("max_equity") or reference_equity), equity, 1.0)
        current_qty = float(open_positions.get(symbol, 0.0))
        current_position = 1.0 if current_qty > 0 else -1.0 if current_qty < 0 else 0.0
        portfolio_value = equity / max(reference_equity, 1e-9)
        drawdown = max(0.0, 1.0 - (equity / max(max_equity, 1e-9)))
        return {
            "current_position": current_position,
            "portfolio_value": portfolio_value,
            "drawdown": drawdown,
            "reference_equity": reference_equity,
            "max_equity": max_equity,
        }

    def _estimate_open_notional(
        self,
        open_positions: Dict[str, float],
        latest_prices: Dict[str, float] | None = None,
    ) -> float:
        latest_prices = dict(latest_prices or {})
        total_notional = 0.0
        for open_symbol, qty in open_positions.items():
            if qty == 0:
                continue
            price = latest_prices.get(open_symbol)
            if price is None:
                try:
                    price = float(self.market_data.latest_feature_row(open_symbol)["close"])
                except (KeyError, IndexError, ValueError, RuntimeError) as exc:
                    logger.warning("Unable to estimate current price for open position %s: %s", open_symbol, exc)
                    continue
                latest_prices[open_symbol] = price
            total_notional += abs(float(qty)) * abs(float(price))
        return total_notional

    def _write_worker_status(self, payload: Dict[str, Any]) -> None:
        current = self.repository.read_runtime_state("worker_status")
        current.update(payload)
        current["heartbeat_at"] = utc_now_iso()
        self.repository.write_runtime_state("worker_status", current)

    def _combine_rl_agents(self, symbol: str, observation: np.ndarray, models: ModelBundle | None = None) -> ModelSignal:
        models = models or self.models
        ppo_signal = models.ppo.predict(observation, symbol=symbol)
        dqn_signal = models.dqn.predict(observation, symbol=symbol)
        combined_score = (0.60 * ppo_signal.score) + (0.40 * dqn_signal.score)
        combined_confidence = (0.60 * ppo_signal.confidence) + (0.40 * dqn_signal.confidence)
        direction = "flat"
        if combined_score > 0.10:
            direction = "long"
        elif combined_score < -0.10:
            direction = "short"
        return ModelSignal(
            name="rl",
            symbol=symbol,
            direction=direction,
            score=combined_score,
            confidence=combined_confidence,
            metadata={
                "ppo_action": ppo_signal.metadata.get("action"),
                "dqn_action": dqn_signal.metadata.get("action"),
            },
        )

    def _can_place_order(self) -> bool:
        if self.settings.trading_mode == "paper":
            return True
        if not self.settings.enable_live_trading:
            return False
        paper_trades = self.repository.recent_trades(limit=5000)
        distinct_days = {
            str(item.get("created_at", ""))[:10]
            for item in paper_trades
            if item.get("status") in {"submitted", "simulated", "filled"}
        }
        return len({day for day in distinct_days if day}) >= self.settings.paper_days_required

    def _backtest_metrics_for_symbol(
        self,
        symbol: str,
        sentiment_score: float | pd.Series = 0.0,
    ) -> Dict[str, Dict[str, float]]:
        results = self.models.backtester.run_for_symbol(symbol, sentiment_score=sentiment_score)
        metrics = {}
        for strategy_name, result in results.items():
            metrics[strategy_name] = result.metrics
            self.strategy_selector.update_performance(strategy_name, result.metrics)
        return metrics

    def _make_strategy_signals(
        self,
        frame: pd.DataFrame,
        sentiment_score: float,
    ) -> list[StrategySignal]:
        return [
            strategy.generate_latest(frame, sentiment_score=sentiment_score)
            for strategy in self.rule_strategies
        ]

    def _log_prediction(
        self,
        symbol: str,
        decision,
        model_signals: list[ModelSignal],
        selected_strategy: StrategySignal | None,
    ) -> None:
        self.repository.log_prediction(
            {
                "created_at": utc_now_iso(),
                "symbol": symbol,
                "direction": decision.direction,
                "confidence": decision.confidence,
                "weighted_score": decision.weighted_score,
                "selected_strategy": decision.selected_strategy,
                "most_influential_model": decision.most_influential_model,
                "explanation": decision.explanation,
                "payload": {
                    "weights": decision.weights,
                    "weight_scope": decision.weight_scope,
                    "market_regime": decision.market_regime,
                    "contributions": decision.contributions,
                    "model_signals": [signal.model_dump() for signal in model_signals],
                    "selected_strategy": selected_strategy.model_dump() if selected_strategy else None,
                },
            }
        )
        self.repository.save_model_weights(
            {
                "created_at": utc_now_iso(),
                "payload": decision.weights,
            }
        )

    def run_cycle_for_symbol(self, symbol: str) -> Dict[str, object]:
        with self._model_lock:
            models = self.models
        history = self.market_data.fetch_symbol_history(symbol)
        if history.empty:
            raise ValueError(f"No market data returned for {symbol} — cannot run cycle")
        history_frame = history.reset_index(names="ds")
        latest_row = history.iloc[-1]
        latest_price = float(latest_row["close"])
        market_regime = self._market_regime(history)
        self._refresh_model_performance(symbol, history_frame)
        equity = self.broker.account_equity()
        daily_pnl = self.broker.day_pnl()
        open_positions = self.broker.list_open_positions()
        current_open_notional = self._estimate_open_notional(open_positions, latest_prices={symbol: latest_price})
        current_portfolio_heat = (current_open_notional / equity) if equity > 0 else 0.0
        live_state = self._portfolio_state_snapshot(symbol, equity, open_positions)

        texts = self.news_data.collect_text_corpus(symbol)
        finbert_signal = models.finbert.predict_latest(symbol, texts)
        sentiment_series = self.news_data.sentiment_time_series(
            symbol,
            history.index,
            fallback_latest=finbert_signal.score,
        )

        nhits_signal = models.nhits.predict_latest(history_frame, symbol)
        lightgbm_signal = models.lightgbm.predict_latest(latest_row)
        tft_signal = models.tft.predict_latest(history_frame, symbol)

        itransformer_signal = None
        if models.itransformer is not None:
            try:
                itransformer_signal = models.itransformer.predict_latest(history_frame, symbol)
            except (RuntimeError, ValueError, KeyError, IndexError) as exc:
                logger.warning("iTransformer prediction failed for %s: %s", symbol, exc)

        strategy_metrics = self._backtest_metrics_for_symbol(symbol, sentiment_score=sentiment_series)
        strategy_signals = self._make_strategy_signals(history, sentiment_score=finbert_signal.score)
        selected_strategy = self.strategy_selector.select_best(strategy_signals, strategy_metrics)

        observation = self._build_live_rl_observation(history_frame, live_state)
        rl_signal = self._combine_rl_agents(symbol, observation, models=models)

        model_signals = [
            nhits_signal,
            lightgbm_signal,
            tft_signal,
            finbert_signal,
            rl_signal,
        ]
        if itransformer_signal is not None:
            model_signals.append(itransformer_signal)
        decision = self.decision_engine.combine(
            symbol=symbol,
            model_signals=model_signals,
            selected_strategy=selected_strategy,
            regime=market_regime,
        )

        risk_plan = self.risk_manager.build_trade_plan(
            symbol=symbol,
            decision=decision,
            price=latest_price,
            atr=float(latest_row.get("atr_14", latest_price * 0.01)),
            interval_width=float(tft_signal.metadata.get("interval_width", latest_price * 0.01)),
            equity=equity,
            current_daily_pnl=daily_pnl,
            open_positions=open_positions,
            current_open_notional=current_open_notional,
            peak_equity=live_state.get("max_equity"),
        )

        self._log_prediction(symbol, decision, model_signals, selected_strategy)
        self.repository.log_equity(
            {
                "created_at": utc_now_iso(),
                "equity": equity,
                "day_pnl": daily_pnl,
                "payload": {"open_positions": open_positions},
            }
        )
        self.repository.write_runtime_state(
            "live_state",
            {
                **live_state,
                "symbol": symbol,
                "equity": equity,
                "current_portfolio_heat": current_portfolio_heat,
            },
        )

        order_response: Dict[str, object] | None = None
        if risk_plan.approved and self._can_place_order():
            order_response = self.broker.place_bracket_order(risk_plan)
            self.repository.log_trade(
                {
                    "created_at": utc_now_iso(),
                    "symbol": symbol,
                    "direction": risk_plan.direction,
                    "quantity": risk_plan.quantity,
                    "entry_price": risk_plan.entry_price,
                    "stop_loss": risk_plan.stop_loss,
                    "take_profit": risk_plan.take_profit,
                    "risk_amount": risk_plan.risk_amount,
                    "notional": risk_plan.notional,
                    "broker_order_id": order_response.get("id"),
                    "status": order_response.get("status"),
                    "strategy": decision.selected_strategy,
                    "payload": {
                        "decision": decision.model_dump(),
                        "risk_plan": risk_plan.model_dump(),
                    },
                }
            )
        else:
            self.repository.log_learning_event(
                {
                    "created_at": utc_now_iso(),
                    "event_type": "trade_skipped",
                    "message": "; ".join(risk_plan.reasons) or "Order gating prevented trade placement",
                    "payload": {
                        "symbol": symbol,
                        "decision": decision.model_dump(),
                        "risk_plan": risk_plan.model_dump(),
                    },
                }
            )

        status_snapshot = {
            "last_cycle_at": utc_now_iso(),
            "account_equity": equity,
            "day_pnl": daily_pnl,
            "open_positions": open_positions,
            "current_portfolio_heat": current_portfolio_heat,
            "current_strategy": decision.selected_strategy,
            "most_influential_model": decision.most_influential_model,
            "market_regime": decision.market_regime,
            "weight_scope": decision.weight_scope,
            "last_symbol": symbol,
            "last_error": None,
        }

        return {
            "symbol": symbol,
            "decision": decision.model_dump(),
            "risk_plan": risk_plan.model_dump(),
            "order_response": order_response,
            "status_snapshot": status_snapshot,
        }

    def run_cycle(self, symbols: Iterable[str] | None = None) -> list[Dict[str, object]]:
        symbols = list(symbols or self.settings.symbols)
        results = []
        last_error = None
        for symbol in symbols:
            try:
                logger.info("Running cycle for %s", symbol)
                result = self.run_cycle_for_symbol(symbol)
                results.append(result)
            except Exception as exc:
                last_error = f"{symbol} cycle failed: {exc}"
                logger.exception("Trading cycle failed for %s: %s", symbol, exc)
                self.repository.log_learning_event(
                    {
                        "created_at": utc_now_iso(),
                        "event_type": "cycle_error",
                        "message": f"{symbol} cycle failed: {exc}",
                        "payload": {"symbol": symbol},
                    }
                )
        if results:
            status_snapshot = dict(results[-1]["status_snapshot"])
            status_snapshot["last_cycle_symbols"] = symbols
            status_snapshot["last_error"] = last_error
            self._write_worker_status(status_snapshot)
        elif last_error:
            self._write_worker_status(
                {
                    "last_cycle_at": utc_now_iso(),
                    "last_cycle_symbols": symbols,
                    "last_error": last_error,
                }
            )
        return results

    def retrain(self, symbols: Iterable[str] | None = None) -> None:
        logger.info("Starting scheduled retraining")
        self.repository.log_learning_event(
            {
                "created_at": utc_now_iso(),
                "event_type": "retrain_started",
                "message": "Scheduled retraining started",
                "payload": {"symbols": list(symbols or self.settings.symbols)},
            }
        )
        new_models = self.model_trainer.bootstrap_all(symbols=symbols)
        with self._model_lock:
            self.models = new_models
        self.repository.log_learning_event(
            {
                "created_at": utc_now_iso(),
                "event_type": "retrain_completed",
                "message": "Scheduled retraining completed",
                "payload": {"symbols": list(symbols or self.settings.symbols)},
            }
        )
