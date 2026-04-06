from __future__ import annotations

import threading
from datetime import datetime
from typing import Dict, Iterable

import numpy as np
import pandas as pd

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

        self.rule_strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            TrendFollowingStrategy(),
            BreakoutStrategy(),
            SentimentStrategy(),
        ]

    def _build_live_rl_observation(self, frame: pd.DataFrame) -> np.ndarray:
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
        latest_row = work.dropna().iloc[-1]
        observation = latest_row[feature_columns].astype(float).to_numpy(dtype=np.float32)
        extras = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.concatenate([observation, extras]).astype(np.float32)

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

    def _backtest_metrics_for_symbol(self, symbol: str) -> Dict[str, Dict[str, float]]:
        results = self.models.backtester.run_for_symbol(symbol)
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
                "created_at": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "direction": decision.direction,
                "confidence": decision.confidence,
                "weighted_score": decision.weighted_score,
                "selected_strategy": decision.selected_strategy,
                "most_influential_model": decision.most_influential_model,
                "explanation": decision.explanation,
                "payload": {
                    "weights": decision.weights,
                    "contributions": decision.contributions,
                    "model_signals": [signal.model_dump() for signal in model_signals],
                    "selected_strategy": selected_strategy.model_dump() if selected_strategy else None,
                },
            }
        )
        self.repository.save_model_weights(
            {
                "created_at": datetime.utcnow().isoformat(),
                "payload": decision.weights,
            }
        )

    def run_cycle_for_symbol(self, symbol: str) -> Dict[str, object]:
        with self._model_lock:
            models = self.models
        history = self.market_data.fetch_symbol_history(symbol)
        history_frame = history.reset_index(names="ds")
        latest_row = history.iloc[-1]

        texts = self.news_data.collect_text_corpus(symbol)
        finbert_signal = models.finbert.predict_latest(symbol, texts)

        nhits_signal = models.nhits.predict_latest(history_frame, symbol)
        lightgbm_signal = models.lightgbm.predict_latest(latest_row)
        tft_signal = models.tft.predict_latest(history_frame, symbol)

        itransformer_signal = None
        if models.itransformer is not None:
            try:
                itransformer_signal = models.itransformer.predict_latest(history_frame, symbol)
            except Exception as exc:
                logger.warning("iTransformer prediction failed for %s: %s", symbol, exc)

        strategy_metrics = self._backtest_metrics_for_symbol(symbol)
        strategy_signals = self._make_strategy_signals(history, sentiment_score=finbert_signal.score)
        selected_strategy = self.strategy_selector.select_best(strategy_signals, strategy_metrics)

        observation = self._build_live_rl_observation(history_frame)
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
        )

        equity = self.broker.account_equity()
        daily_pnl = self.broker.day_pnl()
        open_positions = self.broker.list_open_positions()
        risk_plan = self.risk_manager.build_trade_plan(
            symbol=symbol,
            decision=decision,
            price=float(latest_row["close"]),
            atr=float(latest_row.get("atr_14", latest_row["close"] * 0.01)),
            interval_width=float(tft_signal.metadata.get("interval_width", latest_row["close"] * 0.01)),
            equity=equity,
            current_daily_pnl=daily_pnl,
            open_positions=open_positions,
        )

        self._log_prediction(symbol, decision, model_signals, selected_strategy)
        self.repository.log_equity(
            {
                "created_at": datetime.utcnow().isoformat(),
                "equity": equity,
                "day_pnl": daily_pnl,
                "payload": {"open_positions": open_positions},
            }
        )

        order_response: Dict[str, object] | None = None
        if risk_plan.approved and self._can_place_order():
            order_response = self.broker.place_bracket_order(risk_plan)
            self.repository.log_trade(
                {
                    "created_at": datetime.utcnow().isoformat(),
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
                    "created_at": datetime.utcnow().isoformat(),
                    "event_type": "trade_skipped",
                    "message": "; ".join(risk_plan.reasons) or "Order gating prevented trade placement",
                    "payload": {
                        "symbol": symbol,
                        "decision": decision.model_dump(),
                        "risk_plan": risk_plan.model_dump(),
                    },
                }
            )

        return {
            "symbol": symbol,
            "decision": decision.model_dump(),
            "risk_plan": risk_plan.model_dump(),
            "order_response": order_response,
        }

    def run_cycle(self, symbols: Iterable[str] | None = None) -> list[Dict[str, object]]:
        symbols = list(symbols or self.settings.symbols)
        results = []
        for symbol in symbols:
            try:
                logger.info("Running cycle for %s", symbol)
                result = self.run_cycle_for_symbol(symbol)
                results.append(result)
            except Exception as exc:
                logger.exception("Trading cycle failed for %s: %s", symbol, exc)
                self.repository.log_learning_event(
                    {
                        "created_at": datetime.utcnow().isoformat(),
                        "event_type": "cycle_error",
                        "message": f"{symbol} cycle failed: {exc}",
                        "payload": {"symbol": symbol},
                    }
                )
        return results

    def retrain(self, symbols: Iterable[str] | None = None) -> None:
        logger.info("Starting scheduled retraining")
        self.repository.log_learning_event(
            {
                "created_at": datetime.utcnow().isoformat(),
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
                "created_at": datetime.utcnow().isoformat(),
                "event_type": "retrain_completed",
                "message": "Scheduled retraining completed",
                "payload": {"symbols": list(symbols or self.settings.symbols)},
            }
        )
