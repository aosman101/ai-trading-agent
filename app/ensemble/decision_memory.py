from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from app.config import get_settings
from app.types import EnsembleDecision, ModelSignal, RiskPlan, StrategySignal
from app.utils.math_utils import clamp
from app.utils.time import utc_now_iso


class DecisionMemory:
    """Persist decisions, resolve later outcomes, and turn them into lessons."""

    def __init__(self, repository) -> None:
        self.repository = repository
        self.settings = get_settings()

    @staticmethod
    def _direction_multiplier(direction: str, rating: str | None = None) -> float:
        rating = (rating or "").lower()
        if direction == "long" or rating in {"buy", "overweight"}:
            return 1.0
        if direction == "short" or rating in {"sell", "underweight"}:
            return -1.0
        return 0.0

    @staticmethod
    def _prepare_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        work = frame.copy()
        if "ds" not in work.columns:
            work = work.reset_index(names="ds")
        if "close" not in work.columns:
            return pd.DataFrame()
        work["ds"] = pd.to_datetime(work["ds"], utc=True).dt.tz_convert(None)
        return work.sort_values("ds").dropna(subset=["ds", "close"]).reset_index(drop=True)

    @staticmethod
    def _trade_date(value: Any) -> str | None:
        if not value:
            return None
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return str(parsed.date())

    @staticmethod
    def _anchor_index(frame: pd.DataFrame, trade_date: str) -> int | None:
        if frame.empty:
            return None
        search_point = pd.Timestamp(trade_date)
        dates = pd.DatetimeIndex(frame["ds"])
        idx = int(dates.searchsorted(search_point, side="right")) - 1
        if idx < 0:
            return None
        return idx

    @staticmethod
    def _resolve_prices(
        frame: pd.DataFrame,
        trade_date: str,
        holding_bars: int,
    ) -> tuple[float, float, int] | None:
        anchor_idx = DecisionMemory._anchor_index(frame, trade_date)
        if anchor_idx is None:
            return None
        exit_idx = anchor_idx + holding_bars
        if exit_idx >= len(frame):
            return None
        entry_price = float(frame.iloc[anchor_idx]["close"])
        exit_price = float(frame.iloc[exit_idx]["close"])
        if entry_price <= 0:
            return None
        return entry_price, exit_price, holding_bars

    def _reflection(
        self,
        entry: Mapping[str, Any],
        raw_return: float,
        benchmark_return: float | None,
        alpha_return: float | None,
        decision_return: float,
        decision_alpha: float,
    ) -> str:
        direction = str(entry.get("direction", "flat"))
        rating = str(entry.get("rating", "hold"))
        regime = str(entry.get("market_regime") or "unknown regime")
        threshold = float(self.settings.decision_memory_alpha_threshold)

        if direction == "flat":
            if abs(alpha_return or raw_return) <= threshold:
                verdict = "The hold call was reasonable because the asset did not meaningfully diverge from the benchmark."
            elif (alpha_return or raw_return) > threshold:
                verdict = "The hold call missed upside because the asset outperformed after the decision."
            else:
                verdict = "The hold call avoided weakness because the asset underperformed after the decision."
        else:
            verdict = (
                "The directional call worked"
                if decision_return > 0 and decision_alpha >= -threshold
                else "The directional call failed"
            )
            if alpha_return is None:
                verdict += f" with a {decision_return:+.1%} direction-adjusted raw return."
            else:
                verdict += f" with {decision_alpha:+.1%} direction-adjusted alpha."

        risk_flags = (entry.get("payload") or {}).get("risk_flags") or []
        if risk_flags and decision_alpha < -threshold:
            lesson = "Respect similar risk flags more aggressively before sizing the next comparable setup."
        elif risk_flags and decision_alpha >= threshold:
            lesson = "Risk flags were present, but the setup still paid, so use them as sizing dampeners rather than automatic vetoes."
        elif decision_alpha < -threshold:
            lesson = "Demand stronger confirmation from the ensemble before repeating this type of setup."
        else:
            lesson = "Keep favouring this pattern when model agreement and regime evidence stay aligned."

        benchmark_text = ""
        if benchmark_return is not None and alpha_return is not None:
            benchmark_text = f" Raw return was {raw_return:+.1%} versus benchmark {benchmark_return:+.1%}."
        else:
            benchmark_text = f" Raw return was {raw_return:+.1%}; benchmark data was unavailable."

        return f"{verdict}{benchmark_text} Rating was {rating} in {regime}. {lesson}"

    def resolve_pending(
        self,
        symbol: str,
        history_frame: pd.DataFrame,
        benchmark_frame: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        if not self.settings.decision_memory_enabled:
            return []

        history = self._prepare_frame(history_frame)
        benchmark = self._prepare_frame(benchmark_frame)
        if history.empty:
            return []

        resolved: list[dict[str, Any]] = []
        pending = self.repository.pending_decision_memory(symbol=symbol, limit=200)
        for entry in pending:
            trade_date = self._trade_date(entry.get("trade_date") or entry.get("created_at"))
            if trade_date is None:
                continue
            holding_bars = int(entry.get("holding_bars") or self.settings.decision_memory_holding_bars)
            prices = self._resolve_prices(history, trade_date, holding_bars)
            if prices is None:
                continue

            entry_price, exit_price, actual_holding_bars = prices
            raw_return = (exit_price / entry_price) - 1.0

            benchmark_entry = None
            benchmark_exit = None
            benchmark_return = None
            alpha_return = None
            if not benchmark.empty:
                benchmark_prices = self._resolve_prices(benchmark, trade_date, holding_bars)
                if benchmark_prices is not None:
                    benchmark_entry, benchmark_exit, _ = benchmark_prices
                    benchmark_return = (benchmark_exit / benchmark_entry) - 1.0
                    alpha_return = raw_return - benchmark_return

            direction_multiplier = self._direction_multiplier(
                str(entry.get("direction", "flat")),
                str(entry.get("rating", "")),
            )
            decision_return = raw_return * direction_multiplier if direction_multiplier else -abs(raw_return) * 0.1
            comparison_return = alpha_return if alpha_return is not None else raw_return
            decision_alpha = comparison_return * direction_multiplier if direction_multiplier else -abs(comparison_return) * 0.1
            reflection = self._reflection(
                entry=entry,
                raw_return=raw_return,
                benchmark_return=benchmark_return,
                alpha_return=alpha_return,
                decision_return=decision_return,
                decision_alpha=decision_alpha,
            )

            updates = {
                "resolved_at": utc_now_iso(),
                "status": "resolved",
                "actual_holding_bars": actual_holding_bars,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "benchmark_entry_price": benchmark_entry,
                "benchmark_exit_price": benchmark_exit,
                "raw_return": raw_return,
                "benchmark_return": benchmark_return,
                "alpha_return": alpha_return,
                "decision_return": decision_return,
                "decision_alpha": decision_alpha,
                "reflection": reflection,
            }
            self.repository.update_decision_memory_outcome(entry, updates)
            resolved.append({**entry, **updates})
        return resolved

    def store_pending(
        self,
        *,
        symbol: str,
        trade_date: str,
        decision: EnsembleDecision,
        risk_plan: RiskPlan,
        model_signals: list[ModelSignal],
        selected_strategy: StrategySignal | None,
        entry_price: float,
        benchmark_symbol: str,
        benchmark_entry_price: float | None,
    ) -> None:
        if not self.settings.decision_memory_enabled:
            return

        self.repository.store_decision_memory(
            {
                "created_at": utc_now_iso(),
                "status": "pending",
                "symbol": symbol,
                "trade_date": trade_date,
                "benchmark_symbol": benchmark_symbol,
                "holding_bars": int(self.settings.decision_memory_holding_bars),
                "direction": decision.direction,
                "rating": decision.rating,
                "confidence": decision.confidence,
                "weighted_score": decision.weighted_score,
                "entry_price": entry_price,
                "benchmark_entry_price": benchmark_entry_price,
                "market_regime": decision.market_regime,
                "selected_strategy": decision.selected_strategy,
                "payload": {
                    "risk_plan_approved": risk_plan.approved,
                    "risk_plan_reasons": risk_plan.reasons,
                    "risk_flags": decision.risk_flags,
                    "adversarial_review": decision.debate,
                    "weights": decision.weights,
                    "contributions": decision.contributions,
                    "selected_strategy": selected_strategy.model_dump() if selected_strategy else None,
                    "model_signals": [signal.model_dump(mode="json") for signal in model_signals],
                },
            }
        )

    def recent_context(self, symbol: str) -> dict[str, Any]:
        if not self.settings.decision_memory_enabled:
            return {"same_symbol": [], "cross_symbol": []}

        limit = int(self.settings.decision_memory_context_limit)
        rows = self.repository.recent_decision_memory(limit=max(limit * 4, 50), status="resolved")
        same: list[dict[str, Any]] = []
        cross: list[dict[str, Any]] = []
        for row in rows:
            target = same if row.get("symbol") == symbol else cross
            if len(target) >= limit:
                continue
            target.append(row)
            if len(same) >= limit and len(cross) >= limit:
                break
        return {"same_symbol": same, "cross_symbol": cross}

    def context_summary(self, context: Mapping[str, Any]) -> str:
        parts: list[str] = []
        for label, rows in (("Same-symbol lessons", context.get("same_symbol") or []), ("Cross-symbol lessons", context.get("cross_symbol") or [])):
            if not rows:
                continue
            parts.append(label + ":")
            for row in rows[:3]:
                alpha = row.get("alpha_return")
                alpha_text = "n/a" if alpha is None else f"{float(alpha):+.1%}"
                reflection = str(row.get("reflection") or "").strip()
                parts.append(f"- {row.get('trade_date')} {row.get('symbol')} {row.get('rating')}: alpha {alpha_text}. {reflection}")
        return "\n".join(parts)

    def assess_decision(self, decision: EnsembleDecision) -> dict[str, Any]:
        if not self.settings.decision_memory_enabled or decision.direction == "flat":
            return {"samples": 0, "confidence_multiplier": 1.0, "notes": []}

        min_samples = int(self.settings.decision_memory_min_samples)
        threshold = float(self.settings.decision_memory_alpha_threshold)
        rows = self.repository.recent_decision_memory(
            limit=200,
            symbol=decision.symbol,
            status="resolved",
        )

        def matches(row: Mapping[str, Any], *, same_regime: bool) -> bool:
            if row.get("direction") != decision.direction:
                return False
            if same_regime and row.get("market_regime") != decision.market_regime:
                return False
            return row.get("decision_alpha") is not None or row.get("decision_return") is not None

        regime_matches = [row for row in rows if matches(row, same_regime=True)]
        candidates = regime_matches if len(regime_matches) >= min_samples else [row for row in rows if matches(row, same_regime=False)]
        values = [
            float(row.get("decision_alpha") if row.get("decision_alpha") is not None else row.get("decision_return"))
            for row in candidates
            if row.get("decision_alpha") is not None or row.get("decision_return") is not None
        ]

        if len(values) < min_samples:
            return {
                "samples": len(values),
                "confidence_multiplier": 1.0,
                "notes": ["Insufficient resolved memory samples for this setup"],
            }

        avg_alpha = float(sum(values) / len(values))
        win_rate = float(sum(1 for value in values if value > 0) / len(values))
        multiplier = 1.0
        notes: list[str] = []

        if avg_alpha <= -threshold or win_rate < 0.40:
            multiplier = 0.85
            notes.append(
                f"Decision memory: similar {decision.direction} setups underperformed "
                f"(avg direction-adjusted alpha {avg_alpha:+.1%}, win rate {win_rate:.0%})"
            )
        elif avg_alpha >= threshold and win_rate >= 0.60:
            multiplier = 1.05
            notes.append(
                f"Decision memory: similar {decision.direction} setups outperformed "
                f"(avg direction-adjusted alpha {avg_alpha:+.1%}, win rate {win_rate:.0%})"
            )

        return {
            "samples": len(values),
            "same_regime_samples": len(regime_matches),
            "avg_decision_alpha": avg_alpha,
            "win_rate": win_rate,
            "confidence_multiplier": multiplier,
            "notes": notes,
        }

    def apply_assessment(self, decision: EnsembleDecision, assessment: Mapping[str, Any]) -> EnsembleDecision:
        multiplier = float(assessment.get("confidence_multiplier", 1.0) or 1.0)
        if assessment.get("samples", 0) <= 0 and multiplier == 1.0:
            return decision

        confidence = clamp(decision.confidence * multiplier, 0.0, 1.0)
        direction = decision.direction
        rating = decision.rating
        risk_flags = list(decision.risk_flags)
        notes = [str(note) for note in assessment.get("notes", []) if note]

        if multiplier < 1.0:
            risk_flags.extend(notes)
        if direction != "flat" and confidence < self.settings.min_confidence_to_trade:
            direction = "flat"
            rating = "hold"
            risk_flags.append("Decision memory reduced confidence below trading threshold")

        debate = dict(decision.debate)
        debate["decision_memory"] = dict(assessment)
        explanation = f"{decision.explanation}; memory_multiplier={multiplier:.2f}"

        return decision.model_copy(
            update={
                "direction": direction,
                "rating": rating,
                "confidence": confidence,
                "risk_flags": list(dict.fromkeys(risk_flags)),
                "debate": debate,
                "explanation": explanation,
            }
        )
