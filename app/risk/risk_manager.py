from __future__ import annotations

from typing import Dict, Mapping

from app.config import get_settings
from app.types import EnsembleDecision, RiskPlan
from app.utils.math_utils import clamp


# Regime-based scaling: dampen sizing in unfavourable regimes, boost in favourable ones.
_REGIME_POSITION_SCALE: Dict[str, float] = {
    "bull_trend_low_vol": 1.0,
    "bull_trend_high_vol": 0.70,
    "bear_trend_low_vol": 0.50,
    "bear_trend_high_vol": 0.35,
    "range_low_vol": 0.60,
    "range_high_vol": 0.45,
}


class RiskManager:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _stop_distance(self, price: float, atr: float, interval_width: float) -> float:
        atr_component = max(atr * self.settings.default_stop_atr_multiplier, price * 0.002)
        interval_component = max(interval_width * 0.5, price * 0.002)
        return max(atr_component, interval_component)

    @staticmethod
    def _regime_scale(regime: str | None) -> float:
        if not regime:
            return 0.60
        return _REGIME_POSITION_SCALE.get(regime, 0.60)

    @staticmethod
    def _drawdown_scale(equity: float, peak_equity: float) -> float:
        """Progressively shrink position size as drawdown deepens."""
        if peak_equity <= 0:
            return 1.0
        dd = 1.0 - (equity / peak_equity)
        if dd <= 0.03:
            return 1.0
        if dd <= 0.06:
            return 0.60
        if dd <= 0.10:
            return 0.30
        return 0.0  # circuit breaker: no new trades beyond 10% drawdown

    def build_trade_plan(
        self,
        symbol: str,
        decision: EnsembleDecision,
        price: float,
        atr: float,
        interval_width: float,
        equity: float,
        current_daily_pnl: float,
        open_positions: Mapping[str, float] | None = None,
        current_open_notional: float = 0.0,
        peak_equity: float | None = None,
    ) -> RiskPlan:
        open_positions = open_positions or {}
        reasons: list[str] = []
        notes: list[str] = []

        if self.settings.kill_switch:
            reasons.append("Kill switch is enabled")
        if current_daily_pnl <= -(equity * self.settings.max_daily_loss_pct):
            reasons.append("Maximum daily loss limit reached")
        if decision.confidence < self.settings.min_confidence_to_trade:
            reasons.append("Model confidence below threshold")
        if len(open_positions) >= self.settings.max_open_positions and symbol not in open_positions:
            reasons.append("Maximum open positions reached")
        if decision.direction == "flat":
            reasons.append("Decision is flat, so no trade should be placed")
        if decision.direction == "short" and not self.settings.allow_shorting:
            reasons.append("Shorting is disabled in settings")

        # --- Drawdown circuit breaker ---
        effective_peak = peak_equity if peak_equity and peak_equity > 0 else equity
        dd_scale = self._drawdown_scale(equity, effective_peak)
        if dd_scale <= 0.0:
            reasons.append("Drawdown circuit breaker: equity drawdown exceeds 10%")

        stop_distance = self._stop_distance(price=price, atr=atr, interval_width=interval_width)
        risk_budget = equity * self.settings.max_risk_per_trade
        confidence_scale = clamp(decision.confidence, 0.25, 1.0)
        uncertainty_scale = clamp(1.0 - min(interval_width / max(price, 1e-9), 0.80), 0.25, 1.0)

        # --- Regime-adaptive sizing ---
        regime_scale = self._regime_scale(decision.market_regime)

        quantity = int(
            (risk_budget / max(stop_distance, 0.01))
            * confidence_scale
            * uncertainty_scale
            * regime_scale
            * dd_scale
        )
        quantity = max(0, quantity)

        if quantity <= 0 and not reasons:
            reasons.append("Calculated quantity is zero after risk sizing")

        direction_multiplier = 1.0 if decision.direction == "long" else -1.0
        stop_loss = price - (direction_multiplier * stop_distance)
        take_profit = price + (
            direction_multiplier
            * stop_distance
            * self.settings.default_rr_multiplier
        )

        if decision.direction == "long" and stop_loss >= price:
            reasons.append("Stop loss is above entry for long trade")
        elif decision.direction == "short" and stop_loss <= price:
            reasons.append("Stop loss is below entry for short trade")

        notional = quantity * price
        projected_heat = ((current_open_notional + notional) / equity) if equity > 0 else 0.0

        if equity > 0 and projected_heat > self.settings.max_portfolio_heat:
            remaining_notional = max((equity * self.settings.max_portfolio_heat) - current_open_notional, 0.0)
            capped_qty = int(remaining_notional / max(price, 0.01))
            quantity = max(0, capped_qty)
            notional = quantity * price
            if quantity == 0:
                reasons.append("Portfolio heat cap reduced quantity to zero")
            else:
                notes.append("Portfolio heat cap reduced position size")
            projected_heat = ((current_open_notional + notional) / equity) if equity > 0 else 0.0

        approved = len(reasons) == 0
        return RiskPlan(
            approved=approved,
            symbol=symbol,
            direction=decision.direction,
            quantity=quantity,
            entry_price=price,
            stop_loss=round(float(stop_loss), 2),
            take_profit=round(float(take_profit), 2),
            risk_amount=round(min(risk_budget, quantity * stop_distance), 2),
            notional=round(float(notional), 2),
            reasons=reasons,
            metadata={
                "confidence_scale": confidence_scale,
                "uncertainty_scale": uncertainty_scale,
                "regime_scale": regime_scale,
                "drawdown_scale": dd_scale,
                "interval_width": interval_width,
                "stop_distance": stop_distance,
                "current_open_notional": current_open_notional,
                "projected_portfolio_heat": projected_heat,
                "notes": notes,
            },
        )
