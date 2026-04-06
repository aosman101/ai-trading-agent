from __future__ import annotations

from typing import Any, Dict

from app.config import get_settings
from app.types import RiskPlan
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AlpacaBroker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = None
        if self.settings.alpaca_api_key and self.settings.alpaca_secret_key:
            try:
                from alpaca.trading.client import TradingClient  # type: ignore
            except Exception as exc:
                raise ImportError("alpaca-py is required for AlpacaBroker") from exc

            self.client = TradingClient(
                api_key=self.settings.alpaca_api_key,
                secret_key=self.settings.alpaca_secret_key,
                paper=self.settings.alpaca_paper,
            )

    @property
    def is_enabled(self) -> bool:
        return self.client is not None

    def account_equity(self) -> float:
        if not self.client:
            return self.settings.paper_equity_fallback
        account = self.client.get_account()
        return float(account.equity)

    def day_pnl(self) -> float:
        if not self.client:
            return 0.0
        account = self.client.get_account()
        return float(account.equity) - float(account.last_equity)

    def list_open_positions(self) -> Dict[str, float]:
        if not self.client:
            return {}
        positions = self.client.get_all_positions()
        return {position.symbol: float(position.qty) for position in positions}

    def place_bracket_order(self, trade_plan: RiskPlan) -> Dict[str, Any]:
        if not trade_plan.approved:
            raise ValueError("Trade plan was not approved by risk manager")

        if not self.client:
            logger.info("Alpaca disabled. Returning simulated order for %s", trade_plan.symbol)
            return {
                "id": f"sim-{trade_plan.symbol}",
                "symbol": trade_plan.symbol,
                "qty": trade_plan.quantity,
                "status": "simulated",
            }

        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce  # type: ignore
        from alpaca.trading.requests import (  # type: ignore
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        side = OrderSide.BUY if trade_plan.direction == "long" else OrderSide.SELL
        order_data = MarketOrderRequest(
            symbol=trade_plan.symbol,
            qty=trade_plan.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=trade_plan.take_profit),
            stop_loss=StopLossRequest(stop_price=trade_plan.stop_loss),
        )
        response = self.client.submit_order(order_data=order_data)
        return {
            "id": str(response.id),
            "symbol": response.symbol,
            "qty": float(response.qty),
            "status": str(response.status),
        }

    def cancel_all_orders(self) -> None:
        if self.client:
            self.client.cancel_orders()

    def close_all_positions(self) -> None:
        if self.client:
            self.client.close_all_positions(cancel_orders=True)
