from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from app.utils.time import utc_now


Direction = Literal["long", "short", "flat"]


class ModelSignal(BaseModel):
    name: str
    symbol: str
    direction: Direction = "flat"
    score: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class StrategySignal(BaseModel):
    strategy: str
    symbol: str
    direction: Direction = "flat"
    confidence: float = 0.0
    reason: str = ""
    metrics: Dict[str, float] = Field(default_factory=dict)


class EnsembleDecision(BaseModel):
    symbol: str
    direction: Direction = "flat"
    confidence: float = 0.0
    weighted_score: float = 0.0
    weights: Dict[str, float] = Field(default_factory=dict)
    contributions: Dict[str, float] = Field(default_factory=dict)
    selected_strategy: Optional[str] = None
    most_influential_model: Optional[str] = None
    explanation: str = ""


class RiskPlan(BaseModel):
    approved: bool = False
    symbol: str
    direction: Direction = "flat"
    quantity: int = 0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_amount: float = 0.0
    notional: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TradeRecord(BaseModel):
    symbol: str
    direction: Direction
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    broker_order_id: Optional[str] = None
    strategy: Optional[str] = None
    model_weights: Dict[str, float] = Field(default_factory=dict)
    status: str = "submitted"
    created_at: datetime = Field(default_factory=utc_now)


class LearningEvent(BaseModel):
    event_type: str
    message: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
