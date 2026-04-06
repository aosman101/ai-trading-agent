from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db.supabase_client import TradeRepository
from app.execution.alpaca_broker import AlpacaBroker
from app.utils.logging import configure_logging
from app.utils.time import utc_now_iso

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title="AI Trading Agent Dashboard", version="0.1.0")
_cors_origins = ["*"] if settings.environment == "dev" else [f"http://localhost:{settings.app_port}"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repository = TradeRepository()
broker = AlpacaBroker()

dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
app.mount("/static", StaticFiles(directory=str(dashboard_dir)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(dashboard_dir / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> JSONResponse:
    snapshot = repository.dashboard_snapshot()
    worker_status = snapshot.get("worker_status") or {}
    last_cycle_at = worker_status.get("last_cycle_at")
    worker_healthy = False
    if last_cycle_at:
        try:
            parsed = datetime.fromisoformat(str(last_cycle_at).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            worker_healthy = (
                (datetime.now(timezone.utc) - parsed).total_seconds()
                <= settings.worker_heartbeat_tolerance_minutes * 60
            )
        except ValueError:
            worker_healthy = False
    payload = {
        "trading_mode": settings.trading_mode,
        "live_enabled": settings.enable_live_trading,
        "kill_switch": settings.kill_switch,
        "current_strategy": snapshot.get("current_strategy"),
        "most_influential_model": snapshot.get("most_influential_model"),
        "market_regime": worker_status.get("market_regime"),
        "weight_scope": worker_status.get("weight_scope"),
        "account_equity": worker_status.get("account_equity", broker.account_equity()),
        "day_pnl": worker_status.get("day_pnl", broker.day_pnl()),
        "open_positions": worker_status.get("open_positions", broker.list_open_positions()),
        "current_portfolio_heat": worker_status.get("current_portfolio_heat"),
        "worker_healthy": worker_healthy,
        "last_cycle_at": last_cycle_at,
        "last_cycle_symbols": worker_status.get("last_cycle_symbols", []),
        "last_error": worker_status.get("last_error"),
    }
    return JSONResponse(payload)


@app.get("/api/dashboard")
def dashboard() -> JSONResponse:
    return JSONResponse(repository.dashboard_snapshot())


@app.get("/api/trades")
def trades(limit: int = Query(default=50, ge=1, le=500)) -> JSONResponse:
    return JSONResponse(repository.recent_trades(limit=limit))


@app.get("/api/predictions")
def predictions(limit: int = Query(default=50, ge=1, le=500)) -> JSONResponse:
    return JSONResponse(repository.recent_predictions(limit=limit))


@app.get("/api/equity")
def equity(limit: int = Query(default=200, ge=1, le=1000)) -> JSONResponse:
    return JSONResponse(repository.equity_curve(limit=limit))


@app.get("/api/model-weights")
def model_weights() -> JSONResponse:
    return JSONResponse(repository.latest_model_weights())


@app.get("/api/learning")
def learning(limit: int = Query(default=100, ge=1, le=500)) -> JSONResponse:
    return JSONResponse(repository.learning_progress(limit=limit))


# ---------------------------------------------------------------------------
# External signals — website sends predictions to the agent
# ---------------------------------------------------------------------------


class ExternalSignalRequest(BaseModel):
    symbol: str
    direction: str = Field(pattern="^(long|short|flat)$")
    score: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "website"
    reasoning: Optional[str] = None


@app.post("/api/signals")
def submit_signal(signal: ExternalSignalRequest) -> JSONResponse:
    """Accept an external prediction from the website.

    The agent will consume it on the next cycle for this symbol and blend it
    into the ensemble alongside its own models.
    """
    record = {
        "created_at": utc_now_iso(),
        "symbol": signal.symbol.upper(),
        "direction": signal.direction,
        "score": signal.score,
        "confidence": signal.confidence,
        "source": signal.source,
        "reasoning": signal.reasoning,
        "consumed_at": None,
        "payload": {},
    }
    repository.submit_external_signal(record)
    return JSONResponse(
        {"status": "accepted", "symbol": record["symbol"], "direction": signal.direction},
        status_code=201,
    )


@app.get("/api/signals")
def list_signals(
    limit: int = Query(default=50, ge=1, le=500),
    symbol: Optional[str] = Query(default=None),
) -> JSONResponse:
    """List recent external signals (both pending and consumed)."""
    rows = repository.list_external_signals(limit=limit)
    if symbol:
        rows = [r for r in rows if r.get("symbol", "").upper() == symbol.upper()]
    return JSONResponse(rows)


# ---------------------------------------------------------------------------
# Journal — the agent's trading diary, readable by the website
# ---------------------------------------------------------------------------


@app.get("/api/journal")
def journal(
    limit: int = Query(default=50, ge=1, le=200),
    symbol: Optional[str] = Query(default=None),
) -> JSONResponse:
    """Return the agent's trading journal — detailed human-readable entries.

    Each entry explains what the agent saw, what it decided, and why.  The
    website can render these directly.
    """
    entries = repository.recent_journal(limit=limit, symbol=symbol.upper() if symbol else None)
    return JSONResponse(entries)
