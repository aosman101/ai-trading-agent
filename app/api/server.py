from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.db.supabase_client import TradeRepository
from app.execution.alpaca_broker import AlpacaBroker
from app.utils.logging import configure_logging

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
