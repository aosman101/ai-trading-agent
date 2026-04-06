from __future__ import annotations

from pathlib import Path

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
    payload = {
        "trading_mode": settings.trading_mode,
        "live_enabled": settings.enable_live_trading,
        "kill_switch": settings.kill_switch,
        "current_strategy": snapshot.get("current_strategy"),
        "most_influential_model": snapshot.get("most_influential_model"),
        "account_equity": broker.account_equity(),
        "day_pnl": broker.day_pnl(),
        "open_positions": broker.list_open_positions(),
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
