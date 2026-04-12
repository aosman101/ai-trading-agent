from __future__ import annotations

import secrets
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
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
settings.validate_runtime_configuration(component="api")
configure_logging(settings.log_level)

app = FastAPI(title="AI Trading Agent Dashboard", version="0.1.0")
_cors_origins = settings.configured_cors_origins or [f"http://localhost:{settings.app_port}"]
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


def _worker_health(worker_status: dict) -> tuple[bool, Optional[str]]:
    last_cycle_at = worker_status.get("last_cycle_at")
    if not last_cycle_at:
        return False, "worker has not completed a cycle yet"
    try:
        parsed = datetime.fromisoformat(str(last_cycle_at).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return False, "worker last_cycle_at is not a valid timestamp"

    age_seconds = (datetime.now(timezone.utc) - parsed).total_seconds()
    if age_seconds > settings.worker_heartbeat_tolerance_minutes * 60:
        return False, "worker heartbeat is stale"
    return True, None


def _broker_health() -> dict[str, object]:
    if not broker.is_enabled:
        return {"enabled": False, "healthy": True, "mode": "simulated", "error": None}
    try:
        equity = broker.account_equity()
        positions = broker.list_open_positions()
        return {
            "enabled": True,
            "healthy": True,
            "mode": "paper" if settings.alpaca_paper else "live",
            "equity": equity,
            "open_positions": positions,
            "error": None,
        }
    except Exception as exc:
        return {
            "enabled": True,
            "healthy": False,
            "mode": "paper" if settings.alpaca_paper else "live",
            "error": str(exc),
        }


def _repository_health() -> dict[str, object]:
    backend = "supabase" if repository.client is not None else "local"
    if repository.client is None:
        return {"backend": backend, "healthy": True, "error": None}
    try:
        repository.client.table("runtime_state").select("state_key").limit(1).execute()
        return {"backend": backend, "healthy": True, "error": None}
    except Exception as exc:
        return {"backend": backend, "healthy": False, "error": str(exc)}


def _normalise_dsi_status(worker_status: dict) -> dict[str, object]:
    dsi_status = worker_status.get("dsi_status") or {}
    if not isinstance(dsi_status, dict):
        dsi_status = {}
    return {
        "configured": bool(dsi_status.get("configured", False)),
        "available": bool(dsi_status.get("available", False)),
        "received_models": list(dsi_status.get("received_models") or []),
        "missing_models": list(dsi_status.get("missing_models") or []),
        "errors": dict(dsi_status.get("errors") or {}),
    }


def _build_status_payload() -> dict[str, object]:
    snapshot = repository.dashboard_snapshot()
    worker_status = snapshot.get("worker_status") or {}
    worker_healthy, worker_error = _worker_health(worker_status)
    dsi_status = _normalise_dsi_status(worker_status)
    broker_status = _broker_health()
    account_equity = worker_status.get("account_equity", broker_status.get("equity", settings.paper_equity_fallback))
    open_positions = worker_status.get("open_positions", broker_status.get("open_positions", {}))
    day_pnl = worker_status.get("day_pnl", 0.0)
    payload = {
        "trading_mode": settings.trading_mode,
        "live_enabled": settings.enable_live_trading,
        "kill_switch": settings.kill_switch,
        "current_strategy": snapshot.get("current_strategy"),
        "most_influential_model": snapshot.get("most_influential_model"),
        "market_regime": worker_status.get("market_regime"),
        "weight_scope": worker_status.get("weight_scope"),
        "account_equity": account_equity,
        "day_pnl": day_pnl,
        "open_positions": open_positions,
        "current_portfolio_heat": worker_status.get("current_portfolio_heat"),
        "worker_healthy": worker_healthy,
        "worker_error": worker_error,
        "last_cycle_at": worker_status.get("last_cycle_at"),
        "last_cycle_symbols": worker_status.get("last_cycle_symbols", []),
        "last_error": worker_status.get("last_error"),
        "dsi_status": dsi_status,
        "broker_status": broker_status,
        "repository_status": _repository_health(),
    }
    return payload


def _require_api_auth(authorization: str | None = Header(default=None)) -> None:
    token = settings.api_bearer_token.strip()
    if not token:
        return
    expected = f"Bearer {token}"
    if not authorization or not secrets.compare_digest(authorization.strip(), expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(dashboard_dir / "index.html")


@app.get("/health")
def health() -> JSONResponse:
    payload = _build_status_payload()
    worker_ok = bool(payload.get("worker_healthy"))
    repository_ok = bool((payload.get("repository_status") or {}).get("healthy", False))
    broker_ok = bool((payload.get("broker_status") or {}).get("healthy", False))
    dsi_status = payload.get("dsi_status") or {}
    dsi_required = bool(dsi_status.get("configured"))
    dsi_ok = (not dsi_required) or (
        bool(dsi_status.get("available"))
        and not dsi_status.get("missing_models")
        and not dsi_status.get("errors")
    )
    overall_ok = worker_ok and repository_ok and broker_ok and dsi_ok
    health_payload = {
        "status": "ok" if overall_ok else "degraded",
        "worker_healthy": worker_ok,
        "repository_healthy": repository_ok,
        "broker_healthy": broker_ok,
        "dsi_healthy": dsi_ok,
        "details": payload,
    }
    return JSONResponse(
        health_payload,
        status_code=status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.get("/api/status")
def status(_: None = Depends(_require_api_auth)) -> JSONResponse:
    return JSONResponse(_build_status_payload())


@app.get("/api/dashboard")
def dashboard(_: None = Depends(_require_api_auth)) -> JSONResponse:
    return JSONResponse(repository.dashboard_snapshot())


@app.get("/api/trades")
def trades(limit: int = Query(default=50, ge=1, le=500), _: None = Depends(_require_api_auth)) -> JSONResponse:
    return JSONResponse(repository.recent_trades(limit=limit))


@app.get("/api/predictions")
def predictions(limit: int = Query(default=50, ge=1, le=500), _: None = Depends(_require_api_auth)) -> JSONResponse:
    return JSONResponse(repository.recent_predictions(limit=limit))


@app.get("/api/equity")
def equity(limit: int = Query(default=200, ge=1, le=1000), _: None = Depends(_require_api_auth)) -> JSONResponse:
    return JSONResponse(repository.equity_curve(limit=limit))


@app.get("/api/model-weights")
def model_weights(_: None = Depends(_require_api_auth)) -> JSONResponse:
    return JSONResponse(repository.latest_model_weights())


@app.get("/api/learning")
def learning(limit: int = Query(default=100, ge=1, le=500), _: None = Depends(_require_api_auth)) -> JSONResponse:
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
def submit_signal(signal: ExternalSignalRequest, _: None = Depends(_require_api_auth)) -> JSONResponse:
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
    _: None = Depends(_require_api_auth),
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
    _: None = Depends(_require_api_auth),
) -> JSONResponse:
    """Return the agent's trading journal — detailed human-readable entries.

    Each entry explains what the agent saw, what it decided, and why.  The
    website can render these directly.
    """
    entries = repository.recent_journal(limit=limit, symbol=symbol.upper() if symbol else None)
    return JSONResponse(entries)
