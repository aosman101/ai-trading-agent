# AI Trading Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Alpaca](https://img.shields.io/badge/Broker-Alpaca-ffd700?logo=alpaca&logoColor=black)](https://alpaca.markets/)
[![Supabase](https://img.shields.io/badge/Database-Supabase-3ecf8e?logo=supabase&logoColor=white)](https://supabase.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Autonomous swing trading agent that blends remote DSI forecasts, local sentiment/RL models, rule-based strategies, and external signals into a risk-managed ensemble. Paper-first.

## Architecture

```
Market + News ─▶ Features ─▶ ┌ DSI (NHITS/TFT/LightGBM) ┐
                             ├ FinBERT sentiment        ├─▶ Ensemble ─▶ Risk ─▶ Alpaca ─▶ Supabase/JSONL
                             ├ PPO / DQN (RL)           │
                             └ External signals         ┘
```

## Models

NHITS, TFT, and LightGBM are served remotely by the Deep Stock Insights (DSI) platform — not trained locally.

| Model | Source | Role |
|-------|--------|------|
| NHITS / TFT / LightGBM | DSI | Forecasts, interval hints, directional score |
| FinBERT | Local | News/Reddit sentiment |
| iTransformer | Local (optional) | Secondary forecaster |
| PPO / DQN | Local | RL meta-controllers, dynamically weighted |
| External signals | `/api/signals` | Rate-limited, idempotent user submissions |

## Strategies

Five rule-based strategies compete via live backtest performance: **Momentum**, **Mean Reversion**, **Trend Following**, **Breakout**, **Sentiment**.

## Risk Management

- ATR-based stops combined with DSI TFT interval width (or stop/target fallback).
- Defaults: 1% risk/trade, 3% daily loss limit, 10% portfolio heat, capped open positions.
- Drawdown- and regime-based weight scaling.
- Bracket orders (stop + target) on every fill.
- `KILL_SWITCH=true` halts trading instantly.

## Quick Start

```bash
git clone https://github.com/aosman101/ai-trading-agent.git
cd ai-trading-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add Alpaca, Supabase, DSI creds, API_BEARER_TOKEN, MODEL_HMAC_SECRET
```

Run:

```bash
python -m scripts.bootstrap_models                       # FinBERT / PPO / DQN / iTransformer
python -m app.worker                                     # trading loop
uvicorn app.api.server:app --host 0.0.0.0 --port 8000    # API + dashboard
```

Docker: `docker-compose up --build`.

## Configuration

Key `.env` settings (see [`.env.example`](.env.example) for all):

```env
TRADING_MODE=paper            # paper or live
ENABLE_LIVE_TRADING=false     # extra safety gate
ALPACA_PAPER=true
KILL_SWITCH=false
API_BEARER_TOKEN=...          # required outside dev
MODEL_HMAC_SECRET=...         # required outside dev
CORS_ALLOWED_ORIGINS=...
MAX_RISK_PER_TRADE=0.01
MAX_DAILY_LOSS_PCT=0.03
WORKER_POLL_MINUTES=60
UNIVERSE=AAPL,MSFT,NVDA,SPY,QQQ
DSI_BASE_URL=https://...      # HTTPS required outside dev
DSI_EMAIL=...
DSI_PASSWORD=...
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health`, `/api/status`, `/api/dashboard` | Health + snapshots |
| `GET` | `/api/trades`, `/api/predictions`, `/api/equity` | Activity logs |
| `GET` | `/api/model-weights`, `/api/learning` | Ensemble weights + learning events |
| `GET` | `/api/signals`, `/api/journal` | External signals + trading journal |
| `POST` | `/api/signals` | Submit an external signal |

## Project Structure

```
app/
├── api/          # FastAPI server + dashboard
├── backtesting/  # Walk-forward backtester
├── data/         # Market, news, macro, DSI client
├── db/           # Supabase + JSONL fallback
├── ensemble/     # Weighted decision engine
├── execution/    # Alpaca broker
├── models/       # Local model classes
├── risk/         # Sizing + limits
├── rl/           # PPO, DQN, trading env
├── strategies/   # Rule-based strategies
├── training/     # Bootstrap + nightly retrain
└── utils/
tests/  scripts/  docs/  Dockerfile  docker-compose.yml
```

## Decision Flow

1. Fetch OHLCV, compute 30+ indicators, score news/Reddit with FinBERT.
2. Pull DSI forecasts (with fallback observability) and local iTransformer if enabled.
3. Get PPO/DQN actions with dynamic weights; pull recent external signals and market regime.
4. Pick best rule-based strategy via live backtest.
5. Combine via weighted-agreement scoring scaled by drawdown + regime.
6. Size with ATR + TFT interval (or DSI stop/target fallback); submit Alpaca bracket orders.
7. Log to Supabase (JSONL fallback); retrain local models nightly.

## Safety

- **Paper-first**: live requires `TRADING_MODE=live`, `ENABLE_LIVE_TRADING=true`, `ALPACA_PAPER=false`, and 30 days of paper history.
- **API protection**: set `API_BEARER_TOKEN` and `CORS_ALLOWED_ORIGINS` outside local dev.
- **DSI transport**: HTTPS required outside dev; `dsi_configured` surfaced on health/status.
- **Signal hygiene**: `/api/signals` is rate-limited and idempotent.
- **Model integrity**: HMAC-SHA256 verification on local artifacts (`MODEL_HMAC_SECRET`).
- **Resilience**: SIGTERM/SIGINT graceful shutdown; JSONL fallback when Supabase is down.

## Docs

- [`docs/setup.md`](docs/setup.md) · [`docs/self_learning.md`](docs/self_learning.md) · [`docs/live_transition.md`](docs/live_transition.md) · [`docs/deployment.md`](docs/deployment.md)
