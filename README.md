# AI Trading Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-9cf)](https://lightgbm.readthedocs.io/)
[![Alpaca](https://img.shields.io/badge/Broker-Alpaca-ffd700?logo=alpaca&logoColor=black)](https://alpaca.markets/)
[![Supabase](https://img.shields.io/badge/Database-Supabase-3ecf8e?logo=supabase&logoColor=white)](https://supabase.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

> Autonomous AI-powered swing trading agent that blends remote DSI forecasts, local sentiment/RL models, and rule-based strategies into a risk-managed ensemble with paper-trading-first safety gates.

---

## Architecture

```
Market Data (yfinance) ──┐
News / RSS / Reddit ─────┤
                          ▼
              ┌───────────────────────┐
              │     Feature Engine    │
              │  technical indicators │
              │   + market context    │
              └───────┬───────────────┘
                      │
         ┌────────────┼───────────────┐
         ▼            ▼               ▼
   ┌──────────┐ ┌──────────┐   ┌────────────┐
   │   DSI    │ │ FinBERT  │   │ PPO / DQN  │
   │NHITS/TFT │ │Sentiment │   │ RL Meta    │
   │LightGBM  │ └────┬─────┘   └────┬───────┘
   └────┬─────┘      │              │
        │            └──────┬───────┘
        │                   ▼
        │         ┌───────────────────────┐
        └────────▶│   Ensemble Decision   │
                  │  Engine (dynamic      │
                  │  weighted scoring)    │
                  └───────┬───────────────┘
                          ▼
                  ┌───────────────────────┐
                  │   Risk Manager        │
                  │  ATR stops · heat cap │
                  │  daily loss limit     │
                  └───────┬───────────────┘
                          ▼
                  ┌───────────────────────┐
                  │   Alpaca Broker       │
                  │  bracket orders       │
                  │  paper / live         │
                  └───────┬───────────────┘
                          ▼
                  ┌───────────────────────┐
                  │  Supabase / JSONL     │
                  │  + runtime state      │
                  │  + journal            │
                  └───────────────────────┘
```

## Models

| Model | Role | Output |
|-------|------|--------|
| **DSI NHITS** | Remote price forecast | Expected return |
| **DSI TFT** | Remote forecast with interval hints | Price prediction + stop/target-derived width |
| **DSI LightGBM** | Remote directional classifier | Directional score |
| **FinBERT** | News/Reddit sentiment | Sentiment score (-1 to 1) |
| **iTransformer** | Secondary forecaster (optional) | Expected return |
| **PPO** | RL meta-controller | Buy/sell/hold action |
| **DQN** | RL meta-controller | Buy/sell/hold action |

## Strategies

Five rule-based strategies compete for selection via live backtest performance:

- **Momentum** — EMA crossover + RSI + MACD confirmation.
- **Mean Reversion** — Bollinger Band + RSI extremes.
- **Trend Following** — EMA 50/200 alignment + ATR filter.
- **Breakout** — Rolling high/low channel breaks + volume surge.
- **Sentiment** — FinBERT score thresholds.

## Risk Management

- Position sizing is determined by ATR-based stop distance and DSI TFT interval width or DSI stop/target fallback.
- Configurable maximum risk per trade, with a default setting of 1%.
- Daily loss limit set to a default of 3%.
- Portfolio heat cap set to a default of 10%.
- Maximum limit on the number of open positions.
- Emergency kill switch for immediate halting of trading.
- Stop loss and take profit implemented on every order using bracket orders.

## Quick Start

### Prerequisites

- Python 3.11+
- [Alpaca](https://alpaca.markets/) account (free paper trading)
- [Supabase](https://supabase.com/) project (free tier works)

### Setup

```bash
# Clone
git clone https://github.com/aosman101/ai-trading-agent.git
cd ai-trading-agent

# Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys, DSI credentials, and API_BEARER_TOKEN
# Set MODEL_HMAC_SECRET outside local dev
```

### Run

```bash
# 1. Bootstrap local models (FinBERT/PPO/DQN/iTransformer)
python -m scripts.bootstrap_models

# 2. Start the trading agent
python -m app.worker

# 3. Start the dashboard (separate terminal)
uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker-compose up --build
```

## Configuration

Key settings in `.env`:

```env
TRADING_MODE=paper          # paper or live
ENABLE_LIVE_TRADING=false   # extra safety gate
ALPACA_PAPER=true           # must match TRADING_MODE
ALLOW_SHORTING=false        # enable short trades
KILL_SWITCH=false           # emergency halt
API_BEARER_TOKEN=...        # required outside dev
MODEL_HMAC_SECRET=...       # required outside dev
MAX_RISK_PER_TRADE=0.01    # 1% per trade
MAX_DAILY_LOSS_PCT=0.03    # 3% daily loss limit
WORKER_POLL_MINUTES=60      # cycle frequency
UNIVERSE=AAPL,MSFT,NVDA,SPY,QQQ
DSI_BASE_URL=https://...
DSI_EMAIL=...
DSI_PASSWORD=...
```

See [`.env.example`](.env.example) for all options.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/status` | Trading mode, equity, open positions |
| `GET` | `/api/dashboard` | Full dashboard snapshot |
| `GET` | `/api/trades?limit=50` | Recent trades |
| `GET` | `/api/predictions?limit=50` | Recent predictions |
| `GET` | `/api/equity?limit=200` | Equity curve |
| `GET` | `/api/model-weights` | Current ensemble weights |
| `GET` | `/api/learning?limit=100` | Learning events log |
| `GET` | `/api/signals?limit=50` | External website signals |
| `GET` | `/api/journal?limit=50` | Human-readable trading journal |
| `POST` | `/api/signals` | Submit an external signal |

## Project Structure

```
ai_trading_agent/
├── app/
│   ├── api/            # FastAPI server + dashboard
│   ├── backtesting/    # Walk-forward backtester + metrics
│   ├── data/           # Market, news, macro data services
│   ├── db/             # Supabase client + local JSONL fallback
│   ├── ensemble/       # Dynamic weighted decision engine
│   ├── execution/      # Alpaca broker integration
│   ├── models/         # Local model classes retained for research/testing
│   ├── risk/           # Position sizing + risk limits
│   ├── rl/             # PPO, DQN agents + trading environment
│   ├── strategies/     # Rule-based strategies
│   ├── training/       # Model bootstrap + nightly retrain
│   └── utils/          # Logging, math, safe model serialisation
├── tests/              # Unit tests
├── scripts/            # Bootstrap + backtest runners
├── docs/               # Setup, deployment, live transition guides
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Decision Flow

1. Fetch OHLCV data and compute over 30 technical indicators.
2. Score news and Reddit text using FinBERT.
3. Fetch NHITS, TFT, and LightGBM forecasts from DSI, plus local iTransformer when available.
4. Obtain actions from a reinforcement learning agent using PPO and DQN.
5. Backtest rule-based strategies to identify the best performer.
6. Combine all signals into a dynamically weighted ensemble.
7. Size positions using ATR and TFT uncertainty or DSI stop/target fallback.
8. Submit bracket orders, including stop loss and take profit.
9. Log all activities to Supabase.
10. Retrain models nightly.

## Safety

- **Paper-first**: Live trading requires `TRADING_MODE=live`, `ENABLE_LIVE_TRADING=true`, `ALPACA_PAPER=false`, and 30 days of paper trading history
- **Kill switch**: Set `KILL_SWITCH=true` to instantly halt all trading
- **API protection**: Set `API_BEARER_TOKEN` before exposing `/api/*` outside local dev
- **Model artifact integrity**: Set `MODEL_HMAC_SECRET` outside local dev so HMAC checks are not using the development default
- **Model integrity**: All saved models are verified with HMAC-SHA256 checksums
- **Graceful shutdown**: Handles SIGTERM/SIGINT for clean Docker stops
- **Fallback logging**: If Supabase is unreachable, data is saved locally as JSONL

## Docs

- [`docs/setup.md`](docs/setup.md) — Initial setup guide
- [`docs/self_learning.md`](docs/self_learning.md) — How the self-learning loop works
- [`docs/live_transition.md`](docs/live_transition.md) — Paper to live migration
- [`docs/deployment.md`](docs/deployment.md) — Cloud deployment guide
