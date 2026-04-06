# Setup

## 1) Create the Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Configure environment variables

```bash
cp .env.example .env
```

Fill in:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `SUPABASE_URL`
- `SUPABASE_KEY`

Optional but useful:

- `ALPHA_VANTAGE_API_KEY` for news sentiment feed
- `FRED_API_KEY` for macro indicators
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` for Reddit sentiment

Keep these values for the first run:

- `TRADING_MODE=paper`
- `ENABLE_LIVE_TRADING=false`
- `ALLOW_SHORTING=false`

## 3) Bootstrap models

```bash
python scripts/bootstrap_models.py
```

This trains:

- NHITS
- LightGBM
- TFT
- PPO
- DQN

FinBERT is loaded from Hugging Face and used directly for inference.

## 4) Run backtests

```bash
python scripts/run_backtests.py
```

The report will be saved to `artifacts/data/backtest_report.csv`.

## 5) Start the dashboard API

```bash
uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## 6) Start the worker

```bash
python -m app.worker
```

The worker will:

- fetch data
- generate forecasts
- score sentiment
- run the ensemble
- apply risk checks
- submit paper trades
- retrain nightly
