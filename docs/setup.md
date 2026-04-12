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
- `API_BEARER_TOKEN` if you plan to expose the API outside local dev
- `MODEL_HMAC_SECRET` for model artifact integrity outside local dev

Optional but useful:

- `ALPHA_VANTAGE_API_KEY` for news sentiment feed
- `FRED_API_KEY` for macro indicators
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` for Reddit sentiment
- `DSI_BASE_URL`, `DSI_EMAIL`, `DSI_PASSWORD` if you want remote NHITS/TFT/LightGBM forecasts

Keep these values for the first run:

- `TRADING_MODE=paper`
- `ENABLE_LIVE_TRADING=false`
- `ALPACA_PAPER=true`
- `ALLOW_SHORTING=false`

## 3) Bootstrap models

```bash
python scripts/bootstrap_models.py
```

This bootstraps the local runtime models:

- PPO
- DQN
- iTransformer when available

FinBERT is loaded from Hugging Face for inference. NHITS, TFT, and LightGBM are now fetched from DSI when DSI credentials are configured.

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

If `API_BEARER_TOKEN` is set, open `http://localhost:8000/?token=YOUR_TOKEN`.

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
