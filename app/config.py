from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "dev"
    log_level: str = "INFO"
    timezone: str = "UTC"

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    trading_mode: str = "paper"
    enable_live_trading: bool = False
    allow_model_autopromotion: bool = False
    allow_shorting: bool = False
    kill_switch: bool = False

    universe: str = "AAPL,MSFT,NVDA,SPY,QQQ"
    benchmark_symbol: str = "SPY"
    lookback_years: int = 5
    bar_interval: str = "1d"
    worker_poll_minutes: int = 60
    paper_equity_fallback: float = 100_000.0

    forecast_horizon: int = 12
    n_hits_input_size: int = 120
    n_hits_max_steps: int = 400
    tft_encoder_length: int = 120
    tft_prediction_length: int = 12
    tft_max_epochs: int = 10
    lightgbm_horizon: int = 5

    min_confidence_to_trade: float = 0.55
    max_risk_per_trade: float = 0.01
    max_daily_loss_pct: float = 0.03
    max_portfolio_heat: float = 0.10
    max_open_positions: int = 6
    default_stop_atr_multiplier: float = 1.5
    default_rr_multiplier: float = 2.0

    min_sharpe_to_deploy: float = 1.0
    max_drawdown_to_deploy: float = 0.15
    min_win_rate_to_deploy: float = 0.52
    min_total_return_to_deploy: float = 0.05

    fee_bps: float = 5.0
    slippage_bps: float = 2.0

    artifacts_dir: str = "artifacts"
    model_dir: str = "artifacts/models"
    data_dir: str = "artifacts/data"

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True

    supabase_url: str = ""
    supabase_key: str = ""

    alpha_vantage_api_key: str = ""
    fred_api_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "trade-agent"
    news_rss_urls: str = (
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    )

    finbert_model_name: str = "ProsusAI/finbert"
    hf_device: str = "cpu"

    model_performance_window: int = 100
    strategy_performance_window: int = 60
    paper_days_required: int = 30

    @field_validator("trading_mode")
    @classmethod
    def validate_trading_mode(cls, value: str) -> str:
        allowed = {"paper", "live"}
        value = value.lower().strip()
        if value not in allowed:
            raise ValueError(f"trading_mode must be one of {sorted(allowed)}")
        return value

    @field_validator("bar_interval")
    @classmethod
    def validate_bar_interval(cls, value: str) -> str:
        allowed = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}
        value = value.strip().lower()
        if value not in allowed:
            raise ValueError(f"bar_interval must be one of {sorted(allowed)}")
        return value

    @property
    def symbols(self) -> List[str]:
        return [token.strip().upper() for token in self.universe.split(",") if token.strip()]

    @property
    def model_path(self) -> Path:
        path = Path(self.model_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_path(self) -> Path:
        path = Path(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
