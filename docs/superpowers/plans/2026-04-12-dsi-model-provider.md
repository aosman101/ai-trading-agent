# DSI Model Provider Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the AI Trading Agent's local N-HiTS, TFT, and LightGBM model inference with API calls to the Deep Stock Insights (DSI) platform, so DSI is the single source of truth for forecasting models.

**Architecture:** A new `DSIClient` class in the agent handles JWT authentication and fetches predictions from DSI's `/api/predictions/{asset}` and `/api/scanner/predict/{symbol}` endpoints. The orchestrator calls `DSIClient` instead of local model `.predict_latest()` methods. `ModelBundle` and `ModelTrainer` are updated to remove the three remote models from bootstrap/retrain/load flows. FinBERT, PPO, DQN, iTransformer, and all five strategies remain local.

**Tech Stack:** Python, httpx (already in requirements.txt), FastAPI, pydantic-settings

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `app/data/dsi_client.py` | DSI API client — auth, prediction fetching, response-to-ModelSignal conversion |
| Create | `tests/test_dsi_client.py` | Unit tests for the DSI client |
| Modify | `app/config.py` | Add DSI connection settings (URL, email, password) |
| Modify | `.env.example` | Document new DSI env vars |
| Modify | `app/training/retrainer.py` | Remove N-HiTS, TFT, LightGBM from ModelBundle and ModelTrainer |
| Modify | `app/orchestrator.py` | Replace local model calls with DSIClient calls, graceful fallback |
| Modify | `tests/test_orchestrator.py` | Update orchestrator tests for DSI integration |
| Modify | `requirements.txt` | No change needed — httpx already present |

---

### Task 1: Add DSI Configuration to Settings

**Files:**
- Modify: `app/config.py:66-77` (after existing API key settings)
- Modify: `.env.example:48-59` (after existing API key settings)

- [ ] **Step 1: Add DSI settings to config.py**

In `app/config.py`, add these three fields after the `news_rss_urls` setting (around line 80):

```python
    dsi_base_url: str = ""
    dsi_email: str = ""
    dsi_password: str = ""
```

- [ ] **Step 2: Add DSI env vars to .env.example**

Append after the `HF_DEVICE=cpu` line:

```env
DSI_BASE_URL=http://localhost:8000
DSI_EMAIL=
DSI_PASSWORD=
```

- [ ] **Step 3: Commit**

```bash
git add app/config.py .env.example
git commit -m "feat: add DSI connection settings for remote model provider"
```

---

### Task 2: Create the DSI API Client

**Files:**
- Create: `app/data/dsi_client.py`
- Create: `tests/test_dsi_client.py`

- [ ] **Step 1: Write failing tests for DSIClient**

Create `tests/test_dsi_client.py`:

```python
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("TRADING_MODE", "paper")

for module_name in ("feedparser", "gymnasium", "yfinance"):
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()

from app.types import ModelSignal


# ── Sample DSI API responses ───────────────────────────────────

NHITS_RESPONSE = {
    "status": "ok",
    "asset": "AAPL",
    "model_key": "nhits",
    "current_price": 180.0,
    "prediction": {
        "predicted_close": 185.0,
        "predicted_change_pct": 2.78,
        "confidence": 0.72,
        "signal": "BUY",
        "signal_strength": 0.65,
        "prediction_horizon": "1d",
        "stop_loss": 176.0,
        "take_profit": 190.0,
    },
    "predictions": [
        {
            "predicted_close": 185.0,
            "predicted_change_pct": 2.78,
            "confidence": 0.72,
            "signal": "BUY",
            "signal_strength": 0.65,
            "prediction_horizon": "1d",
        },
        {
            "predicted_close": 188.0,
            "predicted_change_pct": 4.44,
            "confidence": 0.60,
            "signal": "BUY",
            "signal_strength": 0.55,
            "prediction_horizon": "3d",
        },
    ],
}

TFT_RESPONSE = {
    "status": "ok",
    "asset": "AAPL",
    "model_key": "tft",
    "current_price": 180.0,
    "prediction": {
        "predicted_close": 183.0,
        "predicted_change_pct": 1.67,
        "confidence": 0.68,
        "signal": "BUY",
        "signal_strength": 0.58,
        "prediction_horizon": "1d",
        "stop_loss": 177.0,
        "take_profit": 189.0,
    },
    "predictions": [
        {
            "predicted_close": 183.0,
            "predicted_change_pct": 1.67,
            "confidence": 0.68,
            "signal": "BUY",
            "signal_strength": 0.58,
            "prediction_horizon": "1d",
        },
    ],
}

LIGHTGBM_SCANNER_RESPONSE = {
    "status": "ok",
    "asset": "AAPL",
    "model_key": "lightgbm",
    "current_price": 180.0,
    "predicted_close": 182.5,
    "predicted_change_pct": 1.39,
    "signal": "BUY",
    "confidence": 0.61,
    "signal_strength": 0.55,
    "prediction_horizon": "1d",
}


# ── Tests ──────────────────────────────────────────────────────


def test_convert_nhits_response_to_model_signal():
    from app.data.dsi_client import DSIClient

    signal = DSIClient._to_model_signal(NHITS_RESPONSE, "nhits", "AAPL")
    assert isinstance(signal, ModelSignal)
    assert signal.name == "nhits"
    assert signal.symbol == "AAPL"
    assert signal.direction == "long"
    assert signal.score == pytest.approx(0.0278, abs=0.001)
    assert signal.confidence == pytest.approx(0.72)
    assert signal.metadata["current_price"] == 180.0
    assert signal.metadata["predicted_close"] == 185.0


def test_convert_tft_response_to_model_signal():
    from app.data.dsi_client import DSIClient

    signal = DSIClient._to_model_signal(TFT_RESPONSE, "tft", "AAPL")
    assert isinstance(signal, ModelSignal)
    assert signal.name == "tft"
    assert signal.direction == "long"
    assert signal.metadata["predicted_close"] == 183.0


def test_convert_lightgbm_response_to_model_signal():
    from app.data.dsi_client import DSIClient

    signal = DSIClient._to_model_signal(LIGHTGBM_SCANNER_RESPONSE, "lightgbm", "AAPL")
    assert isinstance(signal, ModelSignal)
    assert signal.name == "lightgbm"
    assert signal.direction == "long"
    assert signal.confidence == pytest.approx(0.61)


def test_convert_sell_signal_maps_to_short():
    sell_response = {**NHITS_RESPONSE, "prediction": {**NHITS_RESPONSE["prediction"], "signal": "SELL", "predicted_change_pct": -2.5}}
    from app.data.dsi_client import DSIClient

    signal = DSIClient._to_model_signal(sell_response, "nhits", "AAPL")
    assert signal.direction == "short"


def test_convert_hold_signal_maps_to_flat():
    hold_response = {**NHITS_RESPONSE, "prediction": {**NHITS_RESPONSE["prediction"], "signal": "HOLD", "predicted_change_pct": 0.1}}
    from app.data.dsi_client import DSIClient

    signal = DSIClient._to_model_signal(hold_response, "nhits", "AAPL")
    assert signal.direction == "flat"


def test_convert_missing_prediction_returns_flat():
    empty_response = {"status": "ok", "asset": "AAPL", "model_key": "nhits", "current_price": 180.0}
    from app.data.dsi_client import DSIClient

    signal = DSIClient._to_model_signal(empty_response, "nhits", "AAPL")
    assert signal.direction == "flat"
    assert signal.score == 0.0


def test_symbol_mapping_stock_ticker():
    from app.data.dsi_client import DSIClient

    assert DSIClient._map_symbol("AAPL") == "AAPL"
    assert DSIClient._map_symbol("MSFT") == "MSFT"
    assert DSIClient._map_symbol("SPY") == "SPY"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/adilosman/Downloads/ai_trading_agent && python -m pytest tests/test_dsi_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.data.dsi_client'`

- [ ] **Step 3: Implement DSIClient**

Create `app/data/dsi_client.py`:

```python
from __future__ import annotations

import time
from typing import Dict, Optional

import httpx

from app.config import get_settings
from app.types import ModelSignal
from app.utils.logging import get_logger

logger = get_logger(__name__)

_DSI_SIGNAL_MAP = {"BUY": "long", "SELL": "short", "HOLD": "flat"}
_REQUEST_TIMEOUT = 30.0


class DSIClient:
    """Fetches predictions from the Deep Stock Insights platform."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._base_url = self.settings.dsi_base_url.rstrip("/")
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self._base_url and self.settings.dsi_email and self.settings.dsi_password)

    def _authenticate(self) -> None:
        if self._token and time.time() < self._token_expiry:
            return
        response = httpx.post(
            f"{self._base_url}/api/auth/login",
            data={"username": self.settings.dsi_email, "password": self.settings.dsi_password},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        self._token = payload["access_token"]
        # Refresh 5 minutes before assumed 60-minute expiry
        self._token_expiry = time.time() + 3300

    def _headers(self) -> Dict[str, str]:
        self._authenticate()
        return {"Authorization": f"Bearer {self._token}"}

    @staticmethod
    def _map_symbol(symbol: str) -> str:
        """Map agent symbol to DSI symbol. Currently 1:1 but centralised for future mapping."""
        return symbol.upper()

    @staticmethod
    def _to_model_signal(response: Dict, model_key: str, symbol: str) -> ModelSignal:
        prediction = response.get("prediction") or {}
        current_price = response.get("current_price") or prediction.get("current_price") or 0.0
        predicted_close = prediction.get("predicted_close") or response.get("predicted_close") or 0.0
        change_pct = prediction.get("predicted_change_pct") or response.get("predicted_change_pct") or 0.0
        confidence = prediction.get("confidence") or response.get("confidence") or 0.0
        signal_strength = prediction.get("signal_strength") or response.get("signal_strength") or 0.0
        raw_signal = prediction.get("signal") or response.get("signal") or "HOLD"
        direction = _DSI_SIGNAL_MAP.get(raw_signal.upper(), "flat")

        score = change_pct / 100.0 if abs(change_pct) > 1.0 else change_pct

        stop_loss = prediction.get("stop_loss") or response.get("stop_loss")
        take_profit = prediction.get("take_profit") or response.get("take_profit")
        horizon = prediction.get("prediction_horizon") or response.get("prediction_horizon") or "1d"

        return ModelSignal(
            name=model_key,
            symbol=symbol,
            direction=direction,
            score=score,
            confidence=confidence,
            metadata={
                "source": "dsi",
                "current_price": current_price,
                "predicted_close": predicted_close,
                "signal_strength": signal_strength,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "prediction_horizon": horizon,
            },
        )

    def fetch_nhits_signal(self, symbol: str) -> ModelSignal:
        dsi_symbol = self._map_symbol(symbol)
        response = httpx.get(
            f"{self._base_url}/api/predictions/{dsi_symbol}",
            params={"horizon": "1d", "model_key": "nhits"},
            headers=self._headers(),
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return self._to_model_signal(response.json(), "nhits", symbol)

    def fetch_tft_signal(self, symbol: str) -> ModelSignal:
        dsi_symbol = self._map_symbol(symbol)
        response = httpx.get(
            f"{self._base_url}/api/predictions/{dsi_symbol}",
            params={"horizon": "1d", "model_key": "tft"},
            headers=self._headers(),
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return self._to_model_signal(response.json(), "tft", symbol)

    def fetch_lightgbm_signal(self, symbol: str) -> ModelSignal:
        dsi_symbol = self._map_symbol(symbol)
        response = httpx.get(
            f"{self._base_url}/api/scanner/predict/{dsi_symbol}",
            headers=self._headers(),
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return self._to_model_signal(response.json(), "lightgbm", symbol)

    def fetch_all_signals(self, symbol: str) -> list[ModelSignal]:
        """Fetch N-HiTS, TFT, and LightGBM signals. Skip any that fail."""
        signals: list[ModelSignal] = []
        fetchers = [
            ("nhits", self.fetch_nhits_signal),
            ("tft", self.fetch_tft_signal),
            ("lightgbm", self.fetch_lightgbm_signal),
        ]
        for model_key, fetcher in fetchers:
            try:
                signals.append(fetcher(symbol))
            except Exception as exc:
                logger.warning("DSI %s prediction failed for %s: %s", model_key, symbol, exc)
        return signals
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/adilosman/Downloads/ai_trading_agent && python -m pytest tests/test_dsi_client.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/data/dsi_client.py tests/test_dsi_client.py
git commit -m "feat: add DSI API client for fetching remote predictions"
```

---

### Task 3: Update ModelBundle and ModelTrainer

**Files:**
- Modify: `app/training/retrainer.py`

- [ ] **Step 1: Write failing test for updated ModelBundle**

Add to `tests/test_dsi_client.py`:

```python
def test_model_bundle_no_longer_requires_nhits_tft_lightgbm():
    from app.training.retrainer import ModelBundle
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(ModelBundle)}
    assert "nhits" not in field_names
    assert "tft" not in field_names
    assert "lightgbm" not in field_names
    assert "finbert" in field_names
    assert "ppo" in field_names
    assert "dqn" in field_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/adilosman/Downloads/ai_trading_agent && python -m pytest tests/test_dsi_client.py::test_model_bundle_no_longer_requires_nhits_tft_lightgbm -v`
Expected: FAIL — `nhits` is still in ModelBundle

- [ ] **Step 3: Update ModelBundle dataclass**

In `app/training/retrainer.py`, replace the `ModelBundle` dataclass and remove the three local model imports:

Replace the imports block at the top:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from app.backtesting.engine import WalkForwardBacktester
from app.config import get_settings
from app.data.market_data import MarketDataService
from app.models.finbert_sentiment import FinBERTSentimentModel
from app.models.itransformer_model import ITransformerForecaster
from app.rl.dqn_agent import DQNTradingAgent
from app.rl.ppo_agent import PPOTradingAgent
from app.rl.trading_env import TradingEnvironment
from app.strategies.breakout import BreakoutStrategy
from app.strategies.mean_reversion import MeanReversionStrategy
from app.strategies.momentum import MomentumStrategy
from app.strategies.sentiment_strategy import SentimentStrategy
from app.strategies.trend_following import TrendFollowingStrategy
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelBundle:
    finbert: FinBERTSentimentModel
    ppo: PPOTradingAgent
    dqn: DQNTradingAgent
    itransformer: ITransformerForecaster | None
    backtester: WalkForwardBacktester
```

Replace `bootstrap_all` to remove N-HiTS, TFT, LightGBM training:

```python
    def bootstrap_all(self, symbols: Iterable[str] | None = None) -> ModelBundle:
        symbols = list(symbols or self.settings.symbols)
        universe_frame = self.market_data.fetch_universe_history(symbols=symbols)

        itransformer = None
        try:
            itransformer = ITransformerForecaster()
            itransformer.fit(universe_frame)
        except Exception as exc:
            logger.warning("iTransformer bootstrap skipped: %s", exc)

        rl_frame, feature_columns = self._build_rl_frame(universe_frame)
        env = TradingEnvironment(rl_frame, feature_columns=feature_columns, return_column="forward_return_1")
        ppo = PPOTradingAgent()
        ppo.fit(env, total_timesteps=20_000)
        dqn = DQNTradingAgent()
        dqn.fit(env, total_timesteps=20_000)

        finbert = FinBERTSentimentModel()

        bundle = ModelBundle(
            finbert=finbert,
            ppo=ppo,
            dqn=dqn,
            itransformer=itransformer,
            backtester=self.backtester,
        )
        self.save(bundle)
        return bundle
```

Replace `_model_files`:

```python
    def _model_files(self) -> dict[str, Path]:
        base = self.settings.model_path
        return {
            "ppo": base / "ppo_agent",
            "dqn": base / "dqn_agent",
            "itransformer": base / "itransformer.pkl",
        }
```

Replace `save`:

```python
    def save(self, bundle: ModelBundle) -> None:
        files = self._model_files()
        bundle.ppo.save(files["ppo"])
        bundle.dqn.save(files["dqn"])
        if bundle.itransformer is not None:
            bundle.itransformer.save(files["itransformer"])
```

Replace `load`:

```python
    def load(self) -> ModelBundle:
        files = self._model_files()

        ppo = PPOTradingAgent()
        ppo.load(files["ppo"])

        dqn = DQNTradingAgent()
        dqn.load(files["dqn"])

        itransformer = None
        if files["itransformer"].exists():
            try:
                itransformer = ITransformerForecaster()
                itransformer.load(files["itransformer"])
            except Exception as exc:
                logger.warning("Failed to load iTransformer: %s", exc)

        finbert = FinBERTSentimentModel()

        return ModelBundle(
            finbert=finbert,
            ppo=ppo,
            dqn=dqn,
            itransformer=itransformer,
            backtester=self.backtester,
        )
```

Replace `load_or_bootstrap`:

```python
    def load_or_bootstrap(self, symbols: Iterable[str] | None = None) -> ModelBundle:
        files = self._model_files()
        required = [Path(str(files["ppo"]) + ".zip"), Path(str(files["dqn"]) + ".zip")]
        if all(path.exists() for path in required):
            logger.info("Loading existing trained models from disk")
            return self.load()

        logger.info("No full model bundle found, bootstrapping models from fresh history")
        return self.bootstrap_all(symbols=symbols)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/adilosman/Downloads/ai_trading_agent && python -m pytest tests/test_dsi_client.py::test_model_bundle_no_longer_requires_nhits_tft_lightgbm -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/training/retrainer.py tests/test_dsi_client.py
git commit -m "feat: remove N-HiTS, TFT, LightGBM from ModelBundle — now fetched from DSI"
```

---

### Task 4: Update the Orchestrator to Use DSIClient

**Files:**
- Modify: `app/orchestrator.py`

- [ ] **Step 1: Add DSIClient import and initialisation**

In `app/orchestrator.py`, add the import near the top (after the existing imports):

```python
from app.data.dsi_client import DSIClient
```

In `TradingOrchestrator.__init__`, add the DSI client after `self.model_trainer` (around line 43):

```python
        self.dsi_client = DSIClient()
```

- [ ] **Step 2: Replace local model calls in run_cycle_for_symbol**

In `run_cycle_for_symbol`, replace the block that calls local nhits, lightgbm, tft (lines 563-573):

Find:
```python
        nhits_signal = models.nhits.predict_latest(history_frame, symbol)
        lightgbm_signal = models.lightgbm.predict_latest(latest_row)
        tft_signal = models.tft.predict_latest(history_frame, symbol)
```

Replace with:
```python
        # Fetch N-HiTS, TFT, LightGBM from DSI platform
        dsi_signals = []
        if self.dsi_client.configured:
            dsi_signals = self.dsi_client.fetch_all_signals(symbol)
            if not dsi_signals:
                logger.warning("DSI returned no signals for %s — continuing with local models only", symbol)

        # Extract individual signals (or create flat fallbacks)
        nhits_signal = next((s for s in dsi_signals if s.name == "nhits"), ModelSignal(name="nhits", symbol=symbol))
        tft_signal = next((s for s in dsi_signals if s.name == "tft"), ModelSignal(name="tft", symbol=symbol))
        lightgbm_signal = next((s for s in dsi_signals if s.name == "lightgbm"), ModelSignal(name="lightgbm", symbol=symbol))
```

- [ ] **Step 3: Update the TFT interval_width fallback in risk_plan**

The `risk_plan` call uses `tft_signal.metadata.get("interval_width", ...)`. DSI responses don't include `interval_width` directly, so we need to compute it from `stop_loss` and `take_profit` or use a sensible default.

Find in `run_cycle_for_symbol`:
```python
            interval_width=float(tft_signal.metadata.get("interval_width", latest_price * 0.01)),
```

Replace with:
```python
            interval_width=float(
                tft_signal.metadata.get("interval_width")
                or abs((tft_signal.metadata.get("take_profit") or latest_price * 1.01) - (tft_signal.metadata.get("stop_loss") or latest_price * 0.99))
                or latest_price * 0.01
            ),
```

- [ ] **Step 4: Run existing orchestrator tests**

Run: `cd /Users/adilosman/Downloads/ai_trading_agent && python -m pytest tests/test_orchestrator.py -v`
Expected: Tests should still pass (orchestrator tests use mocked models — the DSI client won't be configured in test env, so fallback flat signals will be used)

- [ ] **Step 5: Commit**

```bash
git add app/orchestrator.py
git commit -m "feat: orchestrator fetches N-HiTS/TFT/LightGBM from DSI instead of local models"
```

---

### Task 5: Update Orchestrator Tests

**Files:**
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 1: Add DSI client mock to orchestrator test fixtures**

In `tests/test_orchestrator.py`, add a `FakeDSIClient` class after the existing fake classes:

```python
class FakeDSIClient:
    configured = True

    def fetch_all_signals(self, symbol: str):
        return [
            ModelSignal(name="nhits", symbol=symbol, direction="long", score=0.02, confidence=0.7,
                        metadata={"source": "dsi", "current_price": 110.0, "predicted_close": 112.0,
                                  "signal_strength": 0.6, "stop_loss": 107.0, "take_profit": 116.0,
                                  "prediction_horizon": "1d"}),
            ModelSignal(name="tft", symbol=symbol, direction="long", score=0.015, confidence=0.65,
                        metadata={"source": "dsi", "current_price": 110.0, "predicted_close": 111.5,
                                  "signal_strength": 0.55, "stop_loss": 108.0, "take_profit": 115.0,
                                  "prediction_horizon": "1d"}),
            ModelSignal(name="lightgbm", symbol=symbol, direction="long", score=0.1, confidence=0.6,
                        metadata={"source": "dsi", "current_price": 110.0, "predicted_close": 111.0,
                                  "signal_strength": 0.5, "stop_loss": None, "take_profit": None,
                                  "prediction_horizon": "1d"}),
        ]
```

- [ ] **Step 2: Wire FakeDSIClient into existing test fixtures**

In the test functions that build a `TradingOrchestrator`, add `dsi_client` assignment after the orchestrator is patched. For example, in any fixture or setup that creates an orchestrator:

```python
orchestrator.dsi_client = FakeDSIClient()
```

- [ ] **Step 3: Add a test for DSI-unavailable fallback**

Add this test to `tests/test_orchestrator.py`:

```python
def test_orchestrator_continues_when_dsi_unavailable():
    """When DSI returns no signals, the ensemble should still work with local models."""

    class EmptyDSIClient:
        configured = True
        def fetch_all_signals(self, symbol: str):
            return []

    orchestrator = _build_test_orchestrator()
    orchestrator.dsi_client = EmptyDSIClient()
    # The orchestrator should still run without raising
    # (FinBERT, RL, iTransformer, and strategies still provide signals)
```

Note: `_build_test_orchestrator` refers to the existing test helper pattern used in `test_orchestrator.py`. Adapt to match the actual test setup pattern.

- [ ] **Step 4: Run all orchestrator tests**

Run: `cd /Users/adilosman/Downloads/ai_trading_agent && python -m pytest tests/test_orchestrator.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_orchestrator.py
git commit -m "test: update orchestrator tests for DSI integration and fallback"
```

---

### Task 6: Update .env with DSI Credentials

**Files:**
- Modify: `.env` (local, not committed)

- [ ] **Step 1: Add DSI credentials to .env**

Add to `.env` (this file is gitignored):

```env
DSI_BASE_URL=http://localhost:8000
DSI_EMAIL=your-dsi-email@example.com
DSI_PASSWORD=your-dsi-password
```

- [ ] **Step 2: Verify the client connects**

Run a quick smoke test (requires DSI backend running):

```bash
cd /Users/adilosman/Downloads/ai_trading_agent
python -c "
from app.data.dsi_client import DSIClient
client = DSIClient()
print('Configured:', client.configured)
if client.configured:
    try:
        signal = client.fetch_nhits_signal('AAPL')
        print('NHITS signal:', signal.direction, signal.score, signal.confidence)
    except Exception as e:
        print('Connection failed (expected if DSI not running):', e)
"
```

---

### Task 7: Run Full Test Suite

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/adilosman/Downloads/ai_trading_agent && python -m pytest tests/ -v`
Expected: All tests PASS. Some tests may need minor adjustments if they directly reference `models.nhits`, `models.tft`, or `models.lightgbm` on the bundle.

- [ ] **Step 2: Fix any remaining test failures**

If `tests/test_forecasters.py` has tests for the removed local models (NHITSForecaster, TFTForecaster, LightGBMSignalModel), those tests are still valid as unit tests of the model classes themselves — the classes still exist in `app/models/`, they're just no longer used by the orchestrator. No changes needed to those tests.

If any test constructs a `ModelBundle` with the old fields, update it to use the new fields:

```python
# Old:
# bundle = ModelBundle(nhits=..., lightgbm=..., tft=..., finbert=..., ppo=..., dqn=..., itransformer=..., backtester=...)
# New:
bundle = ModelBundle(finbert=..., ppo=..., dqn=..., itransformer=..., backtester=...)
```

- [ ] **Step 3: Commit any test fixes**

```bash
git add -A tests/
git commit -m "fix: update remaining tests for new ModelBundle shape"
```
