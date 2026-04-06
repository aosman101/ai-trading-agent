from __future__ import annotations

from typing import Any, Dict

import httpx
import pandas as pd

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class MacroDataService:
    DEFAULT_SERIES = {
        "fed_funds": "FEDFUNDS",
        "cpi": "CPIAUCSL",
        "unemployment": "UNRATE",
        "ten_year_yield": "DGS10",
    }

    def __init__(self) -> None:
        self.settings = get_settings()

    def fetch_fred_series(self, series_id: str, limit: int = 24) -> pd.DataFrame:
        if not self.settings.fred_api_key:
            return pd.DataFrame()
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.settings.fred_api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        try:
            response = httpx.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            observations = response.json().get("observations", [])
            rows = [
                {"date": item["date"], "value": float(item["value"])}
                for item in observations
                if item.get("value") not in {".", None, ""}
            ]
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.warning("FRED request failed for %s: %s", series_id, exc)
            return pd.DataFrame()

    def latest_macro_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        for alias, series_id in self.DEFAULT_SERIES.items():
            frame = self.fetch_fred_series(series_id)
            if frame.empty:
                snapshot[alias] = None
                continue
            snapshot[alias] = float(frame.iloc[0]["value"])
        return snapshot
