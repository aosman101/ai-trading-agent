from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.config import get_settings
from app.utils.logging import get_logger
from app.utils.time import utc_now_iso

logger = get_logger(__name__)


class TradeRepository:
    TABLES = (
        "predictions",
        "trades",
        "equity_curve",
        "learning_events",
        "model_weights",
        "runtime_state",
    )

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_path = Path(self.settings.data_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.client = None
        if self.settings.supabase_url and self.settings.supabase_key:
            try:
                from supabase import create_client  # type: ignore
            except Exception as exc:
                raise ImportError("supabase package is required when Supabase is configured") from exc
            self.client = create_client(self.settings.supabase_url, self.settings.supabase_key)

    def _path(self, table: str) -> Path:
        return self.base_path / f"{table}.jsonl"

    def _runtime_state_path(self) -> Path:
        return self.base_path / "runtime_state.json"

    def _append_local(self, table: str, record: Dict[str, Any]) -> None:
        with self._path(table).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def _read_local(self, table: str) -> List[Dict[str, Any]]:
        path = self._path(table)
        if not path.exists():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _read_runtime_state_local(self) -> Dict[str, Dict[str, Any]]:
        path = self._runtime_state_path()
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_runtime_state_local(self, state_key: str, payload: Dict[str, Any]) -> None:
        state = self._read_runtime_state_local()
        state[state_key] = {
            "updated_at": utc_now_iso(),
            "payload": payload,
        }
        with self._runtime_state_path().open("w", encoding="utf-8") as handle:
            json.dump(state, handle, default=str)

    def _validate_table(self, table: str) -> None:
        if table not in self.TABLES:
            raise ValueError(f"Invalid table '{table}'. Must be one of {self.TABLES}")

    def insert(self, table: str, record: Dict[str, Any]) -> None:
        self._validate_table(table)
        if self.client is not None:
            try:
                self.client.table(table).insert(record).execute()
                return
            except Exception as exc:
                logger.warning("Supabase insert failed for %s. Falling back to local JSONL: %s", table, exc)
        self._append_local(table, record)

    def bulk_insert(self, table: str, records: Iterable[Dict[str, Any]]) -> None:
        self._validate_table(table)
        if self.client is not None:
            try:
                self.client.table(table).insert(list(records)).execute()
                return
            except Exception as exc:
                logger.warning("Supabase bulk insert failed for %s: %s", table, exc)
        for record in records:
            self._append_local(table, record)

    def read(self, table: str, limit: int = 100) -> List[Dict[str, Any]]:
        self._validate_table(table)
        if self.client is not None:
            try:
                response = self.client.table(table).select("*").limit(limit).order(
                    "created_at", desc=True
                ).execute()
                return response.data or []
            except Exception as exc:
                logger.warning("Supabase read failed for %s. Falling back to local JSONL: %s", table, exc)
        rows = self._read_local(table)
        return list(reversed(rows[-limit:]))

    def log_prediction(self, record: Dict[str, Any]) -> None:
        self.insert("predictions", record)

    def log_trade(self, record: Dict[str, Any]) -> None:
        self.insert("trades", record)

    def log_equity(self, record: Dict[str, Any]) -> None:
        self.insert("equity_curve", record)

    def log_learning_event(self, record: Dict[str, Any]) -> None:
        self.insert("learning_events", record)

    def save_model_weights(self, record: Dict[str, Any]) -> None:
        self.insert("model_weights", record)

    def recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.read("trades", limit=limit)

    def recent_predictions(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.read("predictions", limit=limit)

    def equity_curve(self, limit: int = 500) -> List[Dict[str, Any]]:
        return self.read("equity_curve", limit=limit)

    def learning_progress(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.read("learning_events", limit=limit)

    def latest_model_weights(self) -> Dict[str, Any]:
        rows = self.read("model_weights", limit=1)
        return rows[0] if rows else {}

    def write_runtime_state(self, state_key: str, payload: Dict[str, Any]) -> None:
        record = {
            "state_key": state_key,
            "updated_at": utc_now_iso(),
            "payload": payload,
        }
        if self.client is not None:
            try:
                self.client.table("runtime_state").upsert(record).execute()
                return
            except Exception as exc:
                logger.warning("Supabase upsert failed for runtime_state[%s]. Falling back locally: %s", state_key, exc)
        self._write_runtime_state_local(state_key, payload)

    def read_runtime_state(self, state_key: str) -> Dict[str, Any]:
        if self.client is not None:
            try:
                response = (
                    self.client.table("runtime_state")
                    .select("*")
                    .eq("state_key", state_key)
                    .limit(1)
                    .execute()
                )
                rows = response.data or []
                if rows:
                    return rows[0].get("payload") or {}
            except Exception as exc:
                logger.warning("Supabase read failed for runtime_state[%s]. Falling back locally: %s", state_key, exc)
        state = self._read_runtime_state_local()
        row = state.get(state_key) or {}
        return row.get("payload") or {}

    def dashboard_snapshot(self) -> Dict[str, Any]:
        predictions = self.recent_predictions(limit=20)
        trades = self.recent_trades(limit=20)
        weights = self.latest_model_weights()
        learning = self.learning_progress(limit=20)
        equity_curve = self.equity_curve(limit=200)
        worker_status = self.read_runtime_state("worker_status")
        model_performance = self.read_runtime_state("model_performance")

        current_strategy = None
        current_model = None
        if predictions:
            last = predictions[0]
            current_strategy = last.get("selected_strategy")
            current_model = last.get("most_influential_model")

        return {
            "current_strategy": current_strategy,
            "most_influential_model": current_model,
            "recent_predictions": predictions,
            "recent_trades": trades,
            "model_weights": weights,
            "model_performance": model_performance,
            "learning_events": learning,
            "equity_curve": equity_curve,
            "worker_status": worker_status,
        }
