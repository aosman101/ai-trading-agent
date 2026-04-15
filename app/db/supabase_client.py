from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.config import get_settings
from app.utils.logging import get_logger
from app.utils.time import utc_now_iso

logger = get_logger(__name__)


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class TradeRepository:
    TABLES = (
        "predictions",
        "trades",
        "equity_curve",
        "learning_events",
        "model_weights",
        "runtime_state",
        "external_signals",
        "journal",
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
        if "id" not in record and "_local_id" not in record:
            record = {**record, "_local_id": self._next_local_id(table)}
        with self._path(table).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def _next_local_id(self, table: str) -> int:
        rows = self._read_local(table)
        if not rows:
            return 1
        max_id = max(
            int(r.get("id") or r.get("_local_id") or 0) for r in rows
        )
        return max_id + 1

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

    # --- External signals ---

    def submit_external_signal(self, record: Dict[str, Any]) -> None:
        self.insert("external_signals", record)

    def pending_external_signals(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Return unconsumed external signals for *symbol*."""
        if self.client is not None:
            try:
                response = (
                    self.client.table("external_signals")
                    .select("*")
                    .eq("symbol", symbol)
                    .is_("consumed_at", "null")
                    .order("created_at", desc=False)
                    .limit(limit)
                    .execute()
                )
                return response.data or []
            except Exception as exc:
                logger.warning("Supabase pending_external_signals failed: %s", exc)
        rows = self._read_local("external_signals")
        return [
            row for row in rows
            if row.get("symbol") == symbol and not row.get("consumed_at")
        ][-limit:]

    def mark_signals_consumed(self, signal_ids: List[int | str]) -> None:
        """Mark signals as consumed so they aren't re-read."""
        now = utc_now_iso()
        if self.client is not None:
            try:
                self.client.table("external_signals").update(
                    {"consumed_at": now}
                ).in_("id", signal_ids).execute()
                return
            except Exception as exc:
                logger.warning("Supabase mark_signals_consumed failed: %s", exc)
        # Local fallback: rewrite the JSONL (signals are small, this is fine)
        path = self._path("external_signals")
        if not path.exists():
            return
        ids_set = set(str(sid) for sid in signal_ids)
        rows = self._read_local("external_signals")
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                if str(row.get("id", "")) in ids_set or str(row.get("_local_id", "")) in ids_set:
                    row["consumed_at"] = now
                handle.write(json.dumps(row, default=str) + "\n")

    def list_external_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.read("external_signals", limit=limit)

    def recent_external_signals(self, limit: int = 500, source: str | None = None) -> List[Dict[str, Any]]:
        if self.client is not None:
            try:
                query = self.client.table("external_signals").select("*").order("created_at", desc=True).limit(limit)
                if source:
                    query = query.eq("source", source)
                response = query.execute()
                return response.data or []
            except Exception as exc:
                logger.warning("Supabase recent_external_signals failed: %s", exc)
        rows = self._read_local("external_signals")
        if source:
            rows = [row for row in rows if row.get("source") == source]
        return list(reversed(rows[-limit:]))

    def count_recent_external_signals(self, source: str, caller_id: str, window_seconds: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        total = 0
        for row in self.recent_external_signals(limit=1000, source=source):
            created_at = _parse_timestamp(row.get("created_at"))
            if created_at is None or created_at < cutoff:
                continue
            payload = row.get("payload") or {}
            if str(payload.get("caller_id", "")) == caller_id:
                total += 1
        return total

    def has_recent_external_signal_idempotency(self, source: str, key: str, ttl_seconds: int) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
        for row in self.recent_external_signals(limit=2000, source=source):
            created_at = _parse_timestamp(row.get("created_at"))
            if created_at is None or created_at < cutoff:
                continue
            payload = row.get("payload") or {}
            if str(payload.get("idempotency_key", "")) == key:
                return True
        return False

    # --- Journal ---

    def log_journal_entry(self, record: Dict[str, Any]) -> None:
        self.insert("journal", record)

    def recent_journal(self, limit: int = 50, symbol: str | None = None) -> List[Dict[str, Any]]:
        if self.client is not None and symbol:
            try:
                response = (
                    self.client.table("journal")
                    .select("*")
                    .eq("symbol", symbol)
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )
                return response.data or []
            except Exception as exc:
                logger.warning("Supabase journal read failed: %s", exc)
        if self.client is not None and not symbol:
            try:
                response = (
                    self.client.table("journal")
                    .select("*")
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )
                return response.data or []
            except Exception as exc:
                logger.warning("Supabase journal read failed: %s", exc)
        rows = self._read_local("journal")
        if symbol:
            rows = [r for r in rows if r.get("symbol") == symbol]
        return list(reversed(rows[-limit:]))

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
