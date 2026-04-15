from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("TRADING_MODE", "paper")

for module_name in ("feedparser", "gymnasium", "httpx", "yfinance"):
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import app.api.server as server
from app.api.server import app, repository


client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_server_settings(monkeypatch):
    signal_store = []

    def submit_external_signal(record):
        signal_store.append(record)

    def list_external_signals(limit: int = 50):
        return list(reversed(signal_store[-limit:]))

    def recent_external_signals(limit: int = 500, source: str | None = None):
        rows = signal_store
        if source:
            rows = [row for row in rows if row.get("source") == source]
        return list(reversed(rows[-limit:]))

    def count_recent_external_signals(source: str, caller_id: str, window_seconds: int):
        return sum(
            1
            for row in signal_store
            if row.get("source") == source and (row.get("payload") or {}).get("caller_id") == caller_id
        )

    def has_recent_external_signal_idempotency(source: str, key: str, ttl_seconds: int):
        return any(
            row.get("source") == source and (row.get("payload") or {}).get("idempotency_key") == key
            for row in signal_store
        )

    monkeypatch.setattr(server.settings, "api_bearer_token", "")
    monkeypatch.setattr(server.settings, "worker_heartbeat_tolerance_minutes", 90)
    monkeypatch.setattr(server.broker, "client", None)
    monkeypatch.setattr(server.repository, "client", None)
    monkeypatch.setattr(server.repository, "submit_external_signal", submit_external_signal)
    monkeypatch.setattr(server.repository, "list_external_signals", list_external_signals)
    monkeypatch.setattr(server.repository, "recent_external_signals", recent_external_signals)
    monkeypatch.setattr(server.repository, "count_recent_external_signals", count_recent_external_signals)
    monkeypatch.setattr(server.repository, "has_recent_external_signal_idempotency", has_recent_external_signal_idempotency)
    monkeypatch.setattr(server.repository, "recent_journal", lambda limit=50, symbol=None: [])


class TestSubmitSignal:
    def test_submit_valid_signal(self):
        response = client.post(
            "/api/signals",
            json={
                "symbol": "AAPL",
                "direction": "long",
                "score": 0.7,
                "confidence": 0.8,
                "source": "my_website",
                "reasoning": "Breakout above resistance",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "accepted"
        assert body["symbol"] == "AAPL"
        assert body["direction"] == "long"

    def test_submit_minimal_signal(self):
        response = client.post(
            "/api/signals",
            json={"symbol": "MSFT", "direction": "short"},
        )
        assert response.status_code == 201
        assert response.json()["symbol"] == "MSFT"

    def test_reject_invalid_direction(self):
        response = client.post(
            "/api/signals",
            json={"symbol": "AAPL", "direction": "yolo"},
        )
        assert response.status_code == 422

    def test_reject_score_out_of_range(self):
        response = client.post(
            "/api/signals",
            json={"symbol": "AAPL", "direction": "long", "score": 5.0},
        )
        assert response.status_code == 422

    def test_reject_confidence_out_of_range(self):
        response = client.post(
            "/api/signals",
            json={"symbol": "AAPL", "direction": "long", "confidence": -0.5},
        )
        assert response.status_code == 422

    def test_symbol_uppercased(self):
        response = client.post(
            "/api/signals",
            json={"symbol": "aapl", "direction": "flat"},
        )
        assert response.status_code == 201
        assert response.json()["symbol"] == "AAPL"

    def test_duplicate_idempotency_key_returns_duplicate(self):
        payload = {
            "symbol": "AAPL",
            "direction": "long",
            "source": "my_website",
            "idempotency_key": "abc123",
        }
        first = client.post("/api/signals", json=payload)
        second = client.post("/api/signals", json=payload)

        assert first.status_code == 201
        assert second.status_code == 200
        assert second.json()["status"] == "duplicate"

    def test_rate_limit_uses_persisted_signal_history(self):
        for _ in range(30):
            response = client.post(
                "/api/signals",
                headers={"x-forwarded-for": "203.0.113.9"},
                json={"symbol": "AAPL", "direction": "long", "source": "rate_limit_test"},
            )
            assert response.status_code == 201

        blocked = client.post(
            "/api/signals",
            headers={"x-forwarded-for": "203.0.113.9"},
            json={"symbol": "AAPL", "direction": "long", "source": "rate_limit_test"},
        )
        assert blocked.status_code == 429


class TestListSignals:
    def test_list_returns_200(self):
        response = client.get("/api/signals")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestJournal:
    def test_journal_returns_200(self):
        response = client.get("/api/journal")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_journal_filter_by_symbol(self):
        response = client.get("/api/journal?symbol=AAPL")
        assert response.status_code == 200


class TestApiAuth:
    def test_status_requires_bearer_token_when_configured(self, monkeypatch):
        monkeypatch.setattr(server.settings, "api_bearer_token", "secret-token")

        response = client.get("/api/status")
        assert response.status_code == 401

        authorized = client.get("/api/status", headers={"Authorization": "Bearer secret-token"})
        assert authorized.status_code == 200

    def test_submit_signal_requires_bearer_token_when_configured(self, monkeypatch):
        monkeypatch.setattr(server.settings, "api_bearer_token", "secret-token")

        response = client.post(
            "/api/signals",
            json={"symbol": "AAPL", "direction": "long"},
        )
        assert response.status_code == 401

        authorized = client.post(
            "/api/signals",
            headers={"Authorization": "Bearer secret-token"},
            json={"symbol": "AAPL", "direction": "long"},
        )
        assert authorized.status_code == 201


class TestHealth:
    def test_health_returns_503_when_worker_is_stale(self, monkeypatch):
        monkeypatch.setattr(
            server.repository,
            "dashboard_snapshot",
            lambda: {
                "worker_status": {
                    "last_cycle_at": "2000-01-01T00:00:00+00:00",
                    "dsi_status": {"configured": False},
                }
            },
        )

        response = client.get("/health")

        assert response.status_code == 503
        assert response.json()["status"] == "degraded"

    def test_health_returns_200_when_worker_is_recent(self, monkeypatch):
        monkeypatch.setattr(
            server.repository,
            "dashboard_snapshot",
            lambda: {
                "worker_status": {
                    "last_cycle_at": "2099-01-01T00:00:00+00:00",
                    "dsi_status": {"configured": False},
                }
            },
        )

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
