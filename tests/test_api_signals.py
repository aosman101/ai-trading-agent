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
    monkeypatch.setattr(server.settings, "api_bearer_token", "")
    monkeypatch.setattr(server.settings, "worker_heartbeat_tolerance_minutes", 90)
    monkeypatch.setattr(server.broker, "client", None)
    monkeypatch.setattr(server.repository, "client", None)


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
