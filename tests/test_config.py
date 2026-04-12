from __future__ import annotations

import pytest

from app.config import Settings


def test_runtime_validation_requires_matching_paper_mode():
    settings = Settings(
        trading_mode="paper",
        alpaca_paper=False,
        dsi_base_url="",
        dsi_email="",
        dsi_password="",
    )

    with pytest.raises(ValueError, match="TRADING_MODE=paper requires ALPACA_PAPER=true"):
        settings.validate_runtime_configuration()


def test_runtime_validation_requires_complete_dsi_configuration():
    settings = Settings(
        dsi_base_url="https://dsi.example.com",
        dsi_email="",
        dsi_password="",
    )

    with pytest.raises(ValueError, match="DSI configuration must include"):
        settings.validate_runtime_configuration()


def test_api_runtime_validation_requires_token_in_non_dev():
    settings = Settings(
        environment="prod",
        api_bearer_token="",
        dsi_base_url="",
        dsi_email="",
        dsi_password="",
    )

    with pytest.raises(ValueError, match="API_BEARER_TOKEN must be set"):
        settings.validate_runtime_configuration(component="api")
