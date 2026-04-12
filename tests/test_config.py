from __future__ import annotations

from app.config import Settings


def test_runtime_validation_requires_matching_paper_mode():
    settings = Settings(
        trading_mode="paper",
        alpaca_paper=False,
    )

    try:
        settings.validate_runtime_configuration()
        assert False, "Expected runtime validation to fail"
    except ValueError as exc:
        assert "TRADING_MODE=paper requires ALPACA_PAPER=true" in str(exc)


def test_runtime_validation_requires_complete_dsi_configuration():
    settings = Settings(
        dsi_base_url="https://dsi.example.com",
        dsi_email="",
        dsi_password="",
    )

    try:
        settings.validate_runtime_configuration()
        assert False, "Expected runtime validation to fail"
    except ValueError as exc:
        assert "DSI configuration must include" in str(exc)


def test_api_runtime_validation_requires_token_in_non_dev():
    settings = Settings(
        environment="prod",
        api_bearer_token="",
    )

    try:
        settings.validate_runtime_configuration(component="api")
        assert False, "Expected runtime validation to fail"
    except ValueError as exc:
        assert "API_BEARER_TOKEN must be set" in str(exc)
