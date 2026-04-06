from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
from typing import Any

import joblib

from app.utils.logging import get_logger

logger = get_logger(__name__)

_SECRET = os.environ.get("MODEL_HMAC_SECRET", "ai-trading-agent-default-key").encode()
_HASH_SUFFIX = ".sha256"


def _hmac_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + _HASH_SUFFIX)


def _compute_hmac(data: bytes) -> str:
    return hmac.new(_SECRET, data, hashlib.sha256).hexdigest()


def save_model(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)
    file_bytes = path.read_bytes()
    _hmac_path(path).write_text(_compute_hmac(file_bytes))
    logger.debug("Saved model to %s with integrity hash", path)


def load_model(path: str | Path) -> Any:
    path = Path(path)
    hash_file = _hmac_path(path)
    if hash_file.exists():
        file_bytes = path.read_bytes()
        expected = hash_file.read_text().strip()
        actual = _compute_hmac(file_bytes)
        if not hmac.compare_digest(expected, actual):
            raise SecurityError(
                f"Integrity check failed for {path}. "
                "The model file may have been tampered with."
            )
    else:
        logger.warning(
            "No integrity hash found for %s. "
            "Run a fresh training cycle to generate one.",
            path,
        )
    return joblib.load(path)


class SecurityError(Exception):
    pass
