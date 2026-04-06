from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("TRADING_MODE", "paper")

from app.utils.safe_model_io import SecurityError, load_model, save_model


class TestSafeModelIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        data = {"weights": [1.0, 2.0, 3.0], "name": "test"}
        path = tmp_path / "model.joblib"
        save_model(data, path)
        loaded = load_model(path)
        assert loaded == data

    def test_integrity_hash_created(self, tmp_path):
        path = tmp_path / "model.joblib"
        save_model({"a": 1}, path)
        hash_path = path.with_suffix(".joblib.sha256")
        assert hash_path.exists()
        assert len(hash_path.read_text().strip()) == 64  # SHA-256 hex

    def test_tampered_file_raises_error(self, tmp_path):
        path = tmp_path / "model.joblib"
        save_model({"a": 1}, path)
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        with pytest.raises(SecurityError, match="tampered"):
            load_model(path)

    def test_missing_hash_warns_but_loads(self, tmp_path):
        path = tmp_path / "model.joblib"
        save_model({"a": 1}, path)
        path.with_suffix(".joblib.sha256").unlink()
        loaded = load_model(path)
        assert loaded == {"a": 1}

    def test_complex_objects(self, tmp_path):
        import numpy as np
        data = {"array": np.array([1, 2, 3]), "nested": {"key": "value"}}
        path = tmp_path / "complex.joblib"
        save_model(data, path)
        loaded = load_model(path)
        assert (loaded["array"] == data["array"]).all()
        assert loaded["nested"] == data["nested"]
