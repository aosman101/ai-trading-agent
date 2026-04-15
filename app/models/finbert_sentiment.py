from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Iterable, List

from app.config import get_settings
from app.types import ModelSignal
from app.utils.logging import get_logger

logger = get_logger(__name__)

_TEXT_CACHE_MAX = 2048


class FinBERTSentimentModel:
    def __init__(self) -> None:
        self.settings = get_settings()
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:
            raise ImportError("transformers is required for FinBERTSentimentModel") from exc

        device = -1 if self.settings.hf_device.lower() == "cpu" else 0
        self.pipeline = pipeline(
            task="sentiment-analysis",
            model=self.settings.finbert_model_name,
            tokenizer=self.settings.finbert_model_name,
            device=device,
            truncation=True,
        )
        self._text_cache: "OrderedDict[str, float]" = OrderedDict()
        self._cache_lock = threading.Lock()

    @staticmethod
    def _label_to_score(label: str, confidence: float) -> float:
        normalized = label.lower().strip()
        if normalized.startswith("pos"):
            return confidence
        if normalized.startswith("neg"):
            return -confidence
        return 0.0

    @staticmethod
    def _text_key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    def _cache_get(self, key: str) -> float | None:
        with self._cache_lock:
            value = self._text_cache.get(key)
            if value is not None:
                self._text_cache.move_to_end(key)
            return value

    def _cache_put(self, key: str, value: float) -> None:
        with self._cache_lock:
            self._text_cache[key] = value
            self._text_cache.move_to_end(key)
            while len(self._text_cache) > _TEXT_CACHE_MAX:
                self._text_cache.popitem(last=False)

    def score_texts(self, texts: Iterable[str]) -> float:
        truncated = [text[:2000] for text in texts if text]
        if not truncated:
            return 0.0
        scores: list[float] = []
        uncached_texts: list[str] = []
        uncached_keys: list[str] = []
        for text in truncated:
            key = self._text_key(text)
            cached = self._cache_get(key)
            if cached is not None:
                scores.append(cached)
            else:
                uncached_texts.append(text)
                uncached_keys.append(key)
        if uncached_texts:
            results = self.pipeline(uncached_texts)
            for key, item in zip(uncached_keys, results):
                value = self._label_to_score(item["label"], float(item["score"]))
                self._cache_put(key, value)
                scores.append(value)
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def predict_latest(self, symbol: str, texts: List[str]) -> ModelSignal:
        sentiment_score = self.score_texts(texts)
        direction = "flat"
        if sentiment_score > 0.10:
            direction = "long"
        elif sentiment_score < -0.10:
            direction = "short"
        return ModelSignal(
            name="finbert",
            symbol=symbol,
            direction=direction,
            score=sentiment_score,
            confidence=abs(sentiment_score),
            metadata={"samples_scored": len(texts)},
        )
