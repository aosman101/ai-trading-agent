from __future__ import annotations

from typing import Iterable, List

from app.config import get_settings
from app.types import ModelSignal
from app.utils.logging import get_logger

logger = get_logger(__name__)


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

    @staticmethod
    def _label_to_score(label: str, confidence: float) -> float:
        normalized = label.lower().strip()
        if normalized.startswith("pos"):
            return confidence
        if normalized.startswith("neg"):
            return -confidence
        return 0.0

    def score_texts(self, texts: Iterable[str]) -> float:
        batch = [text[:2000] for text in texts if text]
        if not batch:
            return 0.0
        results = self.pipeline(batch)
        scores = [self._label_to_score(item["label"], float(item["score"])) for item in results]
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
