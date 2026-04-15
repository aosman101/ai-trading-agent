from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import feedparser
import httpx
import pandas as pd

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_NEWS_CACHE_TTL_SEC = 1800  # 30 minutes — news doesn't change every cycle


class NewsDataService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._cache: dict[tuple[str, str], tuple[float, Any]] = {}
        self._cache_lock = threading.Lock()

    def _cache_get(self, key: tuple[str, str]) -> Any | None:
        with self._cache_lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            timestamp, value = entry
            if time.monotonic() - timestamp > _NEWS_CACHE_TTL_SEC:
                self._cache.pop(key, None)
                return None
            return value

    def _cache_set(self, key: tuple[str, str], value: Any) -> None:
        with self._cache_lock:
            self._cache[key] = (time.monotonic(), value)

    def fetch_alpha_vantage_news(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.settings.alpha_vantage_api_key:
            return []
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": self.settings.alpha_vantage_api_key,
            "limit": limit,
        }
        try:
            response = httpx.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("feed", [])
            return [
                {
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", "alphavantage"),
                    "published_at": item.get("time_published", ""),
                    "url": item.get("url", ""),
                    "sentiment_score": item.get("overall_sentiment_score"),
                }
                for item in rows[:limit]
            ]
        except (httpx.HTTPError, httpx.TimeoutException, ValueError, KeyError) as exc:
            logger.warning("Alpha Vantage news fetch failed for %s: %s", symbol, exc)
            return []

    def fetch_rss_news(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        urls = [token.strip() for token in self.settings.news_rss_urls.split(",") if token.strip()]
        items: list[dict[str, Any]] = []
        for template in urls:
            url = template.format(symbol=symbol)
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:limit]:
                    items.append({
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "source": feed.feed.get("title", "rss"),
                        "published_at": entry.get("published", ""),
                        "url": entry.get("link", ""),
                        "sentiment_score": None,
                    })
            except (ValueError, KeyError, AttributeError) as exc:
                logger.warning("RSS news fetch failed for %s: %s", symbol, exc)
        return items[:limit]

    def fetch_reddit_posts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.settings.reddit_client_id or not self.settings.reddit_client_secret:
            return []
        try:
            import praw  # type: ignore
        except Exception:
            logger.warning("praw is not installed or could not be imported")
            return []

        try:
            reddit = praw.Reddit(
                client_id=self.settings.reddit_client_id,
                client_secret=self.settings.reddit_client_secret,
                user_agent=self.settings.reddit_user_agent,
            )
            posts = []
            for submission in reddit.subreddit("stocks+investing+wallstreetbets").search(
                query, limit=limit, sort="new"
            ):
                posts.append(
                    {
                        "title": submission.title,
                        "summary": submission.selftext[:1000],
                        "source": f"reddit:{submission.subreddit.display_name}",
                        "published_at": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
                        "url": submission.url,
                        "sentiment_score": None,
                    }
                )
            return posts
        except (ValueError, KeyError, AttributeError, OSError) as exc:
            logger.warning("Reddit fetch failed for %s: %s", query, exc)
            return []

    def collect_text_corpus(self, symbol: str, limit: int = 25) -> List[str]:
        rows = self.collect_news_rows(symbol, limit=limit)
        texts = []
        seen = set()
        for row in rows:
            text = " ".join([row.get("title", ""), row.get("summary", "")]).strip()
            if text and text not in seen:
                seen.add(text)
                texts.append(text)
        return texts[:limit]

    def collect_news_rows(self, symbol: str, limit: int = 25) -> List[Dict[str, Any]]:
        rows = self.fetch_alpha_vantage_news(symbol, limit=max(limit, 100))
        if not rows:
            rows = self.fetch_rss_news(symbol, limit=limit)
        rows.extend(self.fetch_reddit_posts(symbol, limit=max(5, limit // 2)))

        deduped: list[dict[str, Any]] = []
        seen = set()
        for row in rows:
            key = (
                row.get("url") or "",
                row.get("published_at") or "",
                row.get("title") or "",
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped[: max(limit, 100)]

    @staticmethod
    def _parse_published_at(value: Any) -> pd.Timestamp | None:
        if not value:
            return None
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            try:
                parsed = pd.to_datetime(str(value), format="%Y%m%dT%H%M%S", utc=True, errors="coerce")
            except Exception:
                return None
        if pd.isna(parsed):
            return None
        return pd.Timestamp(parsed).tz_convert(None)

    def sentiment_time_series(
        self,
        symbol: str,
        index: pd.Index,
        fallback_latest: float | None = None,
        limit: int = 100,
    ) -> pd.Series:
        dt_index = pd.DatetimeIndex(index)
        if dt_index.tz is not None:
            dt_index = dt_index.tz_convert(None)
        if dt_index.empty:
            return pd.Series(dtype=float)

        rows = self.fetch_alpha_vantage_news(symbol, limit=limit)
        daily_scores: list[tuple[pd.Timestamp, float]] = []
        for row in rows:
            score = row.get("sentiment_score")
            if score in {None, ""}:
                continue
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                continue
            published_at = self._parse_published_at(row.get("published_at"))
            if published_at is None:
                continue
            daily_scores.append((published_at.normalize(), max(-1.0, min(1.0, score_value))))

        normalized_index = pd.DatetimeIndex(dt_index.normalize())
        if daily_scores:
            score_frame = pd.DataFrame(daily_scores, columns=["date", "score"])
            daily = score_frame.groupby("date", sort=True)["score"].mean().sort_index()
            aligned = daily.reindex(normalized_index).ffill().fillna(0.0)
            sentiment_series = pd.Series(aligned.to_numpy(), index=dt_index, dtype=float)
        else:
            sentiment_series = pd.Series(0.0, index=dt_index, dtype=float)

        if fallback_latest is not None and not sentiment_series.empty:
            sentiment_series.iloc[-1] = float(fallback_latest)
        return sentiment_series.clip(lower=-1.0, upper=1.0)
