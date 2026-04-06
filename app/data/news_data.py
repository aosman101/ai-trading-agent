from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import feedparser
import httpx

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class NewsDataService:
    def __init__(self) -> None:
        self.settings = get_settings()

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
                }
                for item in rows[:limit]
            ]
        except Exception as exc:
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
                    items.append(
                        {
                            "title": entry.get("title", ""),
                            "summary": entry.get("summary", ""),
                            "source": feed.feed.get("title", "rss"),
                            "published_at": entry.get("published", ""),
                            "url": entry.get("link", ""),
                        }
                    )
            except Exception as exc:
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
                        "published_at": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                        "url": submission.url,
                    }
                )
            return posts
        except Exception as exc:
            logger.warning("Reddit fetch failed for %s: %s", query, exc)
            return []

    def collect_text_corpus(self, symbol: str, limit: int = 25) -> List[str]:
        rows = self.fetch_alpha_vantage_news(symbol, limit=limit)
        if not rows:
            rows = self.fetch_rss_news(symbol, limit=limit)
        rows.extend(self.fetch_reddit_posts(symbol, limit=max(5, limit // 2)))
        texts = []
        seen = set()
        for row in rows:
            text = " ".join([row.get("title", ""), row.get("summary", "")]).strip()
            if text and text not in seen:
                seen.add(text)
                texts.append(text)
        return texts[:limit]
