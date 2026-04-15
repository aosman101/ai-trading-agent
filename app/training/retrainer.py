from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from app.backtesting.engine import WalkForwardBacktester
from app.config import get_settings
from app.data.market_data import MarketDataService
from app.models.finbert_sentiment import FinBERTSentimentModel
from app.models.itransformer_model import ITransformerForecaster
from app.rl.dqn_agent import DQNTradingAgent
from app.rl.ppo_agent import PPOTradingAgent
from app.rl.trading_env import TradingEnvironment
from app.strategies.breakout import BreakoutStrategy
from app.strategies.mean_reversion import MeanReversionStrategy
from app.strategies.momentum import MomentumStrategy
from app.strategies.sentiment_strategy import SentimentStrategy
from app.strategies.trend_following import TrendFollowingStrategy
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelBundle:
    finbert: FinBERTSentimentModel
    ppo: PPOTradingAgent
    dqn: DQNTradingAgent
    itransformer: ITransformerForecaster | None
    backtester: WalkForwardBacktester


class ModelTrainer:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.market_data = MarketDataService()
        self.backtester = WalkForwardBacktester()

    def _model_files(self) -> dict[str, Path]:
        base = self.settings.model_path
        return {
            "ppo": base / "ppo_agent",
            "dqn": base / "dqn_agent",
            "itransformer": base / "itransformer.pkl",
        }

    def _build_rl_frame(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
        work = frame.copy()
        strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            TrendFollowingStrategy(),
            BreakoutStrategy(),
            SentimentStrategy(),
        ]
        for strategy in strategies:
            work[f"signal_{strategy.name}"] = strategy.generate_series(work)
        feature_columns = [
            column
            for column in work.columns
            if column not in {
                "symbol",
                "ds",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "forward_return_1",
                "forward_return_5",
                "forward_return_10",
                "target_up_1",
                "target_up_5",
                "target_up_10",
            }
        ]
        return work.dropna().copy(), feature_columns

    def bootstrap_all(self, symbols: Iterable[str] | None = None, *, persist: bool = True) -> ModelBundle:
        symbols = list(symbols or self.settings.symbols)
        universe_frame = self.market_data.fetch_universe_history(symbols=symbols)

        itransformer = None
        try:
            itransformer = ITransformerForecaster()
            itransformer.fit(universe_frame)
        except Exception as exc:
            logger.warning("iTransformer bootstrap skipped: %s", exc)

        rl_frame, feature_columns = self._build_rl_frame(universe_frame)
        env = TradingEnvironment(rl_frame, feature_columns=feature_columns, return_column="forward_return_1")
        ppo = PPOTradingAgent()
        ppo.fit(env, total_timesteps=20_000)
        dqn = DQNTradingAgent()
        dqn.fit(env, total_timesteps=20_000)

        finbert = FinBERTSentimentModel()

        bundle = ModelBundle(
            finbert=finbert,
            ppo=ppo,
            dqn=dqn,
            itransformer=itransformer,
            backtester=self.backtester,
        )
        if persist:
            self.save(bundle)
        return bundle

    def save(self, bundle: ModelBundle) -> None:
        files = self._model_files()
        bundle.ppo.save(files["ppo"])
        bundle.dqn.save(files["dqn"])
        if bundle.itransformer is not None:
            bundle.itransformer.save(files["itransformer"])

    def load(self) -> ModelBundle:
        files = self._model_files()

        ppo = PPOTradingAgent()
        ppo.load(files["ppo"])

        dqn = DQNTradingAgent()
        dqn.load(files["dqn"])

        itransformer = None
        if files["itransformer"].exists():
            try:
                itransformer = ITransformerForecaster()
                itransformer.load(files["itransformer"])
            except Exception as exc:
                logger.warning("Failed to load iTransformer: %s", exc)

        finbert = FinBERTSentimentModel()

        return ModelBundle(
            finbert=finbert,
            ppo=ppo,
            dqn=dqn,
            itransformer=itransformer,
            backtester=self.backtester,
        )

    def load_or_bootstrap(self, symbols: Iterable[str] | None = None) -> ModelBundle:
        files = self._model_files()
        required = [Path(str(files["ppo"]) + ".zip"), Path(str(files["dqn"]) + ".zip")]
        if all(path.exists() for path in required):
            logger.info("Loading existing trained models from disk")
            return self.load()

        logger.info("No full model bundle found, bootstrapping models from fresh history")
        return self.bootstrap_all(symbols=symbols)
