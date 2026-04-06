from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.types import ModelSignal


class PPOTradingAgent:
    def __init__(self) -> None:
        try:
            from stable_baselines3 import PPO  # type: ignore
            from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
        except Exception as exc:
            raise ImportError("stable-baselines3 is required for PPOTradingAgent") from exc

        self._PPO = PPO
        self._DummyVecEnv = DummyVecEnv
        self.model = None

    def fit(self, env, total_timesteps: int = 50_000) -> None:
        vec_env = self._DummyVecEnv([lambda: env])
        self.model = self._PPO("MlpPolicy", vec_env, verbose=0)
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation: np.ndarray, symbol: str) -> ModelSignal:
        if self.model is None:
            return ModelSignal(name="ppo", symbol=symbol, direction="flat", score=0.0, confidence=0.0)
        action, _ = self.model.predict(observation, deterministic=True)
        action = int(action)
        direction = "flat"
        score = 0.0
        if action == 1:
            direction = "long"
            score = 1.0
        elif action == 2:
            direction = "short"
            score = -1.0
        return ModelSignal(
            name="ppo",
            symbol=symbol,
            direction=direction,
            score=score,
            confidence=0.65 if direction != "flat" else 0.35,
            metadata={"action": action},
        )

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained PPO model")
        self.model.save(str(path))

    def load(self, path: str | Path) -> None:
        self.model = self._PPO.load(str(path))
