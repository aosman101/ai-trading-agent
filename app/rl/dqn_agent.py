from __future__ import annotations

from pathlib import Path

import numpy as np

from app.types import ModelSignal


class DQNTradingAgent:
    def __init__(self) -> None:
        try:
            from stable_baselines3 import DQN  # type: ignore
            from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
        except Exception as exc:
            raise ImportError("stable-baselines3 is required for DQNTradingAgent") from exc

        self._DQN = DQN
        self._DummyVecEnv = DummyVecEnv
        self.model = None

    def fit(self, env, total_timesteps: int = 50_000) -> None:
        vec_env = self._DummyVecEnv([lambda: env])
        self.model = self._DQN("MlpPolicy", vec_env, verbose=0)
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation: np.ndarray, symbol: str) -> ModelSignal:
        if self.model is None:
            return ModelSignal(name="dqn", symbol=symbol, direction="flat", score=0.0, confidence=0.0)
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
            name="dqn",
            symbol=symbol,
            direction=direction,
            score=score,
            confidence=0.60 if direction != "flat" else 0.30,
            metadata={"action": action},
        )

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained DQN model")
        self.model.save(str(path))

    def load(self, path: str | Path) -> None:
        self.model = self._DQN.load(str(path))
