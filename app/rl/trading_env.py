from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnvironment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        return_column: str = "forward_return_1",
        transaction_cost: float = 0.0005,
        drawdown_penalty: float = 0.10,
    ) -> None:
        super().__init__()
        self.frame = frame.dropna(subset=list(feature_columns) + [return_column]).reset_index(drop=True)
        self.feature_columns = list(feature_columns)
        self.return_column = return_column
        self.transaction_cost = transaction_cost
        self.drawdown_penalty = drawdown_penalty

        obs_size = len(self.feature_columns) + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self._index = 0
        self.position = 0.0
        self.portfolio_value = 1.0
        self.max_portfolio_value = 1.0

    def _get_observation(self) -> np.ndarray:
        row = self.frame.iloc[self._index]
        features = row[self.feature_columns].astype(float).fillna(0.0).to_numpy(dtype=np.float32)
        drawdown = 1.0 - (self.portfolio_value / self.max_portfolio_value)
        extras = np.array(
            [self.position, self.portfolio_value, drawdown],
            dtype=np.float32,
        )
        return np.concatenate([features, extras]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._index = 0
        self.position = 0.0
        self.portfolio_value = 1.0
        self.max_portfolio_value = 1.0
        return self._get_observation(), {}

    def step(self, action: int):
        position_before = self.position
        if action == 1:
            self.position = 1.0
        elif action == 2:
            self.position = -1.0
        elif action == 3:
            self.position = 0.0

        row = self.frame.iloc[self._index]
        realized_return = float(row[self.return_column])
        trade_cost = abs(self.position - position_before) * self.transaction_cost

        portfolio_return = self.position * realized_return - trade_cost
        previous_value = self.portfolio_value
        self.portfolio_value *= 1.0 + portfolio_return
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        drawdown = 1.0 - (self.portfolio_value / self.max_portfolio_value)
        reward = (self.portfolio_value / previous_value - 1.0) - (self.drawdown_penalty * drawdown)

        self._index += 1
        terminated = self._index >= len(self.frame) - 1
        truncated = False
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "drawdown": drawdown,
        }
        observation = self._get_observation() if not terminated else np.zeros_like(self._get_observation())
        return observation, float(reward), terminated, truncated, info
