# How the agent self-improves

This build uses three different learning loops.

## 1) Reinforcement learning loop

Every trading state is converted into an observation that includes:

- technical indicators
- model forecasts
- recent strategy signals
- current portfolio state

The PPO and DQN agents learn a policy from reward:

- positive reward for profitable, efficient trades
- negative reward for losses
- extra penalty for drawdown and turnover

## 2) Supervised model retraining loop

Nightly retraining refreshes:

- NHITS on rolling historical price data
- LightGBM on updated feature labels
- TFT on the latest rolling window

This means the forecasting models adapt as the market regime changes.

## 3) Meta-learning loop

The ensemble keeps track of which models have been working recently by
scoring recent model signals against realized next-bar returns.

For each model, the worker updates:

- directional accuracy
- realized signal Sharpe
- confidence calibration error
- realized drawdown from recent signal returns

If one model has better recent live performance, its ensemble weight rises automatically.
If it degrades, its weight falls.

## Why not update everything after every single trade?

Because that is usually a fast route to overfitting and unstable behavior.

The safer pattern is:

- log every trade immediately
- update ensemble weights from recent realized outcomes
- retrain supervised models on a schedule
- retrain RL policies from the full experience buffer

That still makes the system self-learning, but with much better stability.
