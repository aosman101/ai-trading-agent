# Safe transition from paper to live

Do not move to live trading until all of these are true.

## Stage 1: Historical validation

Backtest every strategy on at least 5 years of data and require:

- win rate above your minimum threshold
- Sharpe ratio above your minimum threshold
- max drawdown below your maximum threshold
- positive total return
- stable performance across several symbols, not just one

## Stage 2: Paper trading

Run the system in paper mode for at least 30 days.

Review:

- fill quality
- slippage vs expectations
- stop loss behavior
- number of skipped trades
- whether daily loss limits work
- whether model confidence aligns with good outcomes

## Stage 3: Tiny live launch

When paper metrics hold up:

- keep `ALLOW_SHORTING=false`
- start with 10% to 25% of the capital you eventually plan to deploy
- reduce `MAX_RISK_PER_TRADE` to 0.5% for the first live month
- trade only the top 1 to 3 symbols

## Stage 4: Controlled expansion

Only increase size after:

- 30 more live days
- no risk rule breaches
- no operational failures
- positive expectancy after fees and slippage

## Live settings checklist

Before switching:

```env
TRADING_MODE=live
ENABLE_LIVE_TRADING=true
ALLOW_SHORTING=false
KILL_SWITCH=false
```

For your first live phase, keep shorting off. Learn the system behavior first.

## When to hit the kill switch

Turn `KILL_SWITCH=true` immediately if:

- orders are being duplicated
- stop losses are not attaching
- dashboard data is stale
- the model starts trading with obviously bad inputs
- you hit an unexplained drawdown
