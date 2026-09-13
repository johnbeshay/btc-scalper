# BTC scalper — Phase 1

Decision-support for short-horizon Bitcoin trading. Five agents read the same
candle feed, each returns one signal, and an orchestrator folds them into a
single verdict. You make the trades; this tells you what it sees.

No dependencies. Python 3.10+ and the standard library.

## Run it

```bash
python3 simulate.py              # offline, synthetic candles
python3 run.py --once            # one live read
python3 run.py                   # live loop, 60s refresh
python3 run.py --json            # JSON out, for the dashboard
python3 run.py --taker 0.25      # your real fee tier
```

Binance is geo-blocked in the US, so Coinbase is the default. `--exchange
binance` if you are somewhere it works.

## Layout

```
core/
  feed.py          Candle model, Coinbase and Binance feeds
  indicators.py    SMA, ATR, RSI, swing points, level clustering, FeeModel
  base.py          Agent contract, Signal, Direction, Confidence
  agents.py        The five Tier 1 agents
  orchestrator.py  Runs agents, aggregates, applies the veto
run.py             Live CLI
simulate.py        Offline harness across three market regimes
```

## The agents

| Agent | Reads | Says |
|---|---|---|
| `price_action` | last 3 candles vs 40-candle norm | is this move unusual |
| `volume` | current vs 20-candle average | is anyone participating |
| `levels` | fractal swings, clustered | where support and resistance sit |
| `trend` | 9/21/50 SMA stack, RSI | which way the bias runs |
| `volatility` | ATR percentile | is there enough movement to bother |

## The fee model

This is the part that makes the system honest. `FeeModel.round_trip_pct` is
entry fee + exit fee + slippage both ways. At the default worst-case Coinbase
tier that is **1.30%** — the price must move that far before you keep a cent.

Two consequences are wired through everything:

1. Every directional signal carries `tradeable`. A signal whose expected move
   does not beat costs by 1.5x is marked `[below cost]`.
2. The volatility agent holds a **veto**. If the typical 15-minute range is
   smaller than the round trip, the verdict is `Stand down` regardless of how
   many agents agree.

Set `--taker` to your real tier. Everything downstream keys off it.

## Adding an agent

Subclass `Agent`, set `name` and `warmup`, implement `analyze()`, add it to
`TIER_1`. The orchestrator needs no changes.

```python
class MyAgent(Agent):
    name = "my_agent"
    warmup = 30

    def analyze(self, candles):
        return Signal(
            agent=self.name,
            direction=Direction.BULLISH,
            confidence=Confidence.MEDIUM,
            headline="Something happened",
            detail="Longer explanation.",
            expected_move_pct=0.8,
            metrics={"whatever": 1},
        )
```

## Market data (read-only)

`core/kalshi_api.py` reads Kalshi's public market endpoints - no login, no
order placement. The logger uses it to record the exchange's real strikes and
top-of-book alongside every prediction, so `score.py` can measure the model
against the price it would actually pay rather than against always-50%.

Records written this way carry `"schema": 2`; older lines are still read.

## Not built yet

Phase 2 is news/sentiment. Phase 3 is trade logging and per-agent win rates,
which is the piece that eventually tells you which of these five are actually
worth listening to. Until that exists, treat every signal as unproven.

## Before real money

Log the signals for a few weeks without trading them. Compare what the system
said against what price did. Five agents that agree can still be wrong
together, and the only way to find out is a paper record.
