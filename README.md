# btc-scalper

Probability estimates for Kalshi's 15-minute Bitcoin contracts, and the
machinery to find out whether those estimates are worth anything.

A volatility model prices P(BTC above strike at window close). Five adjusters
modify the volatility estimate. A logger records every prediction alongside
the exchange's own price at that instant. A scorer compares the two.

Nothing here places orders. That is deliberate — see **Current status**.

No dependencies. Python 3.10+ and the standard library.

## Current status

The model does not beat the market. As of 397 priced readings:

| measure | value |
| --- | --- |
| Brier score | 0.1718 |
| Kalshi mid Brier, same contracts | 0.1518 |
| **Skill vs the book** | **−13.2%** |
| Skill vs always-50% | +43.4% |

That +43.4% is not evidence of anything. It comes from correctly calling
strikes far from the money, which are nearly decided already. The number that
matters is skill against the price you would actually pay, and it is negative
in every distance band.

A calibration correction was fitted and **rejected**: it improved in-sample
and got worse out of sample (0.17420 → 0.17890 on a chronological holdout).
The model's miscalibration is not a fixed bias that can be corrected — it
moves between periods.

Sample so far is a few days in one market direction. Not a verdict, but three
independent measurements all point the same way.

## Run it

```
python kalshi_book.py            # is the Kalshi book readable?
python kalshi_book.py discover   # find the BTC series ticker
python logger.py                 # record predictions + book. leave running.
python score.py                  # did it work?
python score.py --no-drift       # re-price with the drift term removed
python score.py --by-horizon     # T-12 vs T-8 vs T-4
python learn.py                  # fit a calibration correction
python learn.py --agents         # which adjusters help, which hurt
python -m unittest discover -p "test_*.py"    # 158 tests
```

The logger needs no Kalshi account. Market data is public and read-only.

## Reading the score

- **Skill vs 50%** — easy to inflate, mostly measures far strikes. Sanity
  check only.
- **Skill vs market** — the real number. The executor should not exist until
  this is clearly positive near the money.
- **Calibration table** — `n` counts calls, `rdg` counts readings. Strikes
  from one reading move together, so error bars use `rdg`.
- **By distance from the money** — the 0.0–0.5 sd band is the only one
  tradeable at a sane fee.
- **conf bias** — negative means the model's confident calls fail more often
  than it claims. Positive means underconfident.

## Layout

```
core/
  feed.py          Candle model, Coinbase and Binance feeds
  kalshi.py        Contract pricing, fee model, prob_above
  kalshi_api.py    Read-only Kalshi market data (no auth, no orders)
  adjusters.py     The five volatility adjusters
  learning.py      Isotonic calibration + agent ablation
  indicators.py    SMA, ATR, RSI, swing points, level clustering
  base.py          Agent contract
  agents.py        Phase 1 candle agents (separate from the adjusters)
  orchestrator.py  Runs the Phase 1 agents
logger.py          Records predictions + book. The important one.
score.py           Scores the log against reality and against the book
learn.py           Fits corrections, refuses them when they don't validate
kalshi_book.py     Inspect the current Kalshi book
```

`predictions.jsonl` is gitignored. It is your data, not the project's.

## The adjusters

Each one modifies the volatility estimate. Multipliers compound; any single
suppressor stops the trade.

| Adjuster | What it does |
| --- | --- |
| `jump_detector` | Flags a large recent candle. Suppresses within 2 min of one. |
| `time_of_day` | Corrects for the hour-of-day volatility profile. |
| `vol_uncertainty` | Flags when the three volatility estimators disagree. |
| `momentum` | Measures short-horizon autocorrelation instead of assuming it. |
| `round_numbers` | Price behaviour near round strikes. |

Two carry fixes worth knowing about:

**`time_of_day`** shrinks its raw ratio toward 1.0 by sample size,
`n/(n+24)`. Before this, 38% of logged windows sat exactly on the old clamp
(0.7 or 1.4) — a bucket of ~12 samples can produce a 1.4 ratio from noise
alone, so the clamp was setting the number rather than guarding it.

**`vol_uncertainty`** no longer widens sigma. Ablation on 1,278 calls showed
removing its multiplier improved Brier by 0.9%: widening pushes probabilities
toward 50%, and this model is already underconfident near the money. Its
suppression flag is kept — that part earns its place. Set `WIDEN = 0.5` to
restore the old behaviour.

Both fixes together measured +2.98% Brier, **in sample**. `ablate_agents` has
no chronological holdout, so treat that as "two broken things are less broken"
rather than proof of improvement.

## The fee model

`core/kalshi.py` implements Kalshi's fee: `roundup(multiplier * contracts *
price * (1 - price))`. It peaks at 50c, where `price*(1-price)` hits its
maximum of 0.25. A 50c contract paying 1.75c is 3.5% of stake — one side.

That is the bar. Near the money, where the fee is worst, you need several
points of genuine edge before anything is left. The model currently has
negative edge there.

## What is not built

**Order execution.** No auth, no order placement, no position management.
The gate is evidence, not engineering: an autonomous system needs something
real to be autonomous about, and skill vs the book is −13%.

**A holdout for `ablate_agents`.** `learn_calibration` splits chronologically
and refuses corrections that don't transfer. The agent ablation does not, so
its verdicts are in-sample and vulnerable to the same failure that just
rejected the calibration fit.

## Before real money

Run the logger for weeks, across more than one market regime. Then run
`score.py` and read one line: **Skill vs market**.

Clearly positive near the money means the model knows something the price does
not, and an executor becomes worth building. Near zero or negative means the
book already knows what the model knows, and trading it pays a fee to be
wrong.

Beating a coin flip and beating a market are different bars. This project
exists to tell the two apart before money is involved.
