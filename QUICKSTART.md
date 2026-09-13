# Quickstart

Python 3.10+. No pip install, no build step.

## The two things to run

**Dashboard** — helps you decide on a trade right now.

```
python kalshi_server.py
```
Then open http://localhost:8000

**Logger** — builds the evidence that tells you whether any of this works.

```
python kalshi_book.py      # once, to confirm the Kalshi book is readable
python logger.py
```
Leave it running. Check progress any time with `python score.py`.

If `kalshi_book.py` prints strikes and prices, the logger records the book
alongside every prediction and `score.py` can measure edge against the
market. If it prints nothing, see "The series ticker" below.

Run both together in two terminal windows. They are independent.

## Set your fee first

```
python kalshi_server.py --multiplier 0.07
```

0.07 is the standard rate, but Kalshi's crypto markets may use a higher
multiplier. Check the order ticket on a real trade and use that number.
Every threshold on the dashboard depends on it.

## Everything you can run

| Command | Does |
|---|---|
| `python kalshi_server.py` | dashboard at localhost:8000 |
| `python logger.py` | record predictions and outcomes |
| `python kalshi_book.py` | show the Kalshi book for the next window |
| `python kalshi_book.py discover` | list series tickers that look like BTC |
| `python score.py` | calibrated? and does it beat the market? |
| `python score.py --no-drift` | same, with the drift term removed |
| `python score.py --by-horizon` | is it better at 12 min or 4 min? |
| `python replay.py` | what would trading it have made, in dollars? |
| `python replay.py --fill mid` | same, optimistic fills |
| `python logger.py --no-market` | log without Kalshi (calibration only) |
| `python kalshi_price.py --table` | fair value by strike, terminal only |
| `python learn.py` | fit a correction from the log |
| `python learn.py --agents` | which agents actually help |
| `python learn.py --apply` | save the correction, if it validates |
| `python -m unittest discover -p "test_*.py"` | 174 tests |

Older spot-trading build, different instrument, kept for reference:
`server.py`, `run.py`, `simulate.py`.

## Reading the score

`score.py` answers two questions. When the model says 70%, does it happen 70%
of the time? And does 70% beat what Kalshi was charging?

- **Brier score** — mean squared error. Always-50% scores 0.25. Lower is better.
- **Skill vs 50%** — improvement over always-50%. Easy to inflate: a strike
  two sigmas away with four minutes left is nearly decided, and getting it
  right is not information you can sell. Treat this as a sanity check.
- **Skill vs market** — improvement over the Kalshi mid price. This is the
  number that decides whether an executor should exist. Needs the book in
  the log (see `kalshi_book.py`).
- **Calibration table** — `n` counts calls, `rdg` counts readings. Nine
  strikes from one reading move together, so error bars use `rdg`. `*`
  marks gaps bigger than noise explains.
- **By distance from the money** — always shown. The 0-0.5 sd band is the
  only one tradeable at a sane fee; its skill is the honest headline.
- **Confidence bias** — negative means the model's confident calls fail more
  often than it claims.

`--no-drift` re-prices every row with the drift term removed. `core/kalshi.py`
argues drift should be omitted over 15 minutes; the logger applies one anyway.
If the mid-range calibration improves with drift zeroed, that is the cause.

Under about 500 resolved calls the numbers are noise. At three readings per
window and four windows an hour, running the logger overnight gets you a few
hundred; a week gets you thousands.

## What the answer means

**Skill vs 50% positive but skill vs market near zero or negative** — the
model knows roughly what the market knows. There is nothing to trade.

**Skill vs market clearly positive near the money** — the model knows
something the price does not. Now check it survives the fee.

**Skill near zero** — it does not predict. No dashboard or automation fixes
that. Better to know after two weeks of logging than after two weeks of
losses.

**Far band shows strong negative confidence bias** — the fat-tail warning is
earning its place and should stay strict.

## The series ticker

Kalshi groups markets under a series ticker. The logger defaults to
`KXBTC15M` for the 15-minute BTC series. If `python kalshi_book.py` finds no
contracts, the name is probably different:

```
python kalshi_book.py discover
set KALSHI_BTC_SERIES=<ticker>       (Windows)
export KALSHI_BTC_SERIES=<ticker>    (Mac/Linux)
```

Or pass it directly: `python logger.py --series <ticker>`.

When Kalshi is unreachable the logger says so once, falls back to a synthetic
ladder around spot, and keeps going. Calibration is still measured; edge is not.

## Troubleshooting

**"Python was not found"** — install from python.org, tick "Add python.exe to
PATH", reopen the terminal.

**Browser cannot connect** — the terminal window running the server has to
stay open.

**"Price feed is not responding"** — Coinbase is unreachable. Check your
connection.

**Port 8000 in use** — `python kalshi_server.py --port 8080`

**score.py says nothing to score** — no window has closed yet. Wait 15 minutes.

## Before automating anything

The model has no track record. Automating it would concentrate money in its
blind spot, because the largest apparent edges appear exactly where it is
least reliable. Log first, read the calibration, then decide.

## The learning layer

`learn.py` fits a correction from the log using isotonic regression: it learns
the mapping between what the model says and what actually happens, so a "70%"
that is really 62% gets corrected.

It refuses to output anything unless the correction beats the uncorrected
model on data it was never fitted to, and it needs at least 400 resolved
calls. The train/test split is chronological, never random - windows minutes
apart share almost the same market state, so a random holdout would contain
near-copies of the training rows and everything would look brilliant.

Three outcomes, all informative:

- **Accepted with a real mapping** - the model was miscalibrated in a
  consistent way and that is now fixable.
- **Refused, already calibrated** - the model is fine as it is.
- **Accepted but the mapping is nearly flat near 50%** - this is the bad one.
  It means the model's output carries no information and the best correction
  is to ignore it entirely.

`learn.py --agents` runs ablation: it divides each agent's volatility
multiplier back out and re-scores. An agent marked HURTS was making
predictions worse and should be switched off.
