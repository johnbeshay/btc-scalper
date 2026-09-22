"""
Score the log. Does the model's 70% actually mean 70%? And does it beat the
price Kalshi was charging?

    python score.py                 # schema 3 only: live-spot records
    python score.py --schema all    # everything, with a warning
    python score.py --schema 2      # the old stale-spot era, for comparison
    python score.py --by-horizon
    python score.py --no-drift      # re-price every row with drift zeroed

Reads predictions.jsonl, joins predictions to outcomes, and reports:

  Brier score     mean squared error of the probabilities. Lower is better.
                  0.25 is what you get by always saying 50%. Above that means
                  the model is worse than useless.

  Skill vs 50%    Brier compared against always-50%. Positive means the model
                  knows something. This number is EASY to inflate: a strike
                  two sigmas away with four minutes left is nearly decided,
                  and getting it right is not information you can sell.

  Skill vs market Brier compared against the Kalshi mid price. This is the
                  number that decides whether an executor should exist. Now
                  reported with a confidence interval, because a point
                  estimate here was being read as far more settled than the
                  data supports.

  Calibration     predictions bucketed by confidence, compared against how
                  often those cases actually happened. n counts calls;
                  rdg counts independent readings.

WHAT COUNTS AS THE RIGHT ANSWER
-------------------------------
`hit` now comes from Kalshi's own settlement result wherever the log has one
(a "settlement" line for the row's ticker). Only rows without one fall back
to the old test, Coinbase candle close versus strike.

That fallback was the only source until 2026-09-14, and it was wrong on
20.6% of windows when checked against real settlements - Kalshi resolves on
CF Benchmarks' BRTI as a 60-second average, which is neither the same index
nor the same kind of comparison. The disagreements sat near the money.
Every number this tool printed before that date was graded against a
partially wrong answer key, and the report now says how many rows are still
on the fallback so that cannot happen silently again.

Run `python backfill.py` to fetch settlements for windows already logged.

TWO ERAS OF DATA
----------------
Records before schema 3 were priced from Coinbase's /candles feed, which runs
minutes behind the market - measured at 5.3 minutes stale and $88 away from
the live price, against a typical 15-minute sigma of about $60. Those windows
measure a model that could not see moves the exchange had already seen, so
every disagreement with the book is contaminated by information the model
simply did not have.

They are not garbage, but they are a different model. Pooling them with
schema 3 averages two things and tells you about neither, so the default is
schema 3 only and everything else is opt-in.

WHY THE ERROR BARS CLUSTER ON THE WINDOW
----------------------------------------
Nine strikes from one reading are one look at the market, not nine - they
rise and fall together with the same price move. The calibration table has
always used readings rather than calls for this reason.

The same argument applies one level up, and was being missed. Three readings
of the same 15-minute window, taken four minutes apart, are not independent
either: they share a window, a sigma, a set of agent multipliers and mostly
the same price path. The independent unit is the WINDOW.

So skill vs market is bootstrapped by resampling windows, not rows. How much
that widens the interval depends on how correlated the errors actually are.
Measured on simulated logs: when the model's error is independent per strike
the difference is small, around 1.1x. When the error is a shared per-window
sigma mis-estimate - which is how a volatility model actually fails, one bad
sigma skewing every strike the same way - the window-clustered interval came
out about 2.1x wider than a row-level one.

Two times is not the order of magnitude a naive count of calls would suggest,
and it is not nothing either. It is the difference between an interval that
excludes zero and one that does not, in exactly the range these numbers sit
in. A headline like "-13%" quoted without an interval was never as settled as
it read.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from core.kalshi import prob_above

LOG = Path(__file__).parent / "predictions.jsonl"
LIVE_SPOT_SCHEMA = 3

# Independent windows, not calls. Below MIN_WINDOWS nothing is printed at
# all; between the two, numbers print with a caution banner.
MIN_WINDOWS = 20
COMFORTABLE_WINDOWS = 200


def load(path: Path, zero_drift: bool = False):
    """
    Join predictions to outcomes. Returns (rows, n_readings, n_unresolved).

    Every row keeps the keys older callers expect (window_id, horizon,
    suppressed, p, sigmas, hit, spot, strike, close) and adds:

        reading     (window_id, horizon) - one look at the market
        schema      record version; 3+ means spot came from the live ticker
        spot_source "ticker" or "candle" on schema 3+, None before
        ladder      "kalshi" or "synthetic"
        p_yes       model probability the YES contract pays
        mkt_p       market's implied P(above) from the mid, or None
        yes_bid / yes_ask / no_ask   book in dollars, or None
        sigma, drift_pct             so p can be re-derived

    zero_drift re-prices p from spot, strike and sigma with no drift term,
    which is what core/kalshi.py says the model should do and logger.py
    historically did not. Fully offline; the log already has every input.
    """
    preds, outs, settled = [], {}, {}
    if not path.exists():
        return [], 0, 0

    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("type")
            if t == "prediction":
                preds.append(rec)
            elif t == "outcome":
                outs[rec["window_id"]] = rec
            elif t == "settlement":
                res = (rec.get("result") or "").lower()
                if rec.get("ticker") and res in ("yes", "no"):
                    settled[rec["ticker"]] = res

    # A reading is only excluded if NONE of its strikes can be graded.
    #
    # The Coinbase close used to be a precondition for the whole reading: no
    # trustworthy close, no row - even when Kalshi's own settlement for that
    # exact ticker was sitting in the log. That made the ground truth depend
    # on the proxy it replaced. Now each strike is graded independently:
    # Kalshi settlement if there is one, the Coinbase close only as a
    # fallback, and the row is skipped only when neither exists. `close` is
    # None on rows graded from settlement with no usable Coinbase close;
    # callers that need it (basis.py, diagnose.py) must check for that.
    rows, unresolved = [], 0
    for p in preds:
        out = outs.get(p["window_id"])
        close = (out["close_price"]
                 if out and out.get("trustworthy", True) else None)
        spot = p["spot"]
        sigma = p.get("sigma")
        graded_any = False

        for item in p["predictions"]:
            strike = item["strike"]
            prob = item["p_above"]
            if zero_drift and sigma and spot > 0 and strike > 0:
                prob = prob_above(spot, strike, sigma)
            mkt = item.get("market") or {}
            ticker = mkt.get("ticker")
            direction = mkt.get("yes_direction", "above")

            # Ground truth if we have it; the Coinbase proxy if we do not.
            if ticker and ticker in settled:
                yes_paid = settled[ticker] == "yes"
                hit = int(yes_paid if direction == "above" else not yes_paid)
                hit_source = "kalshi"
            elif close is not None:
                hit = 1 if close > strike else 0
                hit_source = "close"
            else:
                continue          # neither settlement nor a usable close
            graded_any = True

            rows.append(
                {
                    "hit_source": hit_source,
                    "window_id": p["window_id"],
                    "horizon": p["horizon_min"],
                    "reading": (p["window_id"], p["horizon_min"]),
                    "schema": p.get("schema", 1),
                    "spot_source": p.get("spot_source"),
                    "candle_age_min": p.get("candle_age_min"),
                    "ladder": p.get("ladder", "synthetic"),
                    "suppressed": p.get("suppressed", False),
                    "p": prob,
                    "p_yes": item.get("p_yes", prob),
                    "sigmas": item["sigmas_out"],
                    "hit": hit,
                    "spot": spot,
                    "strike": strike,
                    "close": close,
                    "sigma": sigma,
                    "drift_pct": p.get("drift_pct", 0.0),
                    "agents": p.get("agents") or {},
                    "mkt_p": mkt.get("implied_p_above"),
                    "yes_direction": mkt.get("yes_direction", "above"),
                    "yes_bid": mkt.get("yes_bid"),
                    "yes_ask": mkt.get("yes_ask"),
                    "no_ask": mkt.get("no_ask"),
                    "no_bid": mkt.get("no_bid"),
                    "ticker": mkt.get("ticker"),
                }
            )
        if not graded_any:
            unresolved += 1
    return rows, len(preds), unresolved


def brier(rows, key: str = "p") -> float:
    return sum((r[key] - r["hit"]) ** 2 for r in rows) / len(rows)


def skill_pct(model: float, base: float) -> float:
    return (base - model) / base * 100 if base else 0.0


def n_readings(rows) -> int:
    return len({r["reading"] for r in rows})


def n_windows(rows) -> int:
    return len({r["window_id"] for r in rows})


def bootstrap_skill(rows, key: str = "mkt_p", iters: int = 2000,
                    seed: int = 0) -> tuple[float, float, float] | None:
    """
    Confidence interval for Brier skill, resampling WINDOWS with replacement.

    Returns (point_estimate, lo_2.5pct, hi_97.5pct), or None if there are too
    few windows to say anything.

    Resampling windows rather than rows is the whole point. Rows inside a
    window share a price path; drawing rows independently would manufacture
    a sample far larger than the evidence, and produce a tight interval
    around a number that is not actually pinned down.
    """
    by_window = defaultdict(list)
    for r in rows:
        by_window[r["window_id"]].append(r)
    windows = list(by_window)
    if len(windows) < 20:
        return None

    point = skill_pct(brier(rows), brier(rows, key))

    rng = random.Random(seed)
    draws = []
    for _ in range(iters):
        sample = []
        for _ in windows:
            sample.extend(by_window[rng.choice(windows)])
        try:
            draws.append(skill_pct(brier(sample), brier(sample, key)))
        except ZeroDivisionError:
            continue

    if not draws:
        return None
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]
    return point, lo, hi


def calibration(rows, buckets=10):
    """
    Group by predicted probability, compare against observed frequency.

    The standard error uses the number of distinct readings in the bucket,
    not the number of calls. Strikes from one reading rise and fall together
    with the same price move, so counting each as an independent trial makes
    every gap look more significant than it is.
    """
    grouped = defaultdict(list)
    for r in rows:
        idx = min(int(r["p"] * buckets), buckets - 1)
        grouped[idx].append(r)

    out = []
    for i in sorted(grouped):
        g = grouped[i]
        predicted = sum(x["p"] for x in g) / len(g)
        actual = sum(x["hit"] for x in g) / len(g)
        rdg = n_readings(g)
        se = math.sqrt(max(actual * (1 - actual), 1e-9) / max(rdg, 1))
        out.append(
            {
                "lo": i / buckets,
                "hi": (i + 1) / buckets,
                "n": len(g),
                "readings": rdg,
                "predicted": predicted,
                "actual": actual,
                "error": actual - predicted,
                "se": se,
                "significant": abs(actual - predicted) > 2 * se,
            }
        )
    return out


def bar(pred, act, width=22):
    """A small visual of predicted versus actual."""
    p = int(round(pred * width))
    a = int(round(act * width))
    cells = []
    for i in range(width):
        if i < min(p, a):
            cells.append("#")
        elif i < max(p, a):
            cells.append("-" if a > p else ".")
        else:
            cells.append(" ")
    return "".join(cells)


def distance_band(r) -> str:
    s = abs(r["sigmas"])
    if s < 0.5:
        return "0.0 - 0.5 sd"
    if s < 1.0:
        return "0.5 - 1.0 sd"
    if s < 2.0:
        return "1.0 - 2.0 sd"
    return "beyond 2 sd"


def schema_summary(rows) -> dict:
    counts = defaultdict(set)
    for r in rows:
        counts[r["schema"]].add(r["window_id"])
    return {k: len(v) for k, v in sorted(counts.items())}


def report(rows, total_preds, unresolved):
    n = len(rows)
    rdg = n_readings(rows)
    win = n_windows(rows)

    print()
    print("=" * 64)
    print(f"  {n:,} resolved calls from {rdg:,} readings across {win:,} windows")
    print(f"  the independent unit is the WINDOW: {win:,}")
    if unresolved:
        print(f"  {unresolved:,} readings excluded "
              "(no Kalshi settlement and no trustworthy close)")

    kalshi_rows = sum(1 for r in rows if r.get("hit_source") == "kalshi")
    close_rows = n - kalshi_rows
    if close_rows == 0:
        print(f"  outcomes: all {n:,} from Kalshi settlement")
    elif kalshi_rows == 0:
        print(f"  outcomes: NONE from Kalshi settlement - all {n:,} rows are on")
        print("  the Coinbase-close proxy, which was wrong 20.6% of the time.")
        print("  Run:  python backfill.py")
    else:
        print(f"  outcomes: {kalshi_rows:,} from Kalshi settlement, "
              f"{close_rows:,} still on the Coinbase proxy")
        print("  Run backfill.py to close that gap before trusting the totals.")
    print("=" * 64)

    # The floor is deliberately low now. Refusing to print was the right
    # call when the headline number came with no interval attached - a bare
    # "-13%" off thirty windows invites exactly the wrong conclusion. With a
    # bootstrap interval on it, a small sample speaks for itself: the
    # interval comes out enormous and says so. Showing that beats silence,
    # which teaches nothing about how much data would be enough.
    if win < MIN_WINDOWS:
        print()
        print(f"  Only {win} independent windows. Nothing here is meaningful yet.")
        print("  Calls and readings are not the sample size - strikes and")
        print("  readings inside a window move together.")
        print("  Keep the logger running.")
        print()
        return

    if win < COMFORTABLE_WINDOWS:
        print()
        print(f"  CAUTION: {win} independent windows. Every number below is")
        print(f"  provisional - {COMFORTABLE_WINDOWS}+ is where they start to")
        print("  settle. Read the interval on skill vs market, not the point")
        print("  estimate.")

    b = brier(rows)
    base = brier([{**r, "p": 0.5} for r in rows])
    skill = skill_pct(b, base)

    print()
    print(f"  Brier score          {b:.4f}")
    print(f"  Always-50% score     {base:.4f}")
    print(f"  Skill vs 50%         {skill:+.1f}%  ", end="")
    if skill > 10:
        print("beats guessing - but see the near-money line below")
    elif skill > 2:
        print("slight edge over guessing")
    elif skill > -2:
        print("no better than guessing")
    else:
        print("WORSE than guessing - something is wrong")

    # ---- the number that matters ----------------------------------------
    priced = [r for r in rows if r["mkt_p"] is not None]
    print()
    if priced:
        mb = brier(priced)
        mk = brier(priced, key="mkt_p")
        pw = n_windows(priced)
        print(f"  Rows with a Kalshi price   {len(priced):,} "
              f"({n_readings(priced):,} readings, {pw:,} windows)")
        print(f"  Model Brier on those       {mb:.4f}")
        print(f"  Market-mid Brier           {mk:.4f}")

        ci = bootstrap_skill(priced)
        s = skill_pct(mb, mk)
        if ci:
            _, lo, hi = ci
            print(f"  Skill vs market            {s:+.1f}%  "
                  f"[95% CI {lo:+.1f}% to {hi:+.1f}%]")
            print()
            if lo > 0:
                print("  The whole interval is above zero: the model beats the")
                print("  book on this data. Check it holds near the money.")
            elif hi < 0:
                print("  The whole interval is below zero: the market prices")
                print("  this better than the model does.")
            else:
                print("  The interval spans zero. On this much data the model")
                print("  is NOT distinguishable from the book, in either")
                print("  direction. A point estimate here means little; more")
                print("  windows is the only thing that narrows it.")
        else:
            print(f"  Skill vs market            {s:+.1f}%  "
                  f"(too few windows for an interval)")
    else:
        print("  Skill vs market      (no Kalshi prices in the log)")
        print("  Run the logger with Kalshi reachable to measure edge.")
        print("  Check with: python kalshi_book.py")

    # ---- calibration ----------------------------------------------------
    print()
    print("  Calibration")
    print("  " + "-" * 62)
    print(f"  {'says':>8} {'actually':>8} {'n':>6} {'rdg':>5} {'off by':>7}  chart")
    for c in calibration(rows):
        flag = " *" if c["significant"] else "  "
        print(
            f"  {c['predicted'] * 100:>7.1f}% {c['actual'] * 100:>7.1f}% "
            f"{c['n']:>6,} {c['readings']:>5,} {c['error'] * 100:>+6.1f}%{flag} "
            f"{bar(c['predicted'], c['actual'])}"
        )
    print("  " + "-" * 62)
    print("  * marks gaps bigger than noise explains, using rdg not n.")
    print("  '#' is agreement, '-' happened more than predicted, '.' less.")

    # ---- near the money, always -----------------------------------------
    print()
    by_group(rows, distance_band, "By distance from the money")

    near = [r for r in rows if abs(r["sigmas"]) < 0.5]
    if len(near) >= 20:
        print("  The 0.0 - 0.5 sd band is the one you can trade. Skill there is")
        print("  the honest version of the headline number above.")
        print()


def confidence_bias(rows) -> float:
    """
    How often the model's favoured side actually won, versus how often it
    claimed it would.

    Measuring raw (hit - p) looks obvious and is wrong for a strike ladder.
    A strike far ABOVE spot and one far BELOW have errors of opposite sign,
    so averaging them cancels exactly the effect you are trying to see. This
    folds every call onto its confident side first, which makes the sign
    meaningful: negative means confident calls failed more often than claimed,
    which is the signature of fat tails.
    """
    if not rows:
        return 0.0
    total = 0.0
    for r in rows:
        favoured_p = r["p"] if r["p"] >= 0.5 else 1 - r["p"]
        favoured_hit = r["hit"] if r["p"] >= 0.5 else 1 - r["hit"]
        total += favoured_hit - favoured_p
    return total / len(rows) * 100


def by_group(rows, key, label, fmt=str):
    grouped = defaultdict(list)
    for r in rows:
        grouped[key(r)].append(r)

    any_market = any(r["mkt_p"] is not None for r in rows)

    print(f"  {label}")
    print("  " + "-" * 62)
    head = (f"  {'group':>14} {'n':>6} {'win':>5} {'brier':>6} "
            f"{'vs 50%':>7} {'conf bias':>9}")
    if any_market:
        head += f" {'vs mkt':>7}"
    print(head)

    for g in sorted(grouped):
        rs = grouped[g]
        if len(rs) < 20:
            continue
        b = brier(rs)
        base = brier([{**r, "p": 0.5} for r in rs])
        line = (
            f"  {fmt(g):>14} {len(rs):>6,} {n_windows(rs):>5,} {b:>6.4f} "
            f"{skill_pct(b, base):>+6.1f}% {confidence_bias(rs):>+8.1f}%"
        )
        if any_market:
            priced = [r for r in rs if r["mkt_p"] is not None]
            if len(priced) >= 20:
                line += f" {skill_pct(brier(priced), brier(priced, 'mkt_p')):>+6.1f}%"
            else:
                line += f" {'-':>7}"
        print(line)

    print("  " + "-" * 62)
    print("  win: independent windows behind the row. A group with few windows")
    print("  says little however large n looks.")
    print("  conf bias: how much more (or less) often the model's favoured side")
    print("  won than it claimed. Negative means overconfident.")
    if any_market:
        print("  vs mkt: Brier skill against the Kalshi mid. The executor should")
        print("  not exist until this is clearly positive near the money.")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description="Score the prediction log")
    p.add_argument("--log", default=str(LOG))
    p.add_argument("--by-horizon", action="store_true")
    p.add_argument("--by-distance", action="store_true",
                   help="(kept for compatibility; distance bands now always shown)")
    p.add_argument("--no-drift", action="store_true",
                   help="re-price every row with the drift term removed")
    p.add_argument("--include-suppressed", action="store_true",
                   help="include windows the agents flagged as untradeable")
    p.add_argument("--schema", default="latest",
                   help=("record version to score: a number, 'all', or "
                         "'latest' (default) for the newest version in the "
                         "log. Each schema is a different model; they are "
                         "never pooled silently."))
    args = p.parse_args()

    rows, total, unresolved = load(Path(args.log), zero_drift=args.no_drift)
    if not rows:
        print()
        print(f"  Nothing scored yet. Is {args.log} there and has a window closed?")
        print("  Start with: python logger.py")
        print()
        return 0

    # ---- schema selection ------------------------------------------------
    present = schema_summary(rows)
    if args.schema.lower() == "latest":
        args.schema = str(max(present))
        print(f"\n  scoring the newest schema in the log: {args.schema}")
    if args.schema.lower() == "all":
        if len(present) > 1:
            print()
            print("  WARNING: pooling schema versions.")
            for v, w in present.items():
                era = "live spot" if v >= LIVE_SPOT_SCHEMA else "STALE spot"
                print(f"    schema {v}: {w:,} windows  ({era})")
            print("  Records before schema 3 were priced from a candle feed")
            print("  running minutes behind the market. They measure a model")
            print("  that could not see moves the exchange already had.")
            print("  These are two different models; the average describes")
            print("  neither.")
    else:
        try:
            want = int(args.schema)
        except ValueError:
            print(f"\n  --schema must be a number or 'all', got {args.schema!r}\n")
            return 1
        kept = [r for r in rows if r["schema"] == want]
        if not kept:
            print()
            print(f"  No schema {want} records in the log yet.")
            if present:
                print("  Present:")
                for v, w in present.items():
                    print(f"    schema {v}: {w:,} windows")
            print(f"  Use --schema all, or --schema {max(present)} "
                  f"to score what is there.")
            print()
            return 0
        if len(kept) < len(rows):
            other = {v: w for v, w in present.items() if v != want}
            print()
            print(f"  scoring schema {want} only "
                  f"({n_windows(kept):,} windows)")
            print(f"  excluded: " + ", ".join(
                f"schema {v} ({w:,} windows)" for v, w in other.items()))
            print("  use --schema all to pool them, but read the warning first")
        rows = kept

    if args.no_drift:
        print("\n  (drift term zeroed; p re-derived from spot, strike, sigma)")

    if not args.include_suppressed:
        kept = [r for r in rows if not r["suppressed"]]
        if len(kept) < len(rows):
            print(f"\n  ({len(rows) - len(kept):,} suppressed calls excluded; "
                  "use --include-suppressed to see them)")
        rows = kept or rows

    report(rows, total, unresolved)

    if args.by_horizon:
        by_group(rows, lambda r: r["horizon"], "By how far ahead",
                 lambda g: f"T-{g} min")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
