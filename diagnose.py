"""
Two checks on the existing log, before spending days collecting more.

    python diagnose.py
    python diagnose.py --schema 2

CHECK 1 - IS THE DISAGREEMENT JUST THE MISSED MOVE?
---------------------------------------------------
Schema 2 priced from a candle feed running about five minutes behind. If that
lag explains the model losing to the book, then the windows where the model
disagreed most with the book should be the windows where price moved most
during the blind gap - the book had seen the move, the model had not.

This splits the rows into quartiles of disagreement and shows how far price
actually travelled in each. A rising move across the quartiles supports the
stale-spot story. A flat one means the disagreement is coming from somewhere
else, and fixing the spot will not fix the edge.

CHECK 2 - IS THE OUTCOME COLUMN EVEN RIGHT?
-------------------------------------------
This one matters more, and it is independent of the stale spot.

score.py decides `hit` by comparing a Coinbase 1-minute candle close against
the strike. Kalshi does not settle that way. The rules on these markets say
they settle on CF Benchmarks' BRTI: the 60-second average before close,
compared against the 60-second average before the window opened.

Different index, and an average-of-averages rather than a point in time. If
those two disagree on a meaningful fraction of windows, then `hit` is wrong
on those windows, and every number score.py produces is being measured
against the wrong truth - including the -20% skill figure, and including
every schema 3 window still to come.

So this fetches how Kalshi actually settled each market and compares.
A disagreement rate of a percent or two is rounding. Ten percent would mean
the scorer needs rebuilding before any more data is worth collecting.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import score
from core.kalshi_api import DEFAULT_SERIES, KalshiError, KalshiMarketData

LOG = Path(__file__).parent / "predictions.jsonl"


# --------------------------------------------------------------------------
# Check 1
# --------------------------------------------------------------------------

def check_disagreement_vs_move(rows) -> None:
    priced = [r for r in rows
              if r["mkt_p"] is not None and r.get("sigma") and r["sigma"] > 0]
    if len(priced) < 40:
        print("  too few priced rows for this check")
        return

    for r in priced:
        r["_disagree"] = abs(r["p"] - r["mkt_p"])
        # how far price travelled from the (possibly stale) spot, in sigmas
        r["_move_sd"] = abs(math.log(r["close"] / r["spot"])) / r["sigma"]
        r["_model_err"] = (r["p"] - r["hit"]) ** 2
        r["_mkt_err"] = (r["mkt_p"] - r["hit"]) ** 2

    ordered = sorted(priced, key=lambda r: r["_disagree"])
    q = len(ordered) // 4
    quartiles = [ordered[:q], ordered[q:2 * q], ordered[2 * q:3 * q], ordered[3 * q:]]

    print()
    print("  CHECK 1  does the disagreement track the move the model missed?")
    print("  " + "-" * 64)
    print(f"  {'quartile':>10} {'n':>5} {'win':>5} {'disagree':>9} "
          f"{'move (sd)':>10} {'model':>7} {'market':>7}")

    for i, g in enumerate(quartiles, 1):
        if not g:
            continue
        print(f"  {'Q' + str(i):>10} {len(g):>5} {score.n_windows(g):>5} "
              f"{sum(r['_disagree'] for r in g) / len(g):>9.3f} "
              f"{sum(r['_move_sd'] for r in g) / len(g):>10.2f} "
              f"{sum(r['_model_err'] for r in g) / len(g):>7.4f} "
              f"{sum(r['_mkt_err'] for r in g) / len(g):>7.4f}")

    print("  " + "-" * 64)
    lo = sum(r["_move_sd"] for r in quartiles[0]) / max(len(quartiles[0]), 1)
    hi = sum(r["_move_sd"] for r in quartiles[-1]) / max(len(quartiles[-1]), 1)
    ratio = hi / lo if lo else 0

    print(f"  move in Q4 vs Q1: {ratio:.2f}x")
    if ratio > 1.5:
        print("  Price moved much further in the high-disagreement windows.")
        print("  Consistent with the model reacting to a spot the book had")
        print("  already moved past - the stale-feed story.")
    elif ratio > 1.15:
        print("  Some relationship, but weaker than a pure stale-spot story")
        print("  would predict. Expect the live-spot fix to help, not cure.")
    else:
        print("  Disagreement does NOT track the missed move. The stale spot")
        print("  is not what is driving the gap, and schema 3 is unlikely to")
        print("  look much different. Look at the model, not the feed.")


# --------------------------------------------------------------------------
# Check 2
# --------------------------------------------------------------------------

def check_settlement_truth(rows, series: str) -> None:
    """
    Compare the scorer's `hit` against how Kalshi actually settled.

    `hit` comes from a Coinbase candle close; Kalshi settles on a BRTI
    average. Anywhere those disagree, the scorer has been grading against
    the wrong answer.
    """
    tickers = {r["ticker"] for r in rows if r.get("ticker")}
    if not tickers:
        print()
        print("  CHECK 2  no tickers in the log - nothing to compare")
        print("  (this needs rows where the Kalshi book was captured)")
        return

    print()
    print(f"  CHECK 2  does Coinbase agree with how Kalshi settled?")
    print("  " + "-" * 64)
    print(f"  looking up {len(tickers):,} settled markets...")

    md = KalshiMarketData(series=series)
    results: dict[str, str] = {}
    try:
        for status in ("settled", "finalized", "closed"):
            try:
                for m in md.markets(status=status):
                    t = str(m.get("ticker", ""))
                    res = (m.get("result") or "").lower()
                    if t in tickers and res in ("yes", "no"):
                        results[t] = res
            except KalshiError:
                continue
    except Exception as exc:
        print(f"  could not reach Kalshi: {exc}")
        return

    if not results:
        print("  Kalshi returned no settled results for these tickers.")
        print("  They may have aged out of the API window. This check needs")
        print("  to run within a few days of the windows being logged.")
        return

    agree = disagree = 0
    examples = []
    for r in rows:
        t = r.get("ticker")
        if not t or t not in results:
            continue
        # our view: did YES pay, per the Coinbase close?
        our_yes = r["hit"] if r.get("yes_direction", "above") == "above" else 1 - r["hit"]
        their_yes = 1 if results[t] == "yes" else 0
        if our_yes == their_yes:
            agree += 1
        else:
            disagree += 1
            if len(examples) < 8:
                examples.append((t, r["spot"], r["strike"], r["close"],
                                 our_yes, their_yes))

    total = agree + disagree
    if not total:
        print("  no overlap between the log and the settled markets returned")
        return

    pct = disagree / total * 100
    print(f"  compared {total:,} rows: {agree:,} agree, {disagree:,} disagree "
          f"({pct:.1f}%)")

    if examples:
        print()
        print(f"  {'ticker':>28} {'strike':>10} {'cb close':>10} "
              f"{'ours':>5} {'kalshi':>7}")
        for t, spot, strike, close, ours, theirs in examples:
            print(f"  {t[:28]:>28} {strike:>10,.0f} {close:>10,.2f} "
                  f"{'yes' if ours else 'no':>5} "
                  f"{'yes' if theirs else 'no':>7}")

    print("  " + "-" * 64)
    if pct < 2:
        print("  The Coinbase close is a good enough proxy for settlement.")
        print("  The scorer is grading against the right answer.")
    elif pct < 8:
        print("  Some disagreement. These are near-the-money windows where")
        print("  the BRTI average and a Coinbase close land on opposite sides")
        print("  of the strike. Worth knowing, not fatal - but note it falls")
        print("  entirely in the band you would actually trade.")
    else:
        print("  SIGNIFICANT disagreement. `hit` is wrong this often, which")
        print("  means score.py has been grading against the wrong outcome.")
        print("  Fix the outcome source before collecting more data -")
        print("  schema 3 inherits this exactly as schema 2 has it.")


def main() -> int:
    p = argparse.ArgumentParser(description="Sanity-check the log")
    p.add_argument("--log", default=str(LOG))
    p.add_argument("--schema", default="2")
    p.add_argument("--series", default=DEFAULT_SERIES)
    p.add_argument("--skip-kalshi", action="store_true",
                   help="run check 1 only, no network")
    args = p.parse_args()

    rows, _, _ = score.load(Path(args.log))
    if not rows:
        print(f"\n  nothing in {args.log}\n")
        return 0

    if args.schema.lower() != "all":
        want = int(args.schema)
        rows = [r for r in rows if r["schema"] == want]
        if not rows:
            print(f"\n  no schema {want} rows\n")
            return 0

    print()
    print("=" * 68)
    print(f"  {len(rows):,} rows across {score.n_windows(rows):,} windows "
          f"(schema {args.schema})")
    print("=" * 68)

    check_disagreement_vs_move(rows)

    if not args.skip_kalshi:
        check_settlement_truth(rows, args.series)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
