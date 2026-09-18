"""
Why is the model wrong exactly when it disagrees with the book?

    python disagree.py
    python disagree.py --schema 2     # compare against the old era

THE PUZZLE
----------
Calibration is close to perfect: the model's 45% happens 45.6% of the time.
So its probabilities are right on average. And yet the more it disagrees with
the Kalshi mid, the worse it does - claimed edge +19 points, realised -5.

Those two facts together are specific. A model that was simply noisy would be
badly calibrated. A model that was well calibrated and uninformative would
break even against the book. Being well calibrated overall AND wrong on the
disagreements means the errors are concentrated somewhere, and somewhere is
a thing you can look for.

This splits the log by how far the model was from the book and asks what is
different about the far group: what time it was, how volatile, which adjuster
was firing, which horizon, which side it took, how wide the spread was.

HOW TO READ IT
--------------
A column that shifts cleanly between the low-disagreement and
high-disagreement groups is a lead. The model is doing something specific in
those windows, and that something can be switched off or filtered out.

Columns that look the same in both groups rule things out, which is just as
useful and cheaper than guessing.

If nothing separates them, the honest conclusion is that the disagreements
are not a distinguishable subset - the model simply has no information the
book lacks, and no filter will rescue it.

Standard library only. Reads the existing log; changes nothing.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import score

LOG = Path(__file__).parent / "predictions.jsonl"


def enrich(rows):
    """Attach disagreement, realised outcome and the features to test."""
    out = []
    for r in rows:
        if r.get("mkt_p") is None:
            continue
        gap = r["p"] - r["mkt_p"]

        # Which side would the model have taken, and did that side win?
        took_yes = gap > 0
        won = (r["hit"] == 1) if took_yes else (r["hit"] == 0)

        # Brier against the outcome, model and market, so we can say who was
        # closer on exactly these rows.
        r = dict(r)
        r["_gap"] = abs(gap)
        r["_signed_gap"] = gap
        r["_side"] = "yes" if took_yes else "no"
        r["_won"] = won
        r["_model_err"] = (r["p"] - r["hit"]) ** 2
        r["_mkt_err"] = (r["mkt_p"] - r["hit"]) ** 2

        # spread, where both sides are known
        yb, ya = r.get("yes_bid"), r.get("yes_ask")
        r["_spread"] = (ya - yb) if (yb is not None and ya is not None) else None

        # hour of day, UTC, from the window id
        try:
            r["_hour"] = datetime.strptime(r["window_id"], "%Y%m%dT%H%M").hour
        except (ValueError, TypeError):
            r["_hour"] = None

        out.append(r)
    return out


def split_by_gap(rows, cut: float = 0.10):
    lo = [r for r in rows if r["_gap"] < cut]
    hi = [r for r in rows if r["_gap"] >= cut]
    return lo, hi


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(v, pct=False, places=3):
    if v is None:
        return "    -"
    return f"{v * 100:>6.1f}%" if pct else f"{v:>6.{places}f}"


def compare(lo, hi, label, fn, pct=False, places=3):
    a, b = mean([fn(r) for r in lo]), mean([fn(r) for r in hi])
    delta = ""
    if a is not None and b is not None and a != 0:
        ratio = b / a if a else 0
        if ratio > 1.3 or ratio < 0.77:
            delta = f"   <-- {ratio:.2f}x"
    print(f"  {label:>24} {fmt(a, pct, places)} {fmt(b, pct, places)}{delta}")


def agents_firing(rows) -> dict:
    """Mean |multiplier - 1| per agent: how hard each one is pushing."""
    totals = defaultdict(list)
    for r in rows:
        for name, info in (r.get("agents") or {}).items():
            m = info.get("vol_multiplier")
            if m is not None:
                totals[name].append(abs(m - 1.0))
    return {k: sum(v) / len(v) for k, v in totals.items() if v}


def by_bucket(rows, key, label, buckets):
    """Win rate and model-vs-market Brier within each bucket."""
    grouped = defaultdict(list)
    for r in rows:
        b = key(r)
        if b is not None:
            grouped[b].append(r)

    print()
    print(f"  {label}")
    print("  " + "-" * 62)
    print(f"  {'bucket':>14} {'n':>5} {'win%':>6} {'model':>8} {'market':>8} "
          f"{'model-mkt':>10}")
    for b in buckets:
        g = grouped.get(b)
        if not g or len(g) < 15:
            continue
        m = mean([r["_model_err"] for r in g])
        k = mean([r["_mkt_err"] for r in g])
        flag = "  <-- worse" if m - k > 0.01 else ""
        print(f"  {str(b):>14} {len(g):>5} "
              f"{mean([r['_won'] for r in g]) * 100:>5.0f}% "
              f"{m:>8.4f} {k:>8.4f} {m - k:>+10.4f}{flag}")
    print("  " + "-" * 62)


def main() -> int:
    p = argparse.ArgumentParser(description="Anatomy of the disagreements")
    p.add_argument("--log", default=str(LOG))
    p.add_argument("--schema", default="latest")
    p.add_argument("--cut", type=float, default=0.10,
                   help="disagreement threshold splitting low from high")
    p.add_argument("--horizon", type=int, default=None)
    args = p.parse_args()

    rows, _, _ = score.load(Path(args.log))
    present = score.schema_summary(rows)
    label = str(max(present)) if args.schema == "latest" else args.schema
    if label != "all":
        rows = [r for r in rows if r["schema"] == int(label)]
    rows = [r for r in rows if not r["suppressed"]]
    if args.horizon is not None:
        rows = [r for r in rows if r["horizon"] == args.horizon]

    rows = enrich(rows)
    if len(rows) < 100:
        print(f"\n  only {len(rows)} priced rows - not enough to split\n")
        return 0

    lo, hi = split_by_gap(rows, args.cut)

    print()
    print("=" * 66)
    print(f"  {len(rows):,} priced rows, schema {label}"
          + (f", T-{args.horizon}" if args.horizon else ""))
    print(f"  split at |model - book| = {args.cut:.0%}: "
          f"{len(lo):,} low, {len(hi):,} high")
    print("=" * 66)

    # ---- the headline: who is closer in each group -----------------------
    print()
    print("  Who is closer to the outcome")
    print("  " + "-" * 62)
    for name, g in (("low disagreement", lo), ("high disagreement", hi)):
        if len(g) < 15:
            continue
        m = mean([r["_model_err"] for r in g])
        k = mean([r["_mkt_err"] for r in g])
        verdict = "model better" if m < k else "MARKET better"
        print(f"  {name:>20}  n={len(g):>4}  model {m:.4f}  market {k:.4f}  "
              f"-> {verdict}")
    print("  " + "-" * 62)
    print("  If the market is better only in the high group, the model's")
    print("  disagreements are where its errors live.")

    # ---- what is different about the high group --------------------------
    print()
    print("  What differs between the two groups")
    print("  " + "-" * 62)
    print(f"  {'':>24} {'low':>7} {'high':>7}")
    compare(lo, hi, "sigma", lambda r: r.get("sigma"), places=5)
    compare(lo, hi, "|sigmas from money|", lambda r: abs(r["sigmas"]))
    compare(lo, hi, "book spread", lambda r: r.get("_spread"))
    compare(lo, hi, "book price (yes mid)", lambda r: r.get("mkt_p"))
    compare(lo, hi, "model p", lambda r: r["p"])
    compare(lo, hi, "win rate", lambda r: r["_won"], pct=True)
    compare(lo, hi, "took yes side", lambda r: r["_side"] == "yes", pct=True)
    compare(lo, hi, "vol change pct", lambda r: r.get("vol_change_pct"))

    a_lo, a_hi = agents_firing(lo), agents_firing(hi)
    for name in sorted(set(a_lo) | set(a_hi)):
        x, y = a_lo.get(name), a_hi.get(name)
        d = ""
        if x and y and (y / x > 1.3 or y / x < 0.77):
            d = f"   <-- {y / x:.2f}x"
        print(f"  {name + ' push':>24} {fmt(x)} {fmt(y)}{d}")
    print("  " + "-" * 62)
    print("  A row marked <-- differs enough between the groups to be worth")
    print("  chasing. Rows that look the same rule that feature out.")

    # ---- buckets ---------------------------------------------------------
    by_bucket(rows, lambda r: r["_hour"], "By hour of day (UTC)",
              list(range(24)))
    by_bucket(rows, lambda r: r["horizon"], "By horizon", [12, 8, 4])
    by_bucket(rows, lambda r: r["_side"], "By side taken", ["yes", "no"])

    def sigma_bucket(r):
        s = r.get("sigma")
        if not s:
            return None
        if s < 0.0008:
            return "quiet"
        if s < 0.0012:
            return "normal"
        return "loud"
    by_bucket(rows, sigma_bucket, "By volatility", ["quiet", "normal", "loud"])

    def gap_bucket(r):
        g = r["_gap"]
        return ("0-5%" if g < 0.05 else "5-10%" if g < 0.10
                else "10-20%" if g < 0.20 else "20%+")
    by_bucket(rows, gap_bucket, "By size of disagreement",
              ["0-5%", "5-10%", "10-20%", "20%+"])

    print()
    print("  If no column separates the groups and no bucket stands out, the")
    print("  disagreements are not a distinguishable subset. That would mean")
    print("  the model has no information the book lacks, and no filter will")
    print("  change that.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
