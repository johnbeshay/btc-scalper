"""
Score the log. Does the model's 70% actually mean 70%? And does it beat the
price Kalshi was charging?

    python score.py
    python score.py --by-horizon
    python score.py --no-drift        # re-price every row with drift zeroed

Reads predictions.jsonl, joins predictions to outcomes, and reports:

  Brier score      mean squared error of the probabilities. Lower is better.
                   0.25 is what you get by always saying 50%. Above that means
                   the model is worse than useless.

  Skill vs 50%     Brier compared against always-50%. Positive means the model
                   knows something. This number is EASY to inflate: a strike
                   two sigmas away with four minutes left is nearly decided,
                   and getting it right is not information you can sell.

  Skill vs market  Brier compared against the Kalshi mid price. This is the
                   number that decides whether an executor should exist. Only
                   available for rows where the logger captured the book.

  Calibration      predictions bucketed by confidence, compared against how
                   often those cases actually happened. n counts calls;
                   rdg counts independent readings. Nine strikes from one
                   reading are one look at the market, not nine, and the
                   error bars use rdg for that reason.

  Near the money   the 0-0.5 sigma band, shown by default because it is the
                   only band anyone can trade at a sane fee.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from core.kalshi import prob_above

LOG = Path(__file__).parent / "predictions.jsonl"


def load(path: Path, zero_drift: bool = False):
    """
    Join predictions to outcomes. Returns (rows, n_readings, n_unresolved).

    Every row keeps the keys older callers expect (window_id, horizon,
    suppressed, p, sigmas, hit, spot, strike, close) and adds:

        reading   (window_id, horizon) - the independent unit
        ladder    "kalshi" or "synthetic"
        p_yes     model probability the YES contract pays
        mkt_p     market's implied P(above) from the mid, or None
        yes_bid / yes_ask / no_ask   book in dollars, or None
        sigma, drift_pct             so p can be re-derived

    zero_drift re-prices p from spot, strike and sigma with no drift term,
    which is what core/kalshi.py says the model should do and logger.py
    historically did not. Fully offline; the log already has every input.
    """
    preds, outs = [], {}

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
            if rec.get("type") == "prediction":
                preds.append(rec)
            elif rec.get("type") == "outcome":
                outs[rec["window_id"]] = rec

    rows, unresolved = [], 0
    for p in preds:
        out = outs.get(p["window_id"])
        if not out or not out.get("trustworthy", True):
            unresolved += 1
            continue
        close = out["close_price"]
        spot = p["spot"]
        sigma = p.get("sigma")
        for item in p["predictions"]:
            strike = item["strike"]
            prob = item["p_above"]
            if zero_drift and sigma and spot > 0 and strike > 0:
                prob = prob_above(spot, strike, sigma)
            mkt = item.get("market") or {}
            rows.append(
                {
                    "window_id": p["window_id"],
                    "horizon": p["horizon_min"],
                    "reading": (p["window_id"], p["horizon_min"]),
                    "ladder": p.get("ladder", "synthetic"),
                    "suppressed": p.get("suppressed", False),
                    "p": prob,
                    "p_yes": item.get("p_yes", prob),
                    "sigmas": item["sigmas_out"],
                    "hit": 1 if close > strike else 0,
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
    return rows, len(preds), unresolved


def brier(rows, key: str = "p") -> float:
    return sum((r[key] - r["hit"]) ** 2 for r in rows) / len(rows)


def skill_pct(model: float, base: float) -> float:
    return (base - model) / base * 100 if base else 0.0


def n_readings(rows) -> int:
    return len({r["reading"] for r in rows})


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


def report(rows, total_preds, unresolved):
    n = len(rows)
    rdg = n_readings(rows)
    print()
    print("=" * 64)
    print(f"  {n:,} resolved calls from {rdg:,} readings "
          f"({rdg / 3:.0f} windows if 3 per window)")
    if unresolved:
        print(f"  {unresolved:,} readings excluded (no trustworthy close yet)")
    print("=" * 64)

    if n < 50:
        print()
        print("  Not enough data to say anything yet.")
        print("  Keep the logger running. Around 500 calls is where the")
        print(f"  calibration curve starts to mean something; you have {n}.")
        print()
        return

    b = brier(rows)
    base = brier([{**r, "p": 0.5} for r in rows])
    skill = skill_pct(b, base)

    print()
    print(f"  Brier score       {b:.4f}")
    print(f"  Always-50% score  {base:.4f}")
    print(f"  Skill vs 50%      {skill:+.1f}%   ", end="")
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
        print(f"  Rows with a Kalshi price   {len(priced):,} "
              f"({n_readings(priced):,} readings)")
        print(f"  Model Brier on those       {mb:.4f}")
        print(f"  Market-mid Brier           {mk:.4f}")
        print(f"  Skill vs market            {skill_pct(mb, mk):+.1f}%   ", end="")
        s = skill_pct(mb, mk)
        if s > 5:
            print("model beats the book - check it holds near the money")
        elif s > 0:
            print("marginal - fees will eat most of this")
        elif s > -5:
            print("about even with the book")
        else:
            print("the market prices this better than the model does")
    else:
        print("  Skill vs market            (no Kalshi prices in the log)")
        print("  Run the logger with Kalshi reachable to measure edge.")
        print("  Check with:  python kalshi_book.py")

    # ---- calibration ----------------------------------------------------
    print()
    print("  Calibration")
    print("  " + "-" * 62)
    print(f"  {'says':>8}  {'actually':>8}  {'n':>6}  {'rdg':>5}  {'off by':>7}   chart")
    for c in calibration(rows):
        flag = " *" if c["significant"] else "  "
        print(
            f"  {c['predicted'] * 100:>7.1f}%  {c['actual'] * 100:>7.1f}%  "
            f"{c['n']:>6,}  {c['readings']:>5,}  {c['error'] * 100:>+6.1f}%{flag} "
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
    head = f"  {'group':>14}  {'n':>6}  {'rdg':>5}  {'brier':>6}  {'vs 50%':>7}  {'conf bias':>9}"
    if any_market:
        head += f"  {'vs mkt':>7}"
    print(head)
    for g in sorted(grouped):
        rs = grouped[g]
        if len(rs) < 20:
            continue
        b = brier(rs)
        base = brier([{**r, "p": 0.5} for r in rs])
        line = (
            f"  {fmt(g):>14}  {len(rs):>6,}  {n_readings(rs):>5,}  {b:>6.4f}  "
            f"{skill_pct(b, base):>+6.1f}%  {confidence_bias(rs):>+8.1f}%"
        )
        if any_market:
            priced = [r for r in rs if r["mkt_p"] is not None]
            if len(priced) >= 20:
                line += f"  {skill_pct(brier(priced), brier(priced, 'mkt_p')):>+6.1f}%"
            else:
                line += f"  {'-':>7}"
        print(line)
    print("  " + "-" * 62)
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
    args = p.parse_args()

    rows, total, unresolved = load(Path(args.log), zero_drift=args.no_drift)

    if not rows:
        print()
        print(f"  Nothing scored yet. Is {args.log} there and has a window closed?")
        print("  Start with:  python logger.py")
        print()
        return 0

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
