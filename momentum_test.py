"""
Does "BTC is going up" tell you anything the price does not already?

    python momentum_test.py
    python momentum_test.py --from 12 --to 4

For every logged window, BTC's direction is measured from an early reading
to the decision reading (T-12 to T-4 by default) and compared with how the
contract actually settled.

TWO QUESTIONS, ASKED ON PURPOSE
-------------------------------
  1. The naive one. When BTC was rising, did it finish above the line more
     often? It almost certainly did - a price that has been rising is more
     likely to be sitting above the line already. This is the number an
     up/down indicator would show you, and it looks impressive.

  2. The one that matters. When BTC was rising, did YES win more often than
     the MARKET PRICE at the time said it would? The market has watched the
     same rise, so the question is whether it under-reacted. Only an edge
     here is worth anything, because you pay the market price to trade.

If question 1 says yes and question 2 says no, the indicator is real but
worthless: it describes something the price already reflects. That is the
usual result, and it is why up/down arrows rarely help on markets this
heavily traded.

The interval on question 2 is bootstrapped by window. If it spans zero,
direction adds nothing you could trade.

Standard library only. Reads the existing log.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import score

LOG = Path(__file__).parent / "predictions.jsonl"


def windows(rows, frm: int, to: int):
    """One record per ticker: spot at `frm`, spot and market price at `to`, result."""
    by = defaultdict(dict)
    for r in rows:
        if r.get("ticker"):
            by[r["ticker"]][r["horizon"]] = r
    out = []
    for t, h in by.items():
        a, b = h.get(frm), h.get(to)
        if not a or not b or b.get("mkt_p") is None:
            continue
        out.append({
            "window_id": b["window_id"],
            "move": b["spot"] - a["spot"],
            "typical": b["spot"] * (b.get("sigma") or 0),
            "market": b["mkt_p"],       # market's P(finish above), at `to`
            "above": b["hit"],          # 1 if it finished above the line
        })
    return out


def label(w, flat_frac: float = 0.1) -> str:
    small = (w["typical"] or 1.0) * flat_frac
    return "up" if w["move"] > small else "down" if w["move"] < -small else "flat"


def surprise(ws) -> float | None:
    """Average of (what happened - what the market said). Positive = market too low."""
    if not ws:
        return None
    return sum(w["above"] - w["market"] for w in ws) / len(ws)


def bootstrap_gap(ups, downs, iters: int = 3000, seed: int = 0):
    """Interval on surprise(up) - surprise(down), resampling windows."""
    if len(ups) < 15 or len(downs) < 15:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(iters):
        su = [rng.choice(ups) for _ in ups]
        sd = [rng.choice(downs) for _ in downs]
        draws.append(surprise(su) - surprise(sd))
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Does direction beat the price?")
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--schema", default="latest")
    ap.add_argument("--from", dest="frm", type=int, default=12)
    ap.add_argument("--to", type=int, default=4)
    args = ap.parse_args()

    rows, _, _ = score.load(Path(args.log))
    present = score.schema_summary(rows)
    lab = str(max(present)) if args.schema == "latest" else args.schema
    if lab != "all":
        rows = [r for r in rows if r["schema"] == int(lab)]
    rows = [r for r in rows if not r["suppressed"]]

    ws = windows(rows, args.frm, args.to)
    groups = defaultdict(list)
    for w in ws:
        groups[label(w)].append(w)

    print()
    print("=" * 64)
    print(f"  BTC direction T-{args.frm} -> T-{args.to}, schema {lab}: "
          f"{len(ws)} windows")
    print("=" * 64)
    if len(ws) < 40:
        print("  Not enough windows yet.\n")
        return 0

    print()
    print("  1. The naive question: did it finish above the line more often")
    print("     when BTC had been rising?")
    print("  " + "-" * 60)
    for g in ("up", "flat", "down"):
        w = groups.get(g, [])
        if w:
            rate = sum(x["above"] for x in w) / len(w)
            print(f"  {g:>6}  {len(w):>4} windows   finished above: {rate * 100:>4.0f}%")
    print("  (Expect a big difference here. That is the indicator 'working' -")
    print("   and it is also something the price already knows.)")

    print()
    print("  2. The question that matters: did it beat what the MARKET said?")
    print("  " + "-" * 60)
    print(f"  {'':>6} {'n':>5} {'market said':>12} {'happened':>9} {'surprise':>9}")
    for g in ("up", "flat", "down"):
        w = groups.get(g, [])
        if w:
            said = sum(x["market"] for x in w) / len(w)
            did = sum(x["above"] for x in w) / len(w)
            print(f"  {g:>6} {len(w):>5} {said * 100:>11.0f}% {did * 100:>8.0f}% "
                  f"{(did - said) * 100:>+8.1f}")
    print("  " + "-" * 60)

    ups, downs = groups.get("up", []), groups.get("down", [])
    ci = bootstrap_gap(ups, downs)
    su, sd = surprise(ups), surprise(downs)
    if su is None or sd is None:
        print("  Not enough rising or falling windows to compare.\n")
        return 0

    gap = (su - sd) * 100
    print(f"  surprise when rising minus when falling:  {gap:+.1f} points", end="")
    print(f"   [95% CI {ci[0] * 100:+.1f} to {ci[1] * 100:+.1f}]" if ci else "")
    print()

    if ci and ci[0] > 0:
        print("  Direction beat the market: rising BTC finished above the line")
        print("  more often than the price implied. That would be a real")
        print("  indicator. Check it holds on new data before relying on it,")
        print("  and check it is bigger than the fee (~1.7c at 50c).")
    elif ci and ci[1] < 0:
        print("  The market OVER-reacts to direction: after a rise, YES won less")
        print("  often than priced. Buying with the trend would lose here.")
    else:
        print("  Direction adds nothing the price did not already have. An up/down")
        print("  arrow would look like it works (see question 1) while giving you")
        print("  no edge over the price you pay.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
