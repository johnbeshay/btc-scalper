"""
Did the resting orders fill, and were the fills any good?

    python makerstats.py

Reads runner.jsonl, written by `python runner.py --maker`.

THE TWO NUMBERS
---------------
Replay says the strategy loses on taker fills and makes money on maker fills,
and the difference is entirely the fee. But replay's maker figure assumes
every resting order fills at the quoted bid. This measures what actually
happens.

  FILL RATE          What fraction of resting orders got taken. An edge you
                     cannot execute is not an edge. If half the windows pass
                     with no trade, the real return is roughly half what the
                     replay suggests, before anything else.

  ADVERSE SELECTION  Whether the orders that filled did worse than the ones
                     that did not. Someone sells into your bid when they
                     want out, and they want out when the price is about to
                     move against you. So fills are not a random sample of
                     the windows you bid on - they are tilted toward the bad
                     ones.

                     The test: compare how often the model was RIGHT on
                     filled windows against unfilled ones. If the model was
                     right 55% of the time when nothing filled and 40% of
                     the time when something did, the difference is the
                     selection, and it comes straight off the edge.

That second number is the one nothing else in this project can measure. It
needs real orders resting in a real book, which is why this runs on demo
even though demo P&L is meaningless.

WAIT TIME MATTERS TOO
---------------------
An order filled in ten seconds filled because the quote was already stale
when it was placed. One filled after nine minutes filled because the market
came to it. The first is closer to a taker fill wearing a maker label.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RUNNER_LOG = Path(__file__).parent / "runner.jsonl"
PRED_LOG = Path(__file__).parent / "predictions.jsonl"


def load_runner(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("action") == "maker":
                rows.append(rec)
    return rows


def load_settlements(path: Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "settlement" and rec.get("ticker"):
                res = (rec.get("result") or "").lower()
                if res in ("yes", "no"):
                    out[rec["ticker"]] = res
    return out


def model_was_right(row: dict, settled: dict) -> bool | None:
    """Did the side the model wanted actually win? None if unknown."""
    res = settled.get(row.get("ticker"))
    if res is None:
        return None
    return (res == "yes") if row.get("side") == "yes" else (res == "no")


def pct(a: int, b: int) -> str:
    return f"{a / b * 100:.0f}%" if b else "   -"


def main() -> int:
    p = argparse.ArgumentParser(description="Maker fill statistics")
    p.add_argument("--log", default=str(RUNNER_LOG))
    p.add_argument("--predictions", default=str(PRED_LOG))
    args = p.parse_args()

    rows = load_runner(Path(args.log))
    settled = load_settlements(Path(args.predictions))

    print()
    print("=" * 64)
    if not rows:
        print("  No maker attempts in the log yet.")
        print("  Run:  python runner.py --maker")
        print("=" * 64)
        print()
        return 0

    filled = [r for r in rows if (r.get("outcome") or {}).get("filled")]
    unfilled = [r for r in rows if not (r.get("outcome") or {}).get("filled")]

    print(f"  {len(rows)} resting orders placed")
    print("=" * 64)
    print()
    print(f"  FILL RATE   {len(filled)}/{len(rows)} = "
          f"{pct(len(filled), len(rows))}")
    print()
    if len(rows) < 30:
        print(f"  Only {len(rows)} attempts. Below about 30 this number moves")
        print("  several points with one more fill. Keep it running.")
        print()

    # ---- wait times ------------------------------------------------------
    waits = sorted(r["outcome"]["waited_sec"] for r in filled)
    if waits:
        med = waits[len(waits) // 2]
        fast = sum(1 for w in waits if w < 30)
        print("  How long the fills took")
        print("  " + "-" * 60)
        print(f"  fastest {waits[0]:>6.0f}s    median {med:>6.0f}s    "
              f"slowest {waits[-1]:>6.0f}s")
        print(f"  filled within 30s: {fast}/{len(waits)} "
              f"({pct(fast, len(waits))})")
        if fast / len(waits) > 0.5:
            print()
            print("  Most fills came almost immediately, which means the quote")
            print("  was already stale when the order was placed. That is a")
            print("  taker fill wearing a maker label, and it will not behave")
            print("  like the replay's maker number.")
        print("  " + "-" * 60)
        print()

    # ---- adverse selection ----------------------------------------------
    f_right = [model_was_right(r, settled) for r in filled]
    u_right = [model_was_right(r, settled) for r in unfilled]
    f_known = [x for x in f_right if x is not None]
    u_known = [x for x in u_right if x is not None]

    print("  Adverse selection: was the model right more often when")
    print("  nothing filled?")
    print("  " + "-" * 60)
    if len(f_known) < 5 or len(u_known) < 5:
        print(f"  Not enough settled windows yet "
              f"({len(f_known)} filled, {len(u_known)} unfilled).")
        print("  Run backfill.py, or keep logging.")
    else:
        fr = sum(f_known) / len(f_known)
        ur = sum(u_known) / len(u_known)
        print(f"  model right on FILLED windows    {fr * 100:>5.0f}%  "
              f"(n={len(f_known)})")
        print(f"  model right on UNFILLED windows  {ur * 100:>5.0f}%  "
              f"(n={len(u_known)})")
        gap = (ur - fr) * 100
        print(f"  gap                              {gap:>+5.0f} pts")
        print()
        if gap > 8:
            print("  The model was right notably more often on the windows")
            print("  that did NOT fill. That is adverse selection: you are")
            print("  being taken out mainly when you are about to be wrong.")
            print("  Subtract it from the replay's maker figure.")
        elif gap < -8:
            print("  Fills did BETTER than non-fills. Unexpected - check the")
            print("  sample size before believing it.")
        else:
            print("  No meaningful gap. Fills look like a fair sample of the")
            print("  windows bid on, which is the good case.")
    print("  " + "-" * 60)
    print()

    # ---- by edge band ----------------------------------------------------
    bands = defaultdict(lambda: [0, 0])
    for r in rows:
        e = abs(r.get("edge", 0))
        b = ("5-8%" if e < 0.08 else "8-12%" if e < 0.12
             else "12-20%" if e < 0.20 else "20%+")
        bands[b][1] += 1
        if (r.get("outcome") or {}).get("filled"):
            bands[b][0] += 1

    print("  Fill rate by size of disagreement")
    print("  " + "-" * 60)
    print(f"  {'edge':>10} {'filled':>8} {'placed':>8} {'rate':>8}")
    for b in ("5-8%", "8-12%", "12-20%", "20%+"):
        if b in bands:
            f, n = bands[b]
            print(f"  {b:>10} {f:>8} {n:>8} {pct(f, n):>8}")
    print("  " + "-" * 60)
    print("  If the big disagreements fill more readily than the small ones,")
    print("  that is the market telling you it disagrees for a reason.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
