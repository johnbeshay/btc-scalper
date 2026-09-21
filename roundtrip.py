"""
Does the book move toward the model before the window closes?

    python roundtrip.py
    python roundtrip.py --entry 12 --exit 4
    python roundtrip.py --entry 8 --exit 4

THE IDEA
--------
Holding a contract to settlement is a bet on the outcome. Entering early and
exiting before close is a bet on something else: that the market will come
round to the model's view. If it does, you sell into the move and never face
the binary resolution at all.

That needs one specific thing to be true - when the model disagrees with the
book early in the window, the book drifts toward the model's number by later
in the window. Every window is logged at T-12, T-8 and T-4 against the same
ticker, so this is measurable from the log as it stands.

WHAT IT MEASURES
----------------
  1. THE SIGNAL. Regress the book's move (T-exit mid minus T-entry mid) on
     the model's disagreement at entry. A positive slope means the book
     moves toward the model. The slope also says how much: 0.3 means the
     book closes 30% of the gap, on average, by the exit reading.

     The interval is bootstrapped by window. A slope whose interval spans
     zero is not a signal.

  2. THE MONEY. What a round trip would actually have made, under four
     execution assumptions from best to worst:

         maker in,  maker out    rest at the bid, rest at the ask. No fees,
                                 both legs assumed to fill. The ceiling.
         maker in,  taker out    rest in, cross the spread to get out.
         taker in,  maker out
         taker in,  taker out    cross both times, pay both fees.

     The spread is paid on any leg that crosses it, and that is the part
     that usually kills round trips: a round trip has TWO chances to lose
     the spread where holding to settlement has one.

THE CONFOUND, AND THE CONTROL FOR IT
------------------------------------
A positive slope has a boring explanation that needs no model at all. If the
book's quote at the entry reading is briefly off - a stale level, a thin
moment, a bid that just got lifted - then:

    disagreement = model - book_entry   contains  - noise
    move         = book_exit - book_entry contains - noise   (it reverts)

The same noise sits in both, with the same sign, so they correlate. The book
"moves toward the model" only because the book was temporarily wrong and
went back to where it was. A model that knew nothing would show it too.

The control uses a reading the noise cannot reach. Take the model's
disagreement at the entry reading, and the book's move between two LATER
readings. Entry-time quote noise is not in that later move, so it cannot
manufacture a correlation. (It is still in the regressor, which biases the
slope toward zero - so a positive slope here is evidence, while a zero slope
is not quite proof of nothing.)

A second control asks whether the book just drifts toward 50c: the same
regression with a "model" that always says 0.5. If that shows the same
slope, the effect is reversion to the middle, and the model's information is
beside the point.

An earlier idea - a placebo model built by shuffling the real predictions
between windows - was rejected. The real model tracks the book closely, so
its disagreements are small; shuffled ones are large. The slope divides by
that variance, so the real model would look better than the placebo for
reasons that have nothing to do with information.

WHY THIS COULD BE DIFFERENT FROM EVERYTHING ELSE
------------------------------------------------
Every other test in this project grades the model against the settlement.
This grades it against the book's own later price. A model can be useless at
predicting where BTC ends up and still be useful at predicting where the
MARKET's opinion is heading - if it reacts faster to the same information.

It can also fail in a way the settlement tests cannot show: the book might
move toward the model and still not move far enough to pay for the spread.

Standard library only. Reads the existing log.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import score

LOG = Path(__file__).parent / "predictions.jsonl"
TAKER = 0.07


def fee(price: float, contracts: int = 1) -> float:
    return TAKER * price * (1 - price) * contracts


def mid(r) -> float | None:
    yb, ya = r.get("yes_bid"), r.get("yes_ask")
    if yb is None or ya is None:
        return None
    return (yb + ya) / 2


def pairs(rows, entry: int, exit_: int):
    """
    One (entry, exit) pair per ticker, where both readings have a real book.

    Matched on ticker, not window, because the question is about one
    contract's price over time.
    """
    by_ticker = defaultdict(dict)
    for r in rows:
        t = r.get("ticker")
        if not t or r.get("mkt_p") is None:
            continue
        by_ticker[t][r["horizon"]] = r

    out = []
    for t, h in by_ticker.items():
        a, b = h.get(entry), h.get(exit_)
        if not a or not b:
            continue
        ma, mb = mid(a), mid(b)
        if ma is None or mb is None:
            continue
        # Signed disagreement at entry, in the YES-price frame the book uses.
        d = a["p_yes"] - ma
        out.append({
            "ticker": t,
            "window_id": a["window_id"],
            "disagree": d,
            "move": mb - ma,
            "a": a,
            "b": b,
        })
    return out


def triples(rows, entry: int, mid_h: int, exit_: int):
    """
    Per ticker: disagreement at `entry`, and the book's move from `mid_h` to
    `exit_`. The move skips the entry reading entirely, so noise in the entry
    quote cannot appear in it.
    """
    by_ticker = defaultdict(dict)
    for r in rows:
        t = r.get("ticker")
        if not t or r.get("mkt_p") is None:
            continue
        by_ticker[t][r["horizon"]] = r
    out = []
    for t, h in by_ticker.items():
        a, m, b = h.get(entry), h.get(mid_h), h.get(exit_)
        if not a or not m or not b:
            continue
        ma, mm, mb = mid(a), mid(m), mid(b)
        if None in (ma, mm, mb):
            continue
        out.append({"ticker": t, "window_id": a["window_id"],
                    "disagree": a["p_yes"] - ma, "move": mb - mm,
                    "center": 0.5 - ma})
    return out


def slope(ps) -> float | None:
    """OLS slope of move on disagreement, through the data (with intercept)."""
    if len(ps) < 10:
        return None
    xs = [p["disagree"] for p in ps]
    ys = [p["move"] for p in ps]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    if vx <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx


def bootstrap_slope(ps, iters: int = 3000, seed: int = 0):
    by_w = defaultdict(list)
    for p in ps:
        by_w[p["window_id"]].append(p)
    wins = list(by_w)
    if len(wins) < 20:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(iters):
        s = []
        for _ in wins:
            s.extend(by_w[rng.choice(wins)])
        b = slope(s)
        if b is not None:
            draws.append(b)
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def roundtrip_pnl(p, entry_maker: bool, exit_maker: bool) -> float:
    """
    Per-contract P&L of one round trip, in dollars.

    Buy the side the model favours at entry, sell it at exit. YES side
    prices come straight from the book; NO side is the complement.
    """
    a, b = p["a"], p["b"]
    long_yes = p["disagree"] > 0

    ya_b, ya_a = a.get("yes_bid"), a.get("yes_ask")
    yb_b, yb_a = b.get("yes_bid"), b.get("yes_ask")

    if long_yes:
        buy = ya_b if entry_maker else ya_a          # rest at bid / take ask
        sell = yb_a if exit_maker else yb_b          # rest at ask / hit bid
    else:
        # Holding NO: buy NO at 1-yes_ask (maker) or 1-yes_bid (taker),
        # sell NO at 1-yes_bid (maker) or 1-yes_ask (taker).
        buy = (1 - ya_a) if entry_maker else (1 - ya_b)
        sell = (1 - yb_b) if exit_maker else (1 - yb_a)

    pnl = sell - buy
    if not entry_maker:
        pnl -= fee(buy)
    if not exit_maker:
        pnl -= fee(sell)
    return pnl


def bootstrap_mean(vals_by_w, iters: int = 3000, seed: int = 1):
    wins = list(vals_by_w)
    if len(wins) < 20:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(iters):
        s = []
        for _ in wins:
            s.extend(vals_by_w[rng.choice(wins)])
        if s:
            draws.append(sum(s) / len(s))
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Does the book move toward the model?")
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--schema", default="latest")
    ap.add_argument("--entry", type=int, default=12, help="entry horizon, minutes")
    ap.add_argument("--exit", dest="exit_", type=int, default=4,
                    help="exit horizon, minutes")
    ap.add_argument("--min-edge", type=float, default=0.05,
                    help="only trade disagreements at least this large")
    args = ap.parse_args()

    rows, _, _ = score.load(Path(args.log))
    present = score.schema_summary(rows)
    label = str(max(present)) if args.schema == "latest" else args.schema
    if label != "all":
        rows = [r for r in rows if r["schema"] == int(label)]
    rows = [r for r in rows if not r["suppressed"]]

    ps = pairs(rows, args.entry, args.exit_)

    print()
    print("=" * 66)
    print(f"  Round trips T-{args.entry} -> T-{args.exit_}, schema {label}")
    print(f"  {len(ps):,} contracts with a book at both readings, "
          f"{len({p['window_id'] for p in ps}):,} windows")
    print("=" * 66)
    if len(ps) < 30:
        print("  Not enough matched pairs.\n")
        return 0

    # ---- 1. the signal ---------------------------------------------------
    b = slope(ps)
    ci = bootstrap_slope(ps)
    print()
    print("  1. Does the book move toward the model?")
    print("  " + "-" * 62)
    print(f"  slope of book move on model disagreement   {b:+.3f}", end="")
    if ci:
        print(f"   [95% CI {ci[0]:+.3f} to {ci[1]:+.3f}]")
    else:
        print()

    agree = sum(1 for p in ps if p["disagree"] * p["move"] > 0)
    nonzero = sum(1 for p in ps if p["disagree"] != 0 and p["move"] != 0)
    if nonzero:
        print(f"  book moved the model's way in {agree}/{nonzero} "
              f"({agree / nonzero * 100:.0f}%)")
    print("  " + "-" * 62)

    if ci and ci[0] > 0:
        print(f"  YES. The book closes about {b * 100:.0f}% of the gap to the model")
        print(f"  between T-{args.entry} and T-{args.exit_}, and the interval excludes")
        print("  zero. The model is anticipating where the market goes.")
    elif ci and ci[1] < 0:
        print("  The book moves AWAY from the model. When they disagree early,")
        print("  the market gets more confident in its own view, not less.")
    else:
        print("  No detectable movement toward the model. The book does not")
        print("  come round to the model's view within the window.")

    # ---- 1b. controls ----------------------------------------------------
    print()
    print("  1b. Is it the model, or the book correcting itself?")
    print("  " + "-" * 62)

    # control A: the move between two later readings, skipping entry noise
    mids = sorted(h for h in (8, 4) if args.exit_ < h < args.entry)
    control_a = None
    if mids:
        mh = mids[-1]
        tr = triples(rows, args.entry, mh, args.exit_)
        ca = slope(tr)
        cci = bootstrap_slope(tr)
        control_a = (ca, cci)
        print(f"  model at T-{args.entry} vs book move T-{mh} -> T-{args.exit_}:  "
              f"{ca:+.3f}", end="")
        print(f"   [{cci[0]:+.3f} to {cci[1]:+.3f}]" if cci else "")
        print("  (entry quote noise cannot reach this move)")
    else:
        print(f"  no reading between T-{args.entry} and T-{args.exit_} to skip to;")
        print("  use --entry 12 --exit 4 for the control")

    # control B: a "model" that always says 0.5
    centre = [dict(p, disagree=0.5 - mid(p["a"])) for p in ps]
    cb = slope(centre)
    cbci = bootstrap_slope(centre)
    print(f"  a model that always says 0.5:            {cb:+.3f}", end="")
    print(f"   [{cbci[0]:+.3f} to {cbci[1]:+.3f}]" if cbci else "")
    print("  " + "-" * 62)

    if control_a and control_a[1]:
        ca, cci = control_a
        if cci[0] > 0:
            print("  The slope survives with entry noise removed. The model is")
            print("  anticipating real moves in the book, not riding a bounce.")
        elif ci and ci[0] > 0:
            print("  The slope does NOT survive once entry-quote noise is taken")
            print("  out. Most likely the book was briefly off at the entry reading")
            print("  and simply went back - which any model, or none, would 'call'.")
        else:
            print("  Inconclusive: neither the raw slope nor the control is")
            print("  distinguishable from zero.")
    if cbci and cbci[0] > 0 and ci and b is not None and cb is not None \
            and abs(cb - b) < 0.5 * abs(b):
        print("  The always-0.5 model shows a similar slope, so a large part of")
        print("  this is the book drifting toward the middle, not model skill.")

    # ---- 2. the money ----------------------------------------------------
    traded = [p for p in ps if abs(p["disagree"]) >= args.min_edge]
    print()
    print(f"  2. What a round trip would make "
          f"(|disagreement| >= {args.min_edge:.0%}, {len(traded)} trades)")
    print("  " + "-" * 62)
    print(f"  {'execution':>26} {'per contract':>13}   95% CI")

    results = {}
    for label_, em, xm in (("maker in, maker out", True, True),
                           ("maker in, taker out", True, False),
                           ("taker in, maker out", False, True),
                           ("taker in, taker out", False, False)):
        by_w = defaultdict(list)
        for p in traded:
            v = roundtrip_pnl(p, em, xm)
            if v == v:  # not NaN
                by_w[p["window_id"]].append(v)
        vals = [v for vs in by_w.values() for v in vs]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        c = bootstrap_mean(by_w)
        results[label_] = (m, c)
        cstr = f"[{c[0] * 100:+.1f}c to {c[1] * 100:+.1f}c]" if c else ""
        print(f"  {label_:>26} {m * 100:>+11.2f}c   {cstr}")
    print("  " + "-" * 62)

    best = results.get("maker in, maker out")
    if best and best[1] and best[1][0] > 0:
        print("  Maker/maker is positive with the whole interval above zero.")
        print("  But it assumes BOTH legs fill as resting orders, which is the")
        print("  same fill question as before, asked twice.")
    elif best and best[1] and best[1][1] < 0:
        print("  Even the ceiling - both legs resting, no fees, every order")
        print("  filled - loses. Round trips do not work on this market.")
    else:
        print("  The ceiling spans zero. Nothing to capture on this much data,")
        print("  even assuming both legs fill for free.")

    tt = results.get("taker in, taker out")
    if tt:
        print()
        verb = "earns" if tt[0] >= 0 else "loses"
        print(f"  Crossing both ways {verb} {abs(tt[0]) * 100:.2f}c per contract on "
              "average, after")
        print("  two spreads and two fees - against one of each if held to")
        print("  settlement.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
