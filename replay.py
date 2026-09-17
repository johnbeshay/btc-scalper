"""
Replay the log as if it had been traded. Report dollars, not Brier.

    python replay.py                # newest schema, fill at ask, 10 contracts
    python replay.py --schema all   # every era pooled (read the warning)
    python replay.py --fill mid     # optimistic: fill at the mid
    python replay.py --horizon 4    # only trade at T-4

For every reading where the book was recorded, this asks: given what the
model said and what the book was charging, would we have traded, on which
side, and what happened.

ERAS ARE NOT POOLED
-------------------
Records carry a schema version and each version is a different model:

    1, 2  spot from a candle feed running ~5 minutes behind the market
    3     spot from the live ticker
    4     live ticker plus a validated settlement basis

Mixing them produces a P&L for a model that never existed. The default is
the newest version in the log; `--schema all` pools them and says so.

This was not a hypothetical. The first run of this tool on schema 3 data
silently included 185 stale-spot windows and reported a profit for the
mixture.

THE HOLDOUT, AND WHY ONE NUMBER IS STILL NOT ENOUGH
---------------------------------------------------
Choosing a threshold on the same data you report P&L on is overfitting, so
the windows are split chronologically: threshold chosen on the first part,
P&L reported on the second.

That is necessary and it is not sufficient. A held-out number from twenty
trades is mostly noise, and the reported figure is the outcome of a
selection - had the sweep picked a different row, a different number would
be reported with equal confidence. So this now also shows:

  - a bootstrap interval on the held-out P&L, resampling trades
  - the full report column, so a threshold that only looks good because it
    was chosen is visible as such
  - a check of whether the two periods even agree on which direction the
    threshold should move

If the choosing period says "higher threshold is better" and the reporting
period says "lower is better", no threshold has been validated, whatever
single number the procedure lands on.

FILL ASSUMPTIONS
----------------
    ask   pay the ask, taker fee. Realistic if there is liquidity.
    mid   pay the mid, taker fee. Optimistic; assumes someone meets you.

If the sign flips between modes, the liquidity question is the whole story.
Note the "volume 0" belief this file was written under was a reader bug:
these markets trade heavily (over a million contracts on some windows), so
`ask` is the honest default and `mid` is the optimistic one.

ONE TRADE PER WINDOW
--------------------
The default takes the first reading in each window where the edge clears
the threshold, which is what a live loop would do. --horizon restricts to
one horizon and is the cleaner experiment.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import score
from core.kalshi import KalshiFees

LOG = Path(__file__).parent / "predictions.jsonl"
THRESHOLDS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

# Below this many held-out trades, a P&L figure is not a result. Twenty
# trades of ten contracts is a handful of coin flips; the interval will be
# wider than the number.
MIN_TRADES_TO_CLAIM = 30

# Minimum trades on the choosing period before a threshold can be selected.
# Guards return-on-stake against picking a row that made 40% on three trades.
MIN_TRAIN_TRADES = 20


@dataclass
class Trade:
    window_id: str
    horizon: int
    side: str            # "yes" or "no"
    contracts: int
    price: float         # paid per contract
    fee: float           # total, dollars
    claimed_edge: float  # model prob of winning minus price
    payoff: float        # 1 or 0 per contract
    sigmas: float

    @property
    def stake(self) -> float:
        return self.price * self.contracts

    @property
    def gross(self) -> float:
        return (self.payoff - self.price) * self.contracts

    @property
    def net(self) -> float:
        return self.gross - self.fee

    @property
    def won(self) -> bool:
        return self.payoff == 1.0


def prices(r: dict):
    """(yes_ask, no_ask, yes_mid, no_mid). None where the side is empty."""
    yb, ya = r.get("yes_bid"), r.get("yes_ask")
    nb, na = r.get("no_bid"), r.get("no_ask")
    if na is None and yb is not None:
        na = round(1 - yb, 4)
    if nb is None and ya is not None:
        nb = round(1 - ya, 4)
    yes_mid = (yb + ya) / 2 if yb is not None and ya is not None else ya
    no_mid = (nb + na) / 2 if nb is not None and na is not None else na
    return ya, na, yes_mid, no_mid


def yes_pays(r: dict) -> float:
    above = r["hit"] == 1
    return 1.0 if (above if r["yes_direction"] == "above" else not above) else 0.0


def decide(r: dict, threshold: float, fill: str):
    ya, na, ym, nm = prices(r)
    p_yes = r["p_yes"]
    yes_price = ym if fill == "mid" else ya
    no_price = nm if fill == "mid" else na

    best = None
    if yes_price is not None and 0 < yes_price < 1:
        e = p_yes - yes_price
        if e >= threshold:
            best = ("yes", yes_price, e)
    if no_price is not None and 0 < no_price < 1:
        e = (1 - p_yes) - no_price
        if e >= threshold and (best is None or e > best[2]):
            best = ("no", no_price, e)
    return best


def simulate(rows, threshold: float, fill: str, contracts: int,
             fees: KalshiFees, horizon: int | None = None) -> list[Trade]:
    by_window: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["mkt_p"] is None:
            continue
        if horizon is not None and r["horizon"] != horizon:
            continue
        by_window[r["window_id"]].append(r)

    trades = []
    for wid in sorted(by_window):
        for r in sorted(by_window[wid], key=lambda x: -x["horizon"]):
            d = decide(r, threshold, fill)
            if d is None:
                continue
            side, price, edge = d
            payoff = yes_pays(r) if side == "yes" else 1.0 - yes_pays(r)
            trades.append(Trade(
                window_id=wid, horizon=r["horizon"], side=side,
                contracts=contracts, price=price,
                fee=fees.order_fee(price, contracts),
                claimed_edge=edge, payoff=payoff, sigmas=abs(r["sigmas"]),
            ))
            break
    return trades


@dataclass
class Summary:
    n: int
    wins: int
    stake: float
    gross: float
    fees: float
    net: float
    claimed: float
    realised: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def roi(self) -> float:
        return self.net / self.stake * 100 if self.stake else 0.0


def summarise(trades: list[Trade]) -> Summary:
    if not trades:
        return Summary(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n = len(trades)
    return Summary(
        n=n,
        wins=sum(t.won for t in trades),
        stake=sum(t.stake for t in trades),
        gross=sum(t.gross for t in trades),
        fees=sum(t.fee for t in trades),
        net=sum(t.net for t in trades),
        claimed=sum(t.claimed_edge for t in trades) / n,
        realised=sum(t.payoff - t.price for t in trades) / n,
    )


def bootstrap_net(trades: list[Trade], iters: int = 3000, seed: int = 0):
    """
    95% interval on total held-out net P&L, resampling trades.

    Each trade is one window, so trades are the independent unit here and
    resampling them directly is correct. The interval is what turns "+$4.69"
    into a statement you can act on or dismiss: on twenty trades of ten
    contracts, a few hundred percent of swing is normal, and the interval
    says so where the point estimate does not.
    """
    if len(trades) < 5:
        return None
    rng = random.Random(seed)
    nets = [t.net for t in trades]
    draws = []
    for _ in range(iters):
        draws.append(sum(rng.choice(nets) for _ in nets))
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def agreement(results, min_trades: int = 5) -> float | None:
    """
    Do the two periods rank the thresholds the same way?

    Correlation between the choose-column net and the report-column net,
    across thresholds. Near +1 means a threshold that looked good on the
    first period also looked good on the second - the sweep is picking up
    something stable. Near or below 0 means the ranking did not carry over,
    so whichever row the procedure selects is arbitrary.

    An earlier version of this asked which DIRECTION each column sloped and
    compared those. That was wrong: both columns routinely peak somewhere in
    the middle, and a linear slope test reads two identical hump shapes as
    a disagreement. Correlating the columns against each other asks the
    question that actually matters.
    """
    # Compared on return on stake, the same quantity the threshold is
    # selected by. Comparing on total net instead would mix "this threshold
    # is good" with "this threshold trades a lot".
    pts = [(tr.roi, te.roi) for _, tr, te in results
           if tr.n >= min_trades and te.n >= min_trades]
    if len(pts) < 4:
        return None
    xs = [a for a, _ in pts]
    ys = [b for _, b in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def split_windows(rows, train_frac: float = 0.6):
    wids = sorted({r["window_id"] for r in rows if r["mkt_p"] is not None})
    cut = int(len(wids) * train_frac)
    return set(wids[:cut]), set(wids[cut:])


def money(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.2f}"


def report(rows, fill: str, contracts: int, fees: KalshiFees,
           horizon: int | None, train_frac: float, schema_label: str):
    priced = [r for r in rows if r["mkt_p"] is not None]
    train_w, test_w = split_windows(rows, train_frac)
    train = [r for r in priced if r["window_id"] in train_w]
    test = [r for r in priced if r["window_id"] in test_w]

    print()
    print("=" * 66)
    print(f"  Replay: {len(priced):,} priced readings across "
          f"{len(train_w) + len(test_w):,} windows   (schema {schema_label})")
    print(f"  fill at {fill}, {contracts} contracts per trade, taker fee "
          f"{fees.taker_multiplier:.4f}")
    if horizon:
        print(f"  horizon T-{horizon} only")
    print(f"  split: {len(train_w)} windows to choose, {len(test_w)} to report")
    print("=" * 66)

    if len(test_w) < 10:
        print()
        print("  Not enough priced windows to hold anything out yet.")
        print("  Keep the logger running with Kalshi reachable.")
        print()
        return

    # ---- sweep -----------------------------------------------------------
    print()
    print("  Threshold sweep. 'choose' is where the threshold gets picked;")
    print("  'report' is untouched data. Only the report column counts.")
    print("  " + "-" * 64)
    print(f"  {'min edge':>8} {'choose n':>8} {'choose net':>11} "
          f"{'report n':>8} {'report net':>11} {'win%':>5}")

    results = []
    for th in THRESHOLDS:
        tr = summarise(simulate(train, th, fill, contracts, fees, horizon))
        te = summarise(simulate(test, th, fill, contracts, fees, horizon))
        results.append((th, tr, te))
        print(f"  {th * 100:>7.0f}% {tr.n:>8} {money(tr.net):>11} "
              f"{te.n:>8} {money(te.net):>11} {te.win_rate * 100:>4.0f}%")
    print("  " + "-" * 64)

    # ---- do the two periods rank thresholds the same way? ----------------
    agree = agreement(results)
    periods_disagree = agree is not None and agree < 0.0

    if agree is not None:
        print()
        print(f"  Threshold agreement between periods: {agree:+.2f}")
        if agree < 0:
            print("  NEGATIVE. Thresholds that did well on the choosing period")
            print("  did badly on the reporting period. The sweep is fitting")
            print("  noise and whatever it picks below is arbitrary.")
        elif agree < 0.4:
            print("  Weak. The ranking barely carries between periods, so the")
            print("  chosen threshold is not well supported.")
        else:
            print("  The two periods rank thresholds similarly, which is what")
            print("  you want before trusting a chosen threshold at all.")

    # ---- choose on train, report on test ---------------------------------
    #
    # Selected on RETURN ON STAKE, not total net.
    #
    # Total net rewards whichever threshold takes the most trades, which is
    # almost always the lowest one. On a model with real edge concentrated in
    # its strong disagreements, that picks "trade everything": gross positive,
    # net negative once every marginal coin-flip pays a fee. Tested on a
    # synthetic model with a known edge at 5%, selecting on net chose 0% and
    # reported a loss; selecting on ROI found the 5%.
    #
    # The minimum trade count stops ROI from picking a threshold that made
    # 40% on three trades.
    eligible = [x for x in results if x[1].n >= MIN_TRAIN_TRADES]
    if not eligible:
        print()
        print(f"  No threshold produced {MIN_TRAIN_TRADES}+ trades in the "
              "choosing set.\n")
        return

    th, tr, te = max(eligible, key=lambda x: x[1].roi)

    print()
    print(f"  Chosen on the first {train_frac * 100:.0f}%: min edge {th * 100:.0f}% "
          f"(best return on stake, not best total)")
    print(f"  (it made {money(tr.net)} there on {tr.n} trades, {tr.roi:+.1f}% - "
          "that number is")
    print(f"   flattered by having been chosen; ignore it)")

    best_report = max(results, key=lambda x: x[2].net)
    if best_report[0] != th and best_report[2].n >= 5:
        print()
        print(f"  For contrast, the best threshold on the REPORTING period was "
              f"{best_report[0] * 100:.0f}%")
        print(f"  ({money(best_report[2].net)} on {best_report[2].n} trades). "
              "The procedure could not")
        print("  have known that. The gap between the two is the size of the")
        print("  selection problem here.")

    print()
    print("  ON HELD-OUT DATA")
    print("  " + "-" * 64)
    if te.n == 0:
        print("  No trades qualified. Nothing to report.")
        print("  " + "-" * 64 + "\n")
        return

    ci = bootstrap_net(simulate(test, th, fill, contracts, fees, horizon))
    print(f"  trades              {te.n}")
    print(f"  win rate            {te.win_rate * 100:.1f}%")
    print(f"  staked              {money(te.stake)}")
    print(f"  gross P&L           {money(te.gross)}")
    print(f"  fees                {money(-te.fees)}")
    print(f"  net P&L             {money(te.net)}", end="")
    if ci:
        print(f"   [95% CI {money(ci[0])} to {money(ci[1])}]")
    else:
        print()
    print(f"  return on stake     {te.roi:+.1f}%")
    print()
    print(f"  claimed edge        {te.claimed * 100:+.1f} pts per contract")
    print(f"  realised edge       {te.realised * 100:+.1f} pts per contract")
    gap = te.realised - te.claimed
    print(f"  gap                 {gap * 100:+.1f} pts   ", end="")
    if gap < -0.05:
        print("model's edge estimates are inflated")
    elif gap > 0.05:
        print("model is more right than it claims")
    else:
        print("claims roughly match reality")
    print("  " + "-" * 64)

    # ---- the verdict -----------------------------------------------------
    print()
    if periods_disagree:
        print("  VERDICT: not a result. The two periods rank thresholds")
        print("  oppositely, so the chosen threshold carries no information")
        print("  about the future.")
    elif te.n < MIN_TRADES_TO_CLAIM:
        print(f"  VERDICT: only {te.n} held-out trades. Below {MIN_TRADES_TO_CLAIM} "
              "this is not a result")
        print("  in either direction, whatever sign it has. Keep logging.")
    elif ci and ci[0] > 0:
        print("  VERDICT: net positive and the whole interval is above zero.")
        print("  The strongest result this tool can produce. Check it survives")
        print("  --fill ask, a longer sample, and a different horizon.")
    elif ci and ci[1] < 0:
        print("  VERDICT: net negative with the whole interval below zero.")
        print("  This loses money as configured.")
    else:
        print("  VERDICT: the interval spans zero. Not distinguishable from")
        print("  breakeven on this much data.")

    if te.gross > 0 and te.net < 0:
        print("  Gross positive, net negative: the fee is the whole problem.")
        print("  Maker fills (resting orders) would cut it by ~75%, IF they fill.")
    elif te.gross < 0:
        print("  Gross is negative: losing before fees are counted. Cheaper")
        print("  execution would not fix this.")

    # ---- confidence vs accuracy ------------------------------------------
    wr = [(x[0], x[2].win_rate) for x in results if x[2].n >= 8]
    if len(wr) >= 5 and all(b[1] <= a[1] for a, b in zip(wr, wr[1:])):
        print()
        print("  Win rate falls as claimed edge rises, at every threshold. The")
        print("  more the model disagrees with the book, the more often the book")
        print("  is right. Strong disagreement is a signal the MODEL is wrong.")

    # ---- by horizon ------------------------------------------------------
    if horizon is None:
        print()
        print(f"  By horizon, held-out, min edge {th * 100:.0f}%")
        print("  " + "-" * 64)
        print(f"  {'horizon':>8} {'n':>4} {'win%':>5} {'net':>10} {'roi':>7}")
        for h in sorted({r["horizon"] for r in test}, reverse=True):
            s = summarise(simulate(test, th, fill, contracts, fees, h))
            if s.n:
                print(f"  {'T-' + str(h):>8} {s.n:>4} {s.win_rate * 100:>4.0f}% "
                      f"{money(s.net):>10} {s.roi:>+6.1f}%")
        print("  " + "-" * 64)

    # ---- by distance -----------------------------------------------------
    trades = simulate(test, th, fill, contracts, fees, horizon)
    bands = defaultdict(list)
    for t in trades:
        b = ("0.0-0.5 sd" if t.sigmas < 0.5 else "0.5-1.0 sd" if t.sigmas < 1.0
             else "1.0-2.0 sd" if t.sigmas < 2.0 else "beyond 2 sd")
        bands[b].append(t)

    print()
    print(f"  By distance from the money, held-out, min edge {th * 100:.0f}%")
    print("  " + "-" * 64)
    print(f"  {'band':>12} {'n':>4} {'win%':>5} {'net':>10} {'roi':>7}")
    for b in ("0.0-0.5 sd", "0.5-1.0 sd", "1.0-2.0 sd", "beyond 2 sd"):
        if b in bands:
            s = summarise(bands[b])
            flag = "" if s.n >= 10 else "   (too few)"
            print(f"  {b:>12} {s.n:>4} {s.win_rate * 100:>4.0f}% "
                  f"{money(s.net):>10} {s.roi:>+6.1f}%{flag}")
    print("  " + "-" * 64)
    near = bands.get("0.0-0.5 sd")
    if near and len(near) >= 5:
        s = summarise(near)
        print("  The 0.0-0.5 sd band is where the liquidity is and where a live")
        print(f"  system would mostly trade. It made {money(s.net)} on {s.n} trades.")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description="Replay the log as trades")
    p.add_argument("--log", default=str(LOG))
    p.add_argument("--fill", choices=["ask", "mid"], default="ask")
    p.add_argument("--contracts", type=int, default=10)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--taker", type=float, default=0.07,
                   help="taker fee multiplier from your order ticket")
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--include-suppressed", action="store_true")
    p.add_argument("--schema", default="latest",
                   help="record version to replay: a number, 'all', or "
                        "'latest' (default). Eras are different models and "
                        "are never pooled silently.")
    args = p.parse_args()

    rows, _, _ = score.load(Path(args.log))
    if not rows:
        print(f"\n  Nothing to replay. Is {args.log} there?\n")
        return 0

    present = score.schema_summary(rows)
    label = args.schema
    if args.schema.lower() == "latest":
        label = str(max(present))
    if label.lower() != "all":
        want = int(label)
        kept = [r for r in rows if r["schema"] == want]
        if not kept:
            print(f"\n  No schema {want} records. Present: "
                  + ", ".join(f"{v} ({w} windows)" for v, w in present.items())
                  + "\n")
            return 0
        if len(kept) < len(rows):
            other = {v: w for v, w in present.items() if v != want}
            print(f"\n  replaying schema {want} only; excluded "
                  + ", ".join(f"schema {v} ({w} windows)"
                              for v, w in other.items()))
        rows = kept
    elif len(present) > 1:
        print("\n  WARNING: pooling schema versions "
              + ", ".join(str(v) for v in present)
              + ". Records before schema 3 were")
        print("  priced from a candle feed minutes behind the market. The P&L")
        print("  below is for a model that never existed.")

    if not args.include_suppressed:
        rows = [r for r in rows if not r["suppressed"]] or rows

    if not any(r["mkt_p"] is not None for r in rows):
        print("\n  No Kalshi prices in these records. Nothing can be replayed.")
        print("  Check with: python kalshi_book.py\n")
        return 0

    report(rows, args.fill, args.contracts,
           KalshiFees(taker_multiplier=args.taker),
           args.horizon, args.train_frac, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
