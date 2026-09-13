"""
Replay the log as if it had been traded. Report dollars, not Brier.

    python replay.py                    # default: fill at the ask, 10 contracts
    python replay.py --fill mid         # optimistic: fill at the mid
    python replay.py --contracts 25
    python replay.py --horizon 4        # only trade at T-4

For every reading where the book was recorded, this asks: given what the
model said and what the book was charging, would we have traded, on which
side, and what happened.

WHAT THIS ANSWERS
-----------------
score.py says the model is 13% worse than the book at predicting outcomes.
That is a statement about accuracy. This is a statement about money: on the
trades the model would actually have taken, what was the P&L, how much of it
was fees, and did the model's claimed edge show up as realised edge.

It also sweeps the entry threshold - the minimum gap between model and book
before a trade is taken - because a model that loses on average can still
win on the subset where it disagrees most with the price. Or not. That is
what the sweep finds out.

THE HOLDOUT, AND WHY ONLY ONE NUMBER COUNTS
-------------------------------------------
Choosing a threshold on the same data you then report P&L on is the exact
overfitting that just rejected the calibration correction. So the windows
are split chronologically: the threshold is chosen on the first part, and
P&L is reported on the second. The second number is the only one that
means anything. The first is shown so you can see how much the sweep
flattered itself.

FILL ASSUMPTIONS
----------------
Every book logged so far had volume 0. Whether an order fills at all is
unknown. Two modes:

    ask   pay the ask, taker fee. Realistic if there is liquidity.
    mid   pay the mid, taker fee. Optimistic; assumes someone meets you.

The gap between them is the cost of crossing the spread. If the sign of the
result flips between modes, the liquidity question is the whole story.

ONE TRADE PER WINDOW
--------------------
A live system would not buy the same contract three times as the window
ticks down. The default takes the first reading in each window where the
edge clears the threshold, which is what a live loop would do. --horizon
restricts to one horizon and is the cleaner experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import score
from core.kalshi import KalshiFees

LOG = Path(__file__).parent / "predictions.jsonl"
THRESHOLDS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


@dataclass
class Trade:
    window_id: str
    horizon: int
    side: str           # "yes" or "no"
    contracts: int
    price: float        # paid per contract
    fee: float          # total, dollars
    claimed_edge: float # model prob of winning minus price
    payoff: float       # 1 or 0 per contract
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


def prices(r: dict) -> tuple[float | None, float | None, float | None, float | None]:
    """(yes_ask, no_ask, yes_mid, no_mid). None where the side is empty."""
    yb, ya = r.get("yes_bid"), r.get("yes_ask")
    nb, na = r.get("no_bid"), r.get("no_ask")
    # A missing NO side can be inferred from the YES side: NO ask is what it
    # costs to bet against, which is one minus the YES bid.
    if na is None and yb is not None:
        na = round(1 - yb, 4)
    if nb is None and ya is not None:
        nb = round(1 - ya, 4)
    yes_mid = (yb + ya) / 2 if yb is not None and ya is not None else ya
    no_mid = (nb + na) / 2 if nb is not None and na is not None else na
    return ya, na, yes_mid, no_mid


def yes_pays(r: dict) -> float:
    """Does the YES contract pay out, given where price closed."""
    above = r["hit"] == 1
    return 1.0 if (above if r["yes_direction"] == "above" else not above) else 0.0


def decide(r: dict, threshold: float, fill: str) -> tuple[str, float, float] | None:
    """
    (side, price, claimed_edge) for the better side if its edge clears the
    threshold, else None.
    """
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
    """One trade per window at most: the first qualifying reading."""
    by_window: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["mkt_p"] is None:
            continue
        if horizon is not None and r["horizon"] != horizon:
            continue
        by_window[r["window_id"]].append(r)

    trades = []
    for wid in sorted(by_window):
        # earliest reading first, i.e. largest horizon
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
    claimed: float      # mean claimed edge per contract
    realised: float     # mean realised edge per contract, before fees

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


def split_windows(rows, train_frac: float = 0.6) -> tuple[set, set]:
    wids = sorted({r["window_id"] for r in rows if r["mkt_p"] is not None})
    cut = int(len(wids) * train_frac)
    return set(wids[:cut]), set(wids[cut:])


def money(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.2f}"


def report(rows, fill: str, contracts: int, fees: KalshiFees,
           horizon: int | None, train_frac: float):
    priced = [r for r in rows if r["mkt_p"] is not None]
    train_w, test_w = split_windows(rows, train_frac)
    train = [r for r in priced if r["window_id"] in train_w]
    test = [r for r in priced if r["window_id"] in test_w]

    print()
    print("=" * 66)
    print(f"  Replay: {len(priced):,} priced readings across "
          f"{len(train_w) + len(test_w):,} windows")
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
    print(f"  {'min edge':>8}  {'choose n':>8} {'choose net':>11}  "
          f"{'report n':>8} {'report net':>11}  {'win%':>5}")
    results = []
    for th in THRESHOLDS:
        tr = summarise(simulate(train, th, fill, contracts, fees, horizon))
        te = summarise(simulate(test, th, fill, contracts, fees, horizon))
        results.append((th, tr, te))
        print(f"  {th * 100:>7.0f}%  {tr.n:>8} {money(tr.net):>11}  "
              f"{te.n:>8} {money(te.net):>11}  {te.win_rate * 100:>4.0f}%")
    print("  " + "-" * 64)

    # ---- choose on train, report on test ---------------------------------
    eligible = [x for x in results if x[1].n >= 5]
    if not eligible:
        print()
        print("  No threshold produced 5+ trades in the choosing set.")
        print()
        return
    th, tr, te = max(eligible, key=lambda x: x[1].net)

    print()
    print(f"  Chosen on the first {train_frac * 100:.0f}%: min edge {th * 100:.0f}%")
    print(f"  (it made {money(tr.net)} there on {tr.n} trades - that number is")
    print(f"   flattered by having been chosen; ignore it)")
    print()
    print("  ON HELD-OUT DATA")
    print("  " + "-" * 64)
    if te.n == 0:
        print("  No trades qualified. Nothing to report.")
    else:
        print(f"  trades              {te.n}")
        print(f"  win rate            {te.win_rate * 100:.1f}%")
        print(f"  staked              {money(te.stake)}")
        print(f"  gross P&L           {money(te.gross)}")
        print(f"  fees                {money(-te.fees)}")
        print(f"  net P&L             {money(te.net)}")
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

    # ---- decomposition ---------------------------------------------------
    if te.n and te.gross != 0:
        print()
        if te.gross > 0 and te.net < 0:
            print("  Gross positive, net negative: the fee is the whole problem.")
            print("  Maker fills (resting orders) would cut it by ~75%, IF they fill.")
        elif te.gross < 0:
            print("  Gross negative: the model is losing before fees are counted.")
            print("  Cheaper execution would not fix this.")
        elif abs(te.roi) < 3.0:
            print("  Breakeven within noise. This is not a result either way.")
        elif te.net > 0:
            print("  Net positive on held-out data. Worth checking it survives")
            print("  --fill ask if you ran --fill mid, and a longer sample.")

    # ---- did the two periods agree? --------------------------------------
    tr_sign = [x[1].net > 0 for x in results if x[1].n >= 5]
    te_sign = [x[2].net > 0 for x in results if x[2].n >= 5]
    if tr_sign and te_sign and (all(not s for s in tr_sign) != all(not s for s in te_sign)):
        print()
        print("  WARNING: the choosing period and the reporting period disagree")
        print("  on sign at nearly every threshold. The market changed character")
        print("  between them. No single number here is trustworthy yet.")

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
        print(f"  {'horizon':>8}  {'n':>4}  {'win%':>5}  {'net':>10}  {'roi':>7}")
        for h in sorted({r["horizon"] for r in test}, reverse=True):
            s = summarise(simulate(test, th, fill, contracts, fees, h))
            if s.n:
                print(f"  {'T-' + str(h):>8}  {s.n:>4}  {s.win_rate * 100:>4.0f}%  "
                      f"{money(s.net):>10}  {s.roi:>+6.1f}%")
        print("  " + "-" * 64)

    # ---- by distance -----------------------------------------------------
    if te.n:
        trades = simulate(test, th, fill, contracts, fees, horizon)
        bands = defaultdict(list)
        for t in trades:
            b = ("0.0-0.5 sd" if t.sigmas < 0.5 else "0.5-1.0 sd" if t.sigmas < 1.0
                 else "1.0-2.0 sd" if t.sigmas < 2.0 else "beyond 2 sd")
            bands[b].append(t)
        print()
        print(f"  By distance from the money, held-out, min edge {th * 100:.0f}%")
        print("  " + "-" * 64)
        print(f"  {'band':>12}  {'n':>4}  {'win%':>5}  {'net':>10}  {'roi':>7}")
        for b in ("0.0-0.5 sd", "0.5-1.0 sd", "1.0-2.0 sd", "beyond 2 sd"):
            if b in bands:
                s = summarise(bands[b])
                print(f"  {b:>12}  {s.n:>4}  {s.win_rate * 100:>4.0f}%  "
                      f"{money(s.net):>10}  {s.roi:>+6.1f}%")
        print("  " + "-" * 64)
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
    args = p.parse_args()

    rows, _, _ = score.load(Path(args.log))
    if not rows:
        print(f"\n  Nothing to replay. Is {args.log} there?\n")
        return 0
    if not args.include_suppressed:
        rows = [r for r in rows if not r["suppressed"]] or rows

    if not any(r["mkt_p"] is not None for r in rows):
        print("\n  No Kalshi prices in the log. Nothing can be replayed.")
        print("  Check with:  python kalshi_book.py\n")
        return 0

    report(rows, args.fill, args.contracts, KalshiFees(taker_multiplier=args.taker),
           args.horizon, args.train_frac)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
