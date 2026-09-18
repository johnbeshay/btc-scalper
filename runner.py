"""
Watch the model trade, on demo money.

    python runner.py --once              one window, then stop
    python runner.py                     keep going
    python runner.py --maker             REST at the bid and measure fills
    python runner.py --edge 0.10         only trade a 10-point disagreement
    python runner.py --dry-run           decide, print, place nothing

MAKER MODE IS THE POINT OF THIS FILE NOW
----------------------------------------
Replay says the strategy loses on taker fills and makes money on maker fills.
The entire difference is the fee: Kalshi charges nothing on orders that rest
rather than match, and KXBTC15M is not in the maker-fee series list.

But replay's maker number is a CEILING. It assumes every resting order fills
at the quoted bid, and that is certainly false. Two things it cannot model:

  fill rate         a resting buy only fills if someone sells into it. Some
                    fraction of windows will simply pass with no trade, and
                    an edge you cannot execute is not an edge.

  adverse selection the ones that DO fill are not a random sample. Someone
                    sells into your bid when they want out - which is when
                    the price is about to go against you. So maker fills are
                    systematically worse than maker quotes.

Neither can be measured from the log. Both can be measured on demo, for
free, because the matching mechanics are real even though the prices are
not. That is what --maker does: rest at the bid, wait out the window, and
record whether it filled, how long it took, and what happened afterwards.

Run it for a week, then `python makerstats.py`.

WHAT THIS IS NOT
----------------
Evidence about profitability. Demo prices do not track production - measured
on one contract at one instant: demo 6.5-9.5c against production 1.7c,
spread 3 points against 0.1, volume 939 against 1,238,784. The P&L here is
meaningless. The FILL RATE is not: order matching works the same way, and
fill rate is the number this cannot tell you from the log alone.

WHAT THIS IS FOR
----------------
Seeing the model make end-to-end decisions: price a window, compare itself to
the exchange, choose, and live with the result. That is worth watching. It is
also the soak test the plan asks for - does auth survive hours, do shard
balances drift, do settlements always parse, does it survive a sleep.

WHAT THIS IS NOT
----------------
Evidence. Three separate reasons, and they stack:

  1. Demo prices do not track real markets. Kalshi says so directly. The
     first live order here filled at 0.85 on a book the production feed
     showed at 0.91.

  2. Which means the edge is computed against one book and executed against
     another. The model prices from the REAL public Kalshi feed - that part
     is honest - but the fill happens on demo at whatever demo felt like.
     A profit can come entirely from that gap.

  3. The model has not been shown to beat the price it would pay. Skill
     versus market was negative in every distance band, and the adjuster
     fixes have almost no data behind them.

So a green number here means the plumbing works. It does not mean the model
works, and it must not move the Phase C gate. That decision belongs to
score.py and replay.py on real logged data.

IT WRITES ITS OWN LOG
---------------------
runner.jsonl, never predictions.jsonl. Two processes appending to one file
interleave and corrupt it, and the evidence file belongs to the logger alone.
Nothing from demo ever flows back into the record that gets scored.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core import rails as R
from core.kalshi_exec import Credentials, DemoClient, KalshiError
from core.kalshi_api import DEFAULT_SERIES, KalshiMarketData
from logger import Recorder, window_close, window_id

HERE = Path(__file__).parent
RUNNER_LOG = HERE / "runner.jsonl"
CREDS = HERE / "kalshi-demo-credentials.json"

_stop = False


def _handle_stop(signum, frame):
    global _stop
    _stop = True
    print("\n  stopping after this window...")


def yes_mid(market: dict) -> float | None:
    """Mid of the YES book, or the one side that exists."""
    bid, ask = market.get("yes_bid"), market.get("yes_ask")
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    return ask if ask is not None else bid


def choose(record: dict, min_edge: float, maker: bool = False) -> dict | None:
    """
    Pick the strike where the model disagrees with the book the most.

    Returns None when nothing clears the threshold, when the window is
    suppressed, or when no contract has a price to disagree with.

    A note on the threshold, because it is the one knob here and the obvious
    instinct is wrong: in the logged data the win rate FELL as the claimed
    edge rose - 58% down to 8%. When this model strongly disagreed with the
    book, the book was right. So a bigger --edge is not a safer setting. It
    selects for exactly the cases that have gone worst.
    """
    if record.get("suppressed"):
        return None

    best = None
    for p in record.get("predictions") or []:
        m = p.get("market")
        if not m or not m.get("ticker"):
            continue
        mid = yes_mid(m)
        if mid is None:
            continue

        edge = p["p_yes"] - mid
        if best is None or abs(edge) > abs(best["edge"]):
            best = {
                "ticker": m["ticker"],
                "strike": p["strike"],
                "p_yes": p["p_yes"],
                "mid": mid,
                "edge": edge,
                "yes_ask": m.get("yes_ask"),
                "no_ask": m.get("no_ask"),
                "yes_bid": m.get("yes_bid"),
                "no_bid": m.get("no_bid"),
                "quoted_at": m.get("quoted_at"),
            }

    if best is None or abs(best["edge"]) < min_edge:
        return None

    # Buy the side the model thinks is cheap.
    #
    # Taker: pay the ask, fills now, pays the fee.
    # Maker: rest at the bid, pays no fee on this series, may never fill.
    if best["edge"] > 0:
        best["side"] = "yes"
        price = best["yes_bid"] if maker else best["yes_ask"]
    else:
        best["side"] = "no"
        price = best["no_bid"] if maker else best["no_ask"]

    if price is None:
        return None
    best["price_cents"] = int(round(price * 100))
    return best


def wait_for_close(close: datetime) -> bool:
    """
    Sleep until the window has closed.

    Without this the caller loops straight back into the same window and
    decides it again, and again - the first version of this file fired a few
    hundred times on one window because deciding at T-4 returned immediately
    and nothing waited out the remaining four minutes.
    """
    remaining = (close - datetime.now(timezone.utc)).total_seconds() + 20
    return _sleep(remaining) if remaining > 0 else False


def check_spot_freshness(rec: Recorder) -> None:
    """
    Print how old the candle the model is pricing from actually is.

    Spot appeared frozen at one price across several minutes, which cannot
    happen with a live feed. Either the feed is caching, or the candle list
    is ordered newest-first and `candles[-1]` is the OLDEST bar rather than
    the newest - in which case the model has been pricing off a spot two
    hours stale, everywhere, including in the logged data. Printing the
    timestamp settles which.
    """
    try:
        candles = rec.feed.candles(granularity=60, limit=5)
    except Exception as exc:
        print(f"   (could not check feed: {exc})")
        return
    if not candles:
        return
    first, last = candles[0], candles[-1]
    now = datetime.now(timezone.utc)
    age = (now - last.ts).total_seconds() / 60
    print(f"   candle[-1] {last.ts.strftime('%H:%M')} (${last.close:,.2f}, "
          f"{age:+.0f} min old)   candle[0] {first.ts.strftime('%H:%M')} "
          f"(${first.close:,.2f})")
    if age > 5:
        print("   WARNING: the bar being priced from is stale. If candle[0] is")
        print("   newer than candle[-1], the list is reversed and every spot")
        print("   in this project is wrong.")


def poll_until_close(client, order_id: str, close: datetime,
                     every: float = 15.0) -> dict:
    """
    Watch one resting order until the window closes.

    Returns what happened: filled (with the wait in seconds and the price),
    or still resting at close, in which case it is cancelled.

    The wait time matters as much as the fill itself. An order that fills in
    ten seconds filled because the quote was already stale; one that fills
    after nine minutes filled because the market came to it. Those are
    different trades wearing the same label.
    """
    started = datetime.now(timezone.utc)
    last = {}
    while True:
        remaining = (close - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 5 or _stop:
            break
        if _sleep(min(every, max(remaining - 5, 1))):
            break
        try:
            orders = (client.orders().get("orders") or [])
        except KalshiError as exc:
            print(f"   poll failed: {exc}")
            continue
        mine = next((o for o in orders if o.get("order_id") == order_id), None)
        if mine is None:
            # Gone from the book entirely - treat as filled and confirm below.
            last = {}
            break
        last = mine
        filled = float(mine.get("fill_count") or 0)
        if filled > 0 and float(mine.get("remaining_count") or 0) == 0:
            break

    waited = (datetime.now(timezone.utc) - started).total_seconds()

    # Confirm against fills rather than trusting the order row.
    try:
        fills = [f for f in (client.fills().get("fills") or [])
                 if f.get("order_id") == order_id]
    except KalshiError:
        fills = []

    if fills:
        return {
            "filled": True,
            "waited_sec": round(waited, 1),
            "fill_price": fills[0].get("yes_price_dollars"),
            "fee": fills[0].get("fee_cost"),
            "count": fills[0].get("count_fp"),
        }

    # Never filled. Cancel so it cannot settle against us after close.
    try:
        client.cancel(order_id)
        cancelled = True
    except KalshiError as exc:
        cancelled = False
        print(f"   cancel failed: {exc}")
    return {"filled": False, "waited_sec": round(waited, 1),
            "cancelled": cancelled,
            "remaining": last.get("remaining_count")}


def run_window(rec: Recorder, client, rails, state, args) -> None:
    now = datetime.now(timezone.utc)
    close = window_close(now)
    wid = window_id(close)
    print(f"\n  window {wid}  (closes {close.strftime('%H:%M:%S')} UTC)")

    fire_at = close - timedelta(minutes=args.horizon)
    wait = (fire_at - datetime.now(timezone.utc)).total_seconds()
    if wait < -30:
        print(f"   started mid-window, waiting for the next one "
              f"({(close - now).total_seconds() / 60:.0f} min)")
        wait_for_close(close)
        return
    if wait > 0 and _sleep(wait):
        return

    check_spot_freshness(rec)

    record = rec.snapshot(close, args.horizon)
    if not record:
        print("   no reading")
        wait_for_close(close)
        return

    print(f"   spot ${record['spot']:,.2f}   sigma {record['sigma'] * 100:.3f}%"
          f"   {record['ladder']} ladder"
          f"{'   SUPPRESSED' if record['suppressed'] else ''}")

    if record["ladder"] != "kalshi":
        print("   no real book - not trading a synthetic ladder")
        wait_for_close(close)
        return

    pick = choose(record, args.edge, maker=args.maker)
    if not pick:
        print(f"   nothing over {args.edge:.0%} edge; standing down")
        wait_for_close(close)
        return

    print(f"   {pick['ticker']}  strike {pick['strike']:,.0f}")
    print(f"   model {pick['p_yes']:.3f} vs book {pick['mid']:.3f}"
          f"   edge {pick['edge']:+.3f}")
    how = "REST at" if args.maker else "take at"
    print(f"   -> buy 1 {pick['side']} {how} {pick['price_cents']}c")

    decision = R.check(
        rails=rails, state=state, window_id=wid, suppressed=record["suppressed"],
        quoted_at=pick["quoted_at"], price_cents=pick["price_cents"],
        count=1, root=HERE,
    )
    if not decision:
        print("   REFUSED")
        for r in decision.reasons:
            print(f"     - {r}")
        log(dict(window_id=wid, action="refused", reasons=decision.reasons, **pick))
        wait_for_close(close)
        return

    if args.dry_run:
        print("   dry run, not sent")
        log(dict(window_id=wid, action="dry_run", **pick))
        wait_for_close(close)
        return

    try:
        idx = client.exchange_index_for(pick["ticker"])
        result = client.place_limit(
            ticker=pick["ticker"], side=pick["side"], action="buy",
            count=1, price_cents=pick["price_cents"],
            client_order_id=str(uuid.uuid4()),
            time_in_force=("good_till_canceled" if args.maker
                           else "immediate_or_cancel"),
            exchange_index=idx,
        )
    except KalshiError as exc:
        print(f"   order failed: {exc}")
        log(dict(window_id=wid, action="error", error=str(exc), **pick))
        wait_for_close(close)
        return

    state.record_order(wid)

    if not args.maker:
        filled = result.get("fill_count")
        print(f"   filled {filled} @ {result.get('average_fill_price')}"
              f"  fee {result.get('average_fee_paid')}")
        log(dict(window_id=wid, action="sent", result=result, **pick))
        wait_for_close(close)
        return

    # Maker: the order is resting. Watch it.
    order_id = result.get("order_id")
    print(f"   resting  order_id {order_id}")
    outcome = poll_until_close(client, order_id, close)
    if outcome["filled"]:
        print(f"   FILLED after {outcome['waited_sec']:.0f}s "
              f"@ {outcome.get('fill_price')}  fee {outcome.get('fee')}")
    else:
        print(f"   never filled ({outcome['waited_sec']:.0f}s), cancelled")
    log(dict(window_id=wid, action="maker", result=result,
             outcome=outcome, **pick))
    wait_for_close(close)


def log(entry: dict) -> None:
    entry["at"] = datetime.now(timezone.utc).isoformat()
    with RUNNER_LOG.open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _sleep(seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if _stop:
            return True
        time.sleep(min(1.0, end - time.time()))
    return False


def main() -> int:
    p = argparse.ArgumentParser(description="Watch the model trade on demo")
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="decide and print, place nothing")
    p.add_argument("--maker", action="store_true",
                   help="rest at the bid instead of crossing the spread, and "
                        "measure whether it fills")
    p.add_argument("--edge", type=float, default=0.05,
                   help="minimum |model - book| to act on (default 0.05)")
    p.add_argument("--horizon", type=int, default=4,
                   help="minutes before close to decide")
    p.add_argument("--series", default=DEFAULT_SERIES)
    args = p.parse_args()

    signal.signal(signal.SIGINT, _handle_stop)

    rec = Recorder(RUNNER_LOG.with_name("runner_predictions.jsonl"),
                   horizons=(args.horizon,),
                   market=KalshiMarketData(series=args.series))

    rails = R.Rails()
    state = R.State(HERE / R.STATE_FILE)

    client = None
    if not args.dry_run:
        try:
            client = DemoClient(Credentials.from_file(CREDS))
            bal = client.balance()
            print(f"  demo balance ${(bal.get('balance') or 0) / 100:,.2f}")
        except KalshiError as exc:
            print(f"  {exc}")
            return 1

    print("  DEMO ONLY. Demo prices do not track real markets, so the P&L")
    print("  here measures the plumbing, not the model.")
    mode = ("MAKER - resting at the bid, measuring fill rate"
            if args.maker else "TAKER - crossing the spread")
    print(f"  {mode}")
    print(f"  edge threshold {args.edge:.0%}, deciding at T-{args.horizon}")
    print(f"  writing to {RUNNER_LOG.name} (predictions.jsonl untouched)")
    print("  Ctrl-C to stop after the current window.")

    seen: set[str] = set()
    while not _stop:
        try:
            wid = window_id(window_close(datetime.now(timezone.utc)))
            if wid in seen:
                # wait_for_close should make this impossible; if it happens,
                # sleep rather than spin.
                time.sleep(10)
                continue
            seen.add(wid)
            run_window(rec, client, rails, state, args)
        except Exception as exc:                     # keep the soak running
            print(f"   unexpected: {exc}", file=sys.stderr)
            log({"action": "crash", "error": str(exc)})
            time.sleep(10)
        if args.once:
            break

    print("\n  stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
