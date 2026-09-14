"""
Watch the model trade, on demo money.

    python runner.py --once              one window, then stop
    python runner.py                     keep going
    python runner.py --edge 0.10         only trade a 10-point disagreement
    python runner.py --dry-run           decide, print, place nothing

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


def choose(record: dict, min_edge: float) -> dict | None:
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
                "quoted_at": m.get("quoted_at"),
            }

    if best is None or abs(best["edge"]) < min_edge:
        return None

    # Buy the side the model thinks is cheap, at the ask so it can fill.
    if best["edge"] > 0:
        best["side"], price = "yes", best["yes_ask"]
    else:
        best["side"], price = "no", best["no_ask"]

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

    pick = choose(record, args.edge)
    if not pick:
        print(f"   nothing over {args.edge:.0%} edge; standing down")
        wait_for_close(close)
        return

    print(f"   {pick['ticker']}  strike {pick['strike']:,.0f}")
    print(f"   model {pick['p_yes']:.3f} vs book {pick['mid']:.3f}"
          f"   edge {pick['edge']:+.3f}")
    print(f"   -> buy 1 {pick['side']} @ {pick['price_cents']}c")

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
            time_in_force="immediate_or_cancel",
            exchange_index=idx,
        )
    except KalshiError as exc:
        print(f"   order failed: {exc}")
        log(dict(window_id=wid, action="error", error=str(exc), **pick))
        wait_for_close(close)
        return

    state.record_order(wid)
    filled = result.get("fill_count")
    print(f"   filled {filled} @ {result.get('average_fill_price')}"
          f"  fee {result.get('average_fee_paid')}")
    log(dict(window_id=wid, action="sent", result=result, **pick))
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
