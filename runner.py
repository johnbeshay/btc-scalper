"""
Automated MAKER loop. Demo or production, decided by BASE in
core/kalshi_exec.py and nothing else.

    python runner.py --dry-run --once    decide, print, place nothing
    python runner.py --once              one window, then stop
    python runner.py                     keep going

WHAT IT DOES, PER WINDOW
------------------------
  1. Syncs settlements into the rails state, so the loss caps see real
     realised P&L. If any record cannot be read, the runner STOPS: a loss
     cap fed by a sync it cannot trust is worse than no cap.
  2. At T-horizon, prices the ladder and applies replay.decide(..., "bid")
     to every strike - the same function the replay used, so live and
     replay cannot silently disagree about what a trade is.
  3. Posts ONE post-only limit order at the bid. It never crosses the
     spread. If the exchange would have matched it immediately, the
     exchange rejects it instead.
  4. Watches the order until CANCEL_LEAD seconds before close, recording
     when it filled, then cancels whatever is still resting.

WHAT THE LOG IS FOR
-------------------
runner.jsonl records every posted order: price, fill or no fill, seconds to
first fill, and later the settlement. Posted-but-unfilled orders are as
important as filled ones - comparing how the two groups would have settled
is the adverse-selection test the replay cannot run.

PRODUCTION REQUIRES A PLAN FILE
-------------------------------
Against production the runner will not start without live_plan.json: the
threshold, horizon, size and budgets, written down BEFORE the first order
and committed to git. The runner hashes the file into every log line, so a
changed plan is visible in the record. CLI flags cannot override it.

Demo still measures plumbing only. Demo prices do not track production.
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

import hashlib

from core import rails as R
from core import reconcile
from core.kalshi_exec import (BASE, CREDS_FILENAME, IS_DEMO, Credentials,
                              DemoClient, KalshiError)
from core.kalshi_api import DEFAULT_SERIES, KalshiMarketData
from logger import Recorder, window_close, window_id
from replay import decide

HERE = Path(__file__).parent
RUNNER_LOG = HERE / "runner.jsonl"
CREDS = HERE / CREDS_FILENAME
PLAN_FILE = HERE / "live_plan.json"

POLL_SEC = 5          # how often to check a resting order
CANCEL_LEAD = 30      # cancel anything still resting this long before close
PLAN_FIELDS = ("threshold", "horizon", "count", "max_total_loss",
               "max_daily_loss")

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


def choose(record: dict, threshold: float) -> dict | None:
    """
    The maker decision, per strike, using replay.decide(fill="bid").

    Edge is measured against the price actually paid - the bid - not the
    mid, because that is what replay measured and what the P&L depends on.
    Of the strikes that clear the threshold, take the largest edge.

    Buying NO at the NO bid is sent as an ask on YES at 100 - no_bid, which
    rests at the YES ask. Both sides are maker orders; neither crosses.
    """
    if record.get("suppressed"):
        return None

    best = None
    for p in record.get("predictions") or []:
        m = p.get("market")
        if not m or not m.get("ticker"):
            continue
        row = {"p_yes": p["p_yes"], "yes_bid": m.get("yes_bid"),
               "yes_ask": m.get("yes_ask"), "no_bid": m.get("no_bid"),
               "no_ask": m.get("no_ask")}
        d = decide(row, threshold, "bid")
        if d is None:
            continue
        side, price, edge = d
        cents = int(round(price * 100))
        if not (1 <= cents <= 99):
            continue
        if best is None or edge > best["edge"]:
            best = {
                "ticker": m["ticker"], "strike": p["strike"],
                "p_yes": p["p_yes"], "side": side, "price_cents": cents,
                "edge": edge, "yes_bid": m.get("yes_bid"),
                "yes_ask": m.get("yes_ask"), "quoted_at": m.get("quoted_at"),
            }
    return best


def _num(v) -> float | None:
    """Kalshi sends counts as ints or fixed-point strings. None if neither."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_order(resp: dict) -> dict:
    """
    (filled, remaining, status) from an order or create-order response.

    Field names differ between endpoints and API versions, so several are
    tried. Anything unreadable comes back None - never 0 - because "we do
    not know whether it filled" must not be recorded as "it did not fill".
    """
    o = resp.get("order", resp) if isinstance(resp, dict) else {}
    filled = _num(o.get("fill_count_fp", o.get("fill_count")))
    remaining = _num(o.get("remaining_count_fp", o.get("remaining_count")))
    return {"filled": filled, "remaining": remaining,
            "status": o.get("status")}


def watch_order(client, order_id: str, deadline: datetime,
                count: int, now=None, sleep=None, root: Path = HERE) -> dict:
    """
    Poll a resting order until it is fully filled or the deadline passes,
    then cancel what remains. `now` and `sleep` are injectable for tests.

    If the order cannot be read or cancelled, the kill switch is set. An
    order this loop can no longer see or control is the moment to stop.
    """
    now = now or (lambda: datetime.now(timezone.utc))
    sleep = sleep or _sleep
    posted = now()
    out = {"order_id": order_id, "first_fill_after_s": None,
           "filled": 0.0, "cancelled": False, "problem": None}

    while now() < deadline:
        try:
            st = read_order(client.order(order_id))
        except KalshiError as exc:
            out["problem"] = f"could not read order: {exc}"
            break
        if st["filled"] is None:
            out["problem"] = f"unreadable order state: {st}"
            break
        if st["filled"] > out["filled"] and out["first_fill_after_s"] is None:
            out["first_fill_after_s"] = round((now() - posted).total_seconds(), 1)
        out["filled"] = st["filled"]
        if st["filled"] >= count:
            return out
        if st["status"] in ("canceled", "cancelled", "executed", "expired"):
            return out
        if sleep(POLL_SEC):
            break                                 # Ctrl-C: still cancel below

    try:
        client.cancel(order_id)
        out["cancelled"] = True
        # A fill can land between the last poll and the cancel. Read once more.
        st = read_order(client.order(order_id))
        if st["filled"] is not None:
            if st["filled"] > out["filled"] and out["first_fill_after_s"] is None:
                out["first_fill_after_s"] = round(
                    (now() - posted).total_seconds(), 1)
            out["filled"] = st["filled"]
    except KalshiError as exc:
        out["problem"] = (out["problem"] or "") + f" cancel failed: {exc}"

    if out["problem"]:
        (Path(root) / R.KILL_FILE).write_text(
            f"set by runner at {now().isoformat()}: {out['problem']}\n")
    return out


def load_plan(path: Path = PLAN_FILE) -> tuple[dict, str]:
    """
    The pre-committed plan, and a short hash of its exact bytes.

    Refuses on a missing file, a missing field, or a null - so the plan
    cannot be half-written and "filled in later" after seeing results.
    """
    if not path.exists():
        raise SystemExit(f"  no {path.name}. Copy live_plan.example.json, "
                         "fill it in, commit it, then start.")
    raw = path.read_bytes()
    plan = json.loads(raw)
    missing = [k for k in PLAN_FIELDS if plan.get(k) is None]
    if missing:
        raise SystemExit(f"  {path.name} is missing: {', '.join(missing)}")
    if plan["count"] != 1:
        raise SystemExit("  count must be 1 for this experiment.")
    return plan, hashlib.sha256(raw).hexdigest()[:12]


def sync_pnl(client, state) -> bool:
    """Fold new settlements into realised P&L. False if anything was unreadable."""
    settlements = client.settlements().get("settlements") or []
    fills = client.fills().get("fills") or []
    report = reconcile.sync(state, settlements, fills, apply=True)
    if report.applied:
        print(f"   settled: {report.total_applied:+.2f}   "
              f"today {state.pnl_today():+.2f}   total {state.pnl_total():+.2f}")
    if not report.clean:
        print("   SYNC NOT CLEAN - unreadable settlement or fee records.")
        print("   Stopping: the loss caps cannot be trusted. Run")
        print("   `python executor.py sync --raw` and fix the field mapping.")
        return False
    return True


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
    global _stop
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

    if client is not None and not sync_pnl(client, state):
        _stop = True
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

    pick = choose(record, args.threshold)
    if not pick:
        print(f"   nothing over {args.threshold:.0%} edge at the bid; standing down")
        log(dict(window_id=wid, action="no_trade", plan=args.plan_hash))
        wait_for_close(close)
        return

    print(f"   {pick['ticker']}  strike {pick['strike']:,.0f}")
    print(f"   model p_yes {pick['p_yes']:.3f}   book {pick['yes_bid']}/"
          f"{pick['yes_ask']}   edge at bid {pick['edge']:+.3f}")
    print(f"   -> rest: buy {args.count} {pick['side']} @ {pick['price_cents']}c"
          f" (post-only)")

    decision = R.check(
        rails=rails, state=state, window_id=wid, suppressed=record["suppressed"],
        quoted_at=pick["quoted_at"], price_cents=pick["price_cents"],
        count=args.count, root=HERE,
    )
    if not decision:
        print("   REFUSED")
        for r in decision.reasons:
            print(f"     - {r}")
        log(dict(window_id=wid, action="refused", reasons=decision.reasons,
                 plan=args.plan_hash, **pick))
        wait_for_close(close)
        return

    if args.dry_run:
        print("   dry run, not sent")
        log(dict(window_id=wid, action="dry_run", plan=args.plan_hash, **pick))
        wait_for_close(close)
        return

    try:
        idx = client.exchange_index_for(pick["ticker"])
        result = client.place_limit(
            ticker=pick["ticker"], side=pick["side"], action="buy",
            count=args.count, price_cents=pick["price_cents"],
            client_order_id=str(uuid.uuid4()),
            time_in_force="good_till_canceled",
            exchange_index=idx, post_only=True,
        )
    except KalshiError as exc:
        # A post-only rejection lands here too. That is the order refusing
        # to become a taker, which is correct; it is logged, not retried.
        print(f"   order not placed: {exc}")
        log(dict(window_id=wid, action="rejected", error=str(exc),
                 plan=args.plan_hash, **pick))
        wait_for_close(close)
        return

    state.record_order(wid)
    posted_at = datetime.now(timezone.utc)
    first = read_order(result)
    order_id = result.get("order_id") or (result.get("order") or {}).get("order_id")
    if first["filled"]:
        # Should be impossible with post_only. If it happens, post_only is
        # not doing what the docs said, and every fill so far may be a taker.
        print(f"   WARNING: filled {first['filled']} on placement - that is a "
              "TAKER fill. post_only is not working. Setting kill switch.")
        (HERE / R.KILL_FILE).write_text("immediate fill on a post-only order\n")

    print(f"   resting, order {order_id}; watching until T-{CANCEL_LEAD}s")
    outcome = watch_order(client, order_id,
                          close - timedelta(seconds=CANCEL_LEAD), args.count)
    if outcome["filled"]:
        print(f"   FILLED {outcome['filled']:g} after "
              f"{outcome['first_fill_after_s']}s")
    else:
        print("   no fill; cancelled" if outcome["cancelled"]
              else "   no fill")
    if outcome["problem"]:
        print(f"   PROBLEM: {outcome['problem']} - kill switch set")

    log(dict(window_id=wid, action="posted", plan=args.plan_hash,
             posted_at=posted_at.isoformat(), immediate_fill=first["filled"],
             create_response=result, **outcome, **pick))
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
    p = argparse.ArgumentParser(description="Maker loop (demo or production)")
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="decide and print, place nothing")
    p.add_argument("--edge", type=float, default=None,
                   help="DEMO ONLY: threshold override (production reads the plan)")
    p.add_argument("--horizon", type=int, default=None,
                   help="DEMO ONLY: horizon override (production reads the plan)")
    p.add_argument("--series", default=DEFAULT_SERIES)
    args = p.parse_args()

    signal.signal(signal.SIGINT, _handle_stop)

    if IS_DEMO:
        plan, args.plan_hash = ({}, "demo")
        if PLAN_FILE.exists():
            plan, args.plan_hash = load_plan()
        args.threshold = args.edge if args.edge is not None else plan.get("threshold", 0.05)
        args.horizon = args.horizon if args.horizon is not None else plan.get("horizon", 4)
        args.count = 1
        rails = R.Rails()
    else:
        if args.edge is not None or args.horizon is not None:
            print("  --edge and --horizon are refused against production.")
            print("  The plan file is the only source of those numbers.")
            return 1
        plan, args.plan_hash = load_plan()
        args.threshold, args.horizon = plan["threshold"], plan["horizon"]
        args.count = plan["count"]
        rails = R.Rails(max_total_loss=plan["max_total_loss"],
                        max_daily_loss=plan["max_daily_loss"])

    rec = Recorder(RUNNER_LOG.with_name("runner_predictions.jsonl"),
                   horizons=(args.horizon,),
                   market=KalshiMarketData(series=args.series))
    state = R.State(HERE / R.STATE_FILE)

    env = "DEMO" if IS_DEMO else "*** PRODUCTION - REAL MONEY ***"
    print(f"\n  environment: {env}\n  endpoint:    {BASE}")

    client = None
    if not args.dry_run:
        try:
            client = DemoClient(Credentials.from_file(CREDS))
            bal = client.balance()
            print(f"  balance ${(bal.get('balance') or 0) / 100:,.2f}")
        except KalshiError as exc:
            print(f"  {exc}")
            return 1

    if IS_DEMO:
        print("  Demo prices do not track production: this tests plumbing only.")
    print(f"  plan {args.plan_hash}: threshold {args.threshold:.0%} at the bid, "
          f"T-{args.horizon}, {args.count} contract")
    print(f"  loss caps: ${rails.max_daily_loss:.2f}/day, "
          f"${rails.max_total_loss:.2f} total "
          f"(spent so far {state.pnl_total():+.2f})")
    print(f"  kill switch: create {HERE / R.KILL_FILE}")
    print("  Ctrl-C stops after the current window (resting orders are cancelled).")

    seen: set[str] = set()
    while not _stop:
        try:
            wid = window_id(window_close(datetime.now(timezone.utc)))
            if wid in seen:
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
