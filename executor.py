"""
Demo executor. Manual, one order at a time.

  python executor.py check                    prove auth works
  python executor.py positions                balance, positions, resting orders
  python executor.py rails                    show limits and today's state
  python executor.py order --ticker T --side yes --count 1 --price 40 \
                          --window 20260913T2245 --quoted-at <iso>
  python executor.py cancel --order-id ...
  python executor.py balance                  per-shard balance breakdown
  python executor.py transfer --to 2 --amount 15
  python executor.py sync                     see what settled (writes nothing)
  python executor.py sync --apply             fold it into realised P&L
  python executor.py pnl --amount -1.25       record a result by hand
  python executor.py fees                     the series' own fee terms

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not read predictions.jsonl. It does not call the model. It does not
loop. Every order is typed by a human who chose the ticker, the side, and the
price.

That is not an oversight, and it is not a stub to be filled in later by
whoever reads this next. The plan gates automated trading on Phase A being
positive out of sample. It is not. Wiring the model into this file is the
single change that turns a practice harness into an unproven strategy trading
on its own, so it stays unwired until the evidence changes.

What this file IS for: proving the auth, the order shape, the fills, the
position accounting, and the rails all work. That plumbing needs debugging
regardless of whether the model ever earns its keep, and debugging it on the
demo environment costs nothing.

Every order goes through core.rails.check() first. There is no override.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core import rails as R
from core import reconcile
from core.kalshi_exec import (
    BASE,
    CREDS_FILENAME,
    IS_DEMO,
    Credentials,
    DemoClient,
    KalshiError,
)

HERE = Path(__file__).parent
CREDS_FILE = HERE / CREDS_FILENAME


def banner() -> None:
    env = "DEMO" if IS_DEMO else "*** PRODUCTION ***"
    print()
    print(f"  environment: {env}")
    print(f"  endpoint:    {BASE}")
    if not IS_DEMO:
        print()
        print("  This is not the demo endpoint. Orders placed here use real")
        print("  money. If you did not deliberately edit core/kalshi_exec.py,")
        print("  stop and put it back.")
    print()


def connect() -> DemoClient:
    creds = Credentials.from_file(CREDS_FILE)
    return DemoClient(creds)


def cmd_check(args) -> int:
    banner()
    try:
        client = connect()
    except KalshiError as exc:
        print(f"  {exc}\n")
        return 1

    try:
        bal = client.whoami()
    except KalshiError as exc:
        print(f"  auth failed.\n  {exc}\n")
        return 1

    cents = bal.get("balance")
    print("  auth OK")
    if isinstance(cents, int):
        print(f"  balance: ${cents / 100:,.2f}")
    print()
    return 0


def cmd_positions(args) -> int:
    banner()
    try:
        client = connect()
        bal = client.balance()
        pos = client.positions()
        orders = client.orders()
    except KalshiError as exc:
        print(f"  {exc}\n")
        return 1

    cents = bal.get("balance")
    if isinstance(cents, int):
        print(f"  balance: ${cents / 100:,.2f}")

    market_pos = [p for p in (pos.get("market_positions") or [])
                  if p.get("position")]
    print(f"  open positions: {len(market_pos)}")
    for p in market_pos:
        print(f"    {p.get('ticker'):<28} {p.get('position'):>5}")

    resting = [o for o in (orders.get("orders") or [])
               if o.get("status") == "resting"]
    print(f"  resting orders: {len(resting)}")
    for o in resting:
        print(f"    {o.get('order_id')}  {o.get('ticker')}  "
              f"{o.get('side')} x{o.get('remaining_count')}")
    print()
    return 0


def cmd_rails(args) -> int:
    rails = R.Rails()
    state = R.State(HERE / R.STATE_FILE)
    day = R.utc_day()

    print()
    print("  Rails")
    print("  " + "-" * 52)
    for k, v in rails.to_dict().items():
        print(f"  {k:<24} {v}")
    print()
    print("  State")
    print("  " + "-" * 52)
    print(f"  utc day                  {day}")
    print(f"  realised pnl today       {state.pnl_today(day):+.2f}")
    print(f"  settlements counted      {state.applied_count()}")
    kill = R.kill_switch_active(HERE)
    print(f"  kill switch              {'ACTIVE' if kill else 'clear'}")
    if kill:
        print(f"    remove {HERE / R.KILL_FILE} to resume")
    print()
    return 0


def cmd_balance(args) -> int:
    """
    Per-shard balance breakdown.

    Collateral is per shard. A total that looks healthy tells you nothing
    about whether the shard your market lives on has anything on it.
    """
    banner()
    try:
        client = connect()
        bal = client.balance()
    except KalshiError as exc:
        print(f"  {exc}\n")
        return 1

    total = bal.get("balance_dollars") or bal.get("balance")
    print(f"  total: {total}")
    print()
    print(f"  {'shard':>8} {'balance':>12}")
    print("  " + "-" * 22)
    for row in (bal.get("balance_breakdown") or []):
        print(f"  {row.get('exchange_index'):>8} {row.get('balance'):>12}")
    print()
    print("  An order routed to a shard with 0.00 is rejected as")
    print("  insufficient_shard_balance, regardless of the total.")
    print()
    return 0


def cmd_transfer(args) -> int:
    banner()
    print(f"  moving ${args.amount:.2f} from shard {args.source} "
          f"to shard {args.to}")
    print()

    if not args.yes:
        print("  Dry run. Re-run with --yes to actually move it.")
        print()
        return 0

    try:
        client = connect()
        result = client.transfer(
            dollars=args.amount, src_shard=args.source, dst_shard=args.to
        )
    except KalshiError as exc:
        print(f"  transfer failed.\n  {exc}\n")
        return 1

    print(f"  transfer_id {result.get('transfer_id')}")
    print()
    print("  Transfers are processed asynchronously and cross-shard moves")
    print("  are not atomic. Check `executor.py balance` before ordering.")
    print()
    return 0


def cmd_sync(args) -> int:
    """
    Derive realised P&L from the exchange rather than from memory.

    Defaults to writing nothing. The field mapping in core/reconcile.py was
    written against documentation, not against real responses, so the first
    run should be read by a human before it is trusted to move the number
    that gates trading.
    """
    banner()
    try:
        client = connect()
        settlements = (client.settlements().get("settlements") or [])
        fills = (client.fills().get("fills") or [])
    except KalshiError as exc:
        print(f"  {exc}\n")
        return 1

    state = R.State(HERE / R.STATE_FILE)
    report = reconcile.sync(state, settlements, fills, apply=args.apply)

    print(f"  {len(settlements)} settlement(s), {len(fills)} fill(s)")
    print()

    if args.raw and settlements:
        print("  First raw settlement record, for checking the field mapping:")
        print("  " + json.dumps(settlements[0], indent=2)[:600].replace("\n", "\n  "))
        print()
        if fills:
            print("  First raw fill record:")
            print("  " + json.dumps(fills[0], indent=2)[:400].replace("\n", "\n  "))
            print()

    if report.unparseable:
        print(f"  {len(report.unparseable)} record(s) could not be read:")
        for s_ in report.unparseable[:10]:
            print(f"    {s_.describe()}")
        print()
        print("  These contribute NOTHING to P&L - they are not counted as zero.")
        print("  Fix FIELD_NAMES in core/reconcile.py, then re-run.")
        print()

    if report.unreadable_fills:
        print(f"  {report.unreadable_fills} fill(s) had no readable fee.")
        print("  Fees are therefore understated, which flatters the P&L.")
        print()

    if report.skipped_duplicate:
        print(f"  {len(report.skipped_duplicate)} already counted, skipped.")
        print()

    if report.applied:
        print(f"  {len(report.applied)} new settlement(s):")
        for s_ in report.applied:
            print(f"    {s_.describe()}")
        print()
        print(f"  total {report.total_applied:+.2f}")
    else:
        print("  Nothing new to apply.")
    print()

    if not args.apply:
        print("  Nothing written. Re-run with --apply once the numbers above")
        print("  match what you see in the Kalshi web UI.")
        print()
        return 0

    print(f"  Applied. Realised P&L today is now {state.pnl_today():+.2f}")
    if not report.clean:
        print("  NOTE: some records were unreadable, so this figure is incomplete.")
    print()
    return 0


def cmd_pnl(args) -> int:
    state = R.State(HERE / R.STATE_FILE)
    state.record_pnl(args.amount)
    print(f"\n  recorded {args.amount:+.2f}; "
          f"today now {state.pnl_today():+.2f}\n")
    return 0


def cmd_order(args) -> int:
    banner()

    rails = R.Rails()
    state = R.State(HERE / R.STATE_FILE)

    quoted_at = args.quoted_at
    if quoted_at == "now":
        quoted_at = datetime.now(timezone.utc).isoformat()

    decision = R.check(
        rails=rails,
        state=state,
        window_id=args.window,
        suppressed=args.suppressed,
        quoted_at=quoted_at,
        price_cents=args.price,
        count=args.count,
        root=HERE,
    )

    notional = (args.price * args.count) / 100.0
    from core.kalshi_exec import to_v2
    book_side, yes_price = to_v2(args.side, args.action, args.price)
    print(f"  {args.action} {args.count} x {args.side} @ {args.price}c "
          f"on {args.ticker}")
    print(f"  sends as: {book_side} {yes_price / 100:.4f} on the YES leg")
    print(f"  notional ${notional:.2f}, window {args.window}")
    print()

    if not decision:
        print("  REFUSED")
        for r in decision.reasons:
            print(f"    - {r}")
        print()
        return 2

    print("  rails clear")

    if not args.yes:
        print()
        print("  Dry run. Re-run with --yes to actually send it.")
        print()
        return 0

    try:
        client = connect()

        idx = args.exchange_index
        if idx is None:
            idx = client.exchange_index_for(args.ticker)
            print(f"  market is on shard {idx}")

        result = client.place_limit(
            ticker=args.ticker,
            side=args.side,
            action=args.action,
            count=args.count,
            price_cents=args.price,
            client_order_id=str(uuid.uuid4()),
            time_in_force=args.tif,
            exchange_index=idx,
            post_only=args.post_only,
        )
    except KalshiError as exc:
        print(f"\n  order failed.\n  {exc}\n")
        return 1

    state.record_order(args.window)

    # V2 response: order_id, fill_count, remaining_count, average_fill_price,
    # average_fee_paid, ts_ms. There is no status field.
    print()
    print(f"  sent. order_id {result.get('order_id')}")
    filled = result.get("fill_count")
    remaining = result.get("remaining_count")
    print(f"  filled {filled}, resting {remaining}")
    if result.get("average_fill_price") is not None:
        print(f"  avg fill price {result.get('average_fill_price')}")
    if result.get("average_fee_paid") is not None:
        print(f"  avg fee paid   {result.get('average_fee_paid')}")
    print()
    print(json.dumps(result, indent=2)[:800])
    print()

    if filled in ("0.00", "0", 0):
        print("  Nothing filled - the order is resting. On a zero-volume book")
        print("  it may never fill. Use --tif immediate_or_cancel to find out")
        print("  straight away instead of waiting.")
        print()
    return 0


def cmd_fees(args) -> int:
    """
    Print every fee-related field the exchange reports for the series.

    Deliberately prints raw keys rather than interpreting them: the fee
    schedule changed in 2026 to per-series multipliers, and a parser written
    against a guess of the field names would be exactly the kind of code that
    returns a plausible zero. Read what comes back.
    """
    banner()
    try:
        client = connect()
        resp = client.series(args.series)
    except KalshiError as exc:
        print(f"  {exc}\n")
        return 1
    series = resp.get("series", resp)
    fee_keys = {k: v for k, v in series.items() if "fee" in k.lower()}
    print(f"  series {args.series}")
    if not fee_keys:
        print("  no fee fields found. Full record, to look for them:")
        print("  " + json.dumps(series, indent=2)[:1200].replace("\n", "\n  "))
    for k, v in fee_keys.items():
        print(f"  {k:<28} {v}")
    print()
    print("  The maker thesis needs this series to charge nothing on resting")
    print("  orders. If a maker fee or multiplier appears above, re-run replay")
    print("  with it before going live.")
    print()
    return 0


def cmd_cancel(args) -> int:
    banner()
    try:
        client = connect()
        result = client.cancel(args.order_id)
    except KalshiError as exc:
        print(f"  {exc}\n")
        return 1
    print(f"  cancelled {args.order_id}")
    print(json.dumps(result, indent=2)[:400])
    print()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Kalshi demo executor (manual)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="prove auth works").set_defaults(fn=cmd_check)
    sub.add_parser("positions", help="balance, positions, orders").set_defaults(
        fn=cmd_positions)
    sub.add_parser("rails", help="show limits and state").set_defaults(fn=cmd_rails)

    o = sub.add_parser("order", help="place ONE limit order")
    o.add_argument("--ticker", required=True)
    o.add_argument("--side", required=True, choices=["yes", "no"])
    o.add_argument("--action", default="buy", choices=["buy", "sell"])
    o.add_argument("--count", type=int, required=True)
    o.add_argument("--price", type=int, required=True, help="cents, 1-99")
    o.add_argument("--window", required=True, help="window_id, for the per-window cap")
    o.add_argument("--quoted-at", default="now",
                   help="ISO time the book was read, or 'now'")
    o.add_argument("--suppressed", action="store_true",
                   help="declare this window suppressed (will be refused)")
    o.add_argument("--tif", default="good_till_canceled",
                   choices=["good_till_canceled", "immediate_or_cancel",
                            "fill_or_kill"],
                   help="time in force")
    o.add_argument("--exchange-index", type=int, default=None,
                   help="shard override; read off the market when omitted")
    o.add_argument("--post-only", action="store_true",
                   help="reject instead of matching immediately (maker only)")
    o.add_argument("--yes", action="store_true", help="actually send it")
    o.set_defaults(fn=cmd_order)

    c = sub.add_parser("cancel", help="cancel a resting order")
    c.add_argument("--order-id", required=True)
    c.set_defaults(fn=cmd_cancel)

    sub.add_parser("balance", help="per-shard balance").set_defaults(
        fn=cmd_balance)

    t = sub.add_parser("transfer", help="move collateral between shards")
    t.add_argument("--to", type=int, required=True, help="destination shard")
    t.add_argument("--source", type=int, default=0, help="source shard")
    t.add_argument("--amount", type=float, required=True, help="dollars")
    t.add_argument("--yes", action="store_true", help="actually move it")
    t.set_defaults(fn=cmd_transfer)

    sy = sub.add_parser("sync", help="derive P&L from settlements")
    sy.add_argument("--apply", action="store_true", help="actually write it")
    sy.add_argument("--raw", action="store_true",
                    help="print a raw record to check the field mapping")
    sy.set_defaults(fn=cmd_sync)

    n = sub.add_parser("pnl", help="record a realised result by hand")
    n.add_argument("--amount", type=float, required=True)
    n.set_defaults(fn=cmd_pnl)

    f = sub.add_parser("fees", help="series fee terms from the exchange")
    f.add_argument("--series", default="KXBTC15M")
    f.set_defaults(fn=cmd_fees)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
