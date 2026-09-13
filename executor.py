"""
Demo executor. Manual, one order at a time.

  python executor.py check                    prove auth works
  python executor.py positions                balance, positions, resting orders
  python executor.py rails                    show limits and today's state
  python executor.py order --ticker T --side yes --count 1 --price 40 \
                          --window 20260913T2245 --quoted-at <iso>
  python executor.py cancel --order-id ...
  python executor.py pnl --amount -1.25       record a realised result

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
from core.kalshi_exec import (
    BASE,
    IS_DEMO,
    Credentials,
    DemoClient,
    KalshiError,
)

HERE = Path(__file__).parent
CREDS_FILE = HERE / "kalshi-demo-credentials.json"


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
    kill = R.kill_switch_active(HERE)
    print(f"  kill switch              {'ACTIVE' if kill else 'clear'}")
    if kill:
        print(f"    remove {HERE / R.KILL_FILE} to resume")
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
    print(f"  {args.action} {args.count} x {args.side} @ {args.price}c "
          f"on {args.ticker}")
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
        result = client.place_limit(
            ticker=args.ticker,
            side=args.side,
            action=args.action,
            count=args.count,
            price_cents=args.price,
            client_order_id=str(uuid.uuid4()),
        )
    except KalshiError as exc:
        print(f"\n  order failed.\n  {exc}\n")
        return 1

    state.record_order(args.window)

    order = result.get("order", result)
    print()
    print(f"  sent. order_id {order.get('order_id')}")
    print(f"  status {order.get('status')}")
    print()
    print(json.dumps(result, indent=2)[:800])
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
    o.add_argument("--yes", action="store_true", help="actually send it")
    o.set_defaults(fn=cmd_order)

    c = sub.add_parser("cancel", help="cancel a resting order")
    c.add_argument("--order-id", required=True)
    c.set_defaults(fn=cmd_cancel)

    n = sub.add_parser("pnl", help="record a realised result")
    n.add_argument("--amount", type=float, required=True)
    n.set_defaults(fn=cmd_pnl)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
