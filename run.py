"""
Live runner. Polls the exchange and prints a verdict every cycle.

    python3 run.py                 # 5-minute candles, refresh every 60s
    python3 run.py --once          # single read, then exit
    python3 run.py --json          # machine-readable, for the dashboard
    python3 run.py --taker 0.25    # set your real fee tier

Set your actual fee tier. The defaults assume the worst case (0.60% per
side), and every gate in the system keys off that number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from core.feed import BinanceFeed, CoinbaseFeed, FeedError
from core.indicators import FeeModel
from core.orchestrator import Orchestrator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BTC 15-minute scalping assistant")
    p.add_argument("--exchange", choices=["coinbase", "binance"], default="coinbase")
    p.add_argument("--granularity", type=int, default=300, help="candle size, seconds")
    p.add_argument("--interval", type=int, default=60, help="refresh seconds")
    p.add_argument("--taker", type=float, default=0.60, help="fee %% per side")
    p.add_argument("--slippage", type=float, default=0.05, help="slippage %% per side")
    p.add_argument("--once", action="store_true")
    p.add_argument("--json", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()

    feed = CoinbaseFeed() if args.exchange == "coinbase" else BinanceFeed()
    fees = FeeModel(
        entry_fee_pct=args.taker,
        exit_fee_pct=args.taker,
        slippage_pct=args.slippage,
    )
    orch = Orchestrator(fees=fees)

    if not args.json:
        print(
            f"Watching BTC on {args.exchange} "
            f"({args.granularity // 60}m candles). "
            f"Round-trip cost: {fees.round_trip_pct:.2f}%. Ctrl-C to stop."
        )

    while True:
        try:
            candles = feed.candles(granularity=args.granularity, limit=200)
            verdict = orch.run(candles)

            if args.json:
                payload = verdict.to_dict()
                payload["price"] = candles[-1].close
                payload["breakeven_long"] = round(
                    fees.breakeven_price(candles[-1].close), 2
                )
                print(json.dumps(payload))
            else:
                print(orch.describe(verdict))
                print(
                    f"  price ${candles[-1].close:,.2f}   "
                    f"breakeven exit ${fees.breakeven_price(candles[-1].close):,.2f}"
                )

            sys.stdout.flush()

        except FeedError as exc:
            print(f"feed unavailable: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nstopped")
            return 0

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
