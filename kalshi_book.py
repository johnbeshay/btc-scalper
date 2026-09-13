"""
Look at the Kalshi book for the current 15-minute BTC window.

    python kalshi_book.py             # quotes for the window closing next
    python kalshi_book.py discover    # which series tickers look like BTC?
    python kalshi_book.py --series KXBTC15M

Read-only. Nothing here can place an order.

Use this once before starting the logger. If it prints strikes and prices,
the logger will record the book alongside every prediction and score.py can
measure edge against the market. If it prints nothing, the series ticker is
probably wrong - run `discover` and set KALSHI_BTC_SERIES.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from core.kalshi_api import DEFAULT_SERIES, KalshiError, KalshiMarketData
from logger import window_close


def main() -> int:
    p = argparse.ArgumentParser(description="Read the Kalshi book")
    p.add_argument("command", nargs="?", default="quotes",
                   choices=["quotes", "discover"])
    p.add_argument("--series", default=DEFAULT_SERIES)
    args = p.parse_args()

    md = KalshiMarketData(series=args.series)

    if args.command == "discover":
        try:
            found = md.discover_series("BTC")
        except KalshiError as exc:
            print(f"  could not reach Kalshi: {exc}", file=sys.stderr)
            return 1
        if not found:
            print("  no series matching BTC found")
            return 1
        print("  series that look like Bitcoin:")
        for t in found:
            print(f"    {t}")
        print()
        print("  set the right one with:")
        print("    set KALSHI_BTC_SERIES=<ticker>       (Windows)")
        print("    export KALSHI_BTC_SERIES=<ticker>    (Mac/Linux)")
        return 0

    close = window_close(datetime.now(timezone.utc))
    try:
        quotes = md.quotes_for_window(close)
    except KalshiError as exc:
        print(f"  could not reach Kalshi: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"  series  {args.series}")
    print(f"  window  closes {close.strftime('%H:%M:%S')} UTC")
    if not quotes:
        print("  no open contracts found for this window.")
        print("  either the series ticker is wrong (try: python kalshi_book.py discover)")
        print("  or the next window has not been listed yet - try again in a minute.")
        return 1

    print()
    print(f"  {'ticker':<28} {'strike':>10}  {'yes bid':>7} {'yes ask':>7}  {'mkt P(above)':>12}  {'vol':>6}")
    for q in quotes:
        bid = f"{q.yes_bid:.2f}" if q.yes_bid is not None else "  -  "
        ask = f"{q.yes_ask:.2f}" if q.yes_ask is not None else "  -  "
        imp = f"{q.implied_p_above:.3f}" if q.implied_p_above is not None else "  -  "
        print(f"  {q.ticker:<28} {q.strike:>10,.0f}  {bid:>7} {ask:>7}  {imp:>12}  {q.volume:>6}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
