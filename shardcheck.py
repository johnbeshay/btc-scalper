"""
Find out which exchange shard a market lives on, and where the money is.

Kalshi split trading across multiple matching engines. Collateral does not
follow an order: funds sitting on shard 0 cannot back an order routed to
shard 2. That mismatch is what `insufficient_shard_balance` means, and it
looks identical to having no money at all.

This prints raw JSON deliberately. The field names for the per-shard balance
breakdown have not been confirmed against a real response, so guessing at them
in code would just move the guess somewhere harder to see.

  python shardcheck.py KXBTC15M-26SEP132015-15
"""

from __future__ import annotations

import json
import sys

from core.kalshi_exec import Credentials, DemoClient, KalshiError

CREDS = "kalshi-demo-credentials.json"


def show(label: str, fn):
    print()
    print("=" * 62)
    print(f"  {label}")
    print("=" * 62)
    try:
        result = fn()
    except KalshiError as exc:
        print(f"  FAILED: {exc}")
        return None
    print(json.dumps(result, indent=2)[:2500])
    return result


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    ticker = sys.argv[1]

    client = DemoClient(Credentials.from_file(CREDS))

    bal = show("GET /portfolio/balance", client.balance)

    mkt = show(
        f"GET /markets/{ticker}",
        lambda: client._request("GET", f"/trade-api/v2/markets/{ticker}"),
    )

    print()
    print("=" * 62)
    print("  Summary")
    print("=" * 62)

    idx = None
    if isinstance(mkt, dict):
        m = mkt.get("market", mkt)
        idx = m.get("exchange_index")
    print(f"  market exchange_index: {idx}")

    if isinstance(bal, dict):
        print(f"  balance top-level keys: {sorted(bal)}")
    print()
    print("  If the market's exchange_index is not 0, the $20 is on the wrong")
    print("  shard and has to be transferred before an order will collateralize.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
