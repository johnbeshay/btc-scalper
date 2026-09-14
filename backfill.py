"""
Fetch how Kalshi actually settled every window already in the log.

    python backfill.py             # fetch and append what is missing
    python backfill.py --dry-run   # show what would be fetched, write nothing

Appends one "settlement" line per ticker to predictions.jsonl. Never rewrites
or removes anything: the file stays append-only, and running this twice is
harmless because tickers that already have a settlement line are skipped.

WHY THIS EXISTS
---------------
Until 2026-09-14 the scorer decided every outcome from a Coinbase candle
close. Kalshi settles on CF Benchmarks' BRTI as a 60-second average. Checked
against real settlements, the two disagreed on 20.6% of windows - one in
five - and the disagreements sat near the money, where the trading is.

The windows already logged are still queryable from Kalshi for a while.
This pulls their real results before they age out, so the existing data can
be re-scored against the right answer instead of thrown away.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from core.kalshi_api import DEFAULT_SERIES, KalshiMarketData

LOG = Path(__file__).parent / "predictions.jsonl"


def scan(path: Path) -> tuple[dict[str, str], set[str]]:
    """(ticker -> window_id for every priced prediction, tickers already settled)"""
    wanted: dict[str, str] = {}
    have: set[str] = set()
    if not path.exists():
        return wanted, have
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("type")
            if t == "prediction":
                for item in rec.get("predictions") or []:
                    tk = (item.get("market") or {}).get("ticker")
                    if tk:
                        wanted[tk] = rec["window_id"]
            elif t == "settlement" and rec.get("ticker"):
                have.add(rec["ticker"])
    return wanted, have


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill Kalshi settlements")
    p.add_argument("--log", default=str(LOG))
    p.add_argument("--series", default=DEFAULT_SERIES)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    path = Path(args.log)
    wanted, have = scan(path)
    missing = {t: w for t, w in wanted.items() if t not in have}

    print()
    print(f"  {len(wanted):,} priced tickers in the log")
    print(f"  {len(have):,} already have a settlement line")
    print(f"  {len(missing):,} to fetch")
    if not missing:
        print("  nothing to do\n")
        return 0

    md = KalshiMarketData(series=args.series)
    print("  asking Kalshi...")
    results = md.results_for(missing)

    got = {t: r for t, r in results.items() if t in missing}
    gone = set(missing) - set(got)

    print(f"  {len(got):,} settled results found")
    if gone:
        print(f"  {len(gone):,} not available (not settled yet, or aged out)")

    if args.dry_run:
        print("  dry run - nothing written\n")
        return 0

    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("a") as fh:
        for t, res in sorted(got.items()):
            fh.write(json.dumps({
                "type": "settlement",
                "window_id": missing[t],
                "ticker": t,
                "result": res,
                "at": stamp,
                "backfilled": True,
            }) + "\n")

    yes = sum(1 for r in got.values() if r == "yes")
    print(f"  wrote {len(got):,} settlement lines "
          f"({yes:,} yes, {len(got) - yes:,} no)")
    print()
    print("  Now re-score:  python score.py --schema 2")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
