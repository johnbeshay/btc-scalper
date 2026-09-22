"""
Did the resting orders fill, and were the fills any good?

    python makerstats.py
    python makerstats.py --reconcile     check fills against the exchange

Reads runner.jsonl as written by runner.py. Three kinds of line matter:

    {"action": "posted",   ...}  a resting order went out
    {"action": "refused",  ...}  the model wanted to trade; a rail said no
    {"action": "no_trade", ...}  nothing cleared the edge threshold

An earlier version of this file looked for `"action": "maker"`, a format
from a runner that was never deployed, and so reported "no maker attempts"
against a log that contained them. The field names below are taken from
records the deployed runner actually wrote.

THE TWO NUMBERS
---------------
Replay's maker figure assumes every resting order fills at the quoted bid.
This measures what actually happens.

  FILL RATE          What fraction of resting orders got taken. An edge you
                     cannot execute is not an edge.

  ADVERSE SELECTION  Whether the orders that filled did worse than the ones
                     that did not. Someone sells into your bid when they
                     want out, and they want out when the price is about to
                     move against you. Tested by comparing how often the
                     model was right on filled windows against unfilled ones.

A RECORD WITH A `problem` CANNOT BE TRUSTED
-------------------------------------------
When the runner cannot read or cancel its own order, it writes what it last
knew - and that can be flatly wrong. The first order this runner ever placed
was logged as `"filled": 0.0` because the order lookup went to the wrong
exchange shard and 404'd. The account balance says otherwise: it moved by
exactly +$0.35, which is a 65-cent NO contract that filled and won.

So records carrying a `problem` are reported separately and left out of the
fill rate, unless `--reconcile` is given. That flag asks the exchange for its
own fills and overwrites the logged fill count with what actually happened.

WAIT TIME
---------
An order filled within seconds filled because the quote was already stale
when it was placed - closer to a taker fill wearing a maker label. One filled
after several minutes filled because the market came to it.

Standard library only, unless --reconcile, which needs the demo credentials.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RUNNER_LOG = HERE / "runner.jsonl"
PRED_LOG = HERE / "predictions.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def split_actions(records: list[dict]) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for r in records:
        by[r.get("action") or "unknown"].append(r)
    return by


def load_settlements(path: Path) -> dict[str, str]:
    out = {}
    for rec in read_jsonl(path):
        if rec.get("type") == "settlement" and rec.get("ticker"):
            res = (rec.get("result") or "").lower()
            if res in ("yes", "no"):
                out[rec["ticker"]] = res
    return out


def filled_count(r: dict) -> float:
    try:
        return float(r.get("filled") or 0)
    except (TypeError, ValueError):
        return 0.0


def trusted(r: dict) -> bool:
    """A posted record is trustworthy unless the runner flagged a problem."""
    return not r.get("problem")


def reconcile(posted: list[dict]) -> tuple[int, int]:
    """
    Overwrite each record's logged fill count with the exchange's own.

    Returns (records checked, records whose fill count changed). The
    exchange's fills endpoint returns every shard, so this works where the
    runner's order lookup did not.
    """
    from core.kalshi_exec import Credentials, DemoClient, KalshiError

    client = DemoClient(Credentials.from_file(HERE / "kalshi-demo-credentials.json"))
    totals: dict[str, float] = defaultdict(float)
    try:
        fills = client.fills(limit=1000).get("fills") or []
    except KalshiError as exc:
        raise SystemExit(f"  could not fetch fills: {exc}")
    for f in fills:
        oid = f.get("order_id")
        if not oid:
            continue
        try:
            totals[oid] += float(f.get("count_fp") or f.get("count") or 0)
        except (TypeError, ValueError):
            continue

    changed = 0
    for r in posted:
        oid = r.get("order_id")
        if not oid:
            continue
        actual = totals.get(oid, 0.0)
        if abs(actual - filled_count(r)) > 1e-9:
            changed += 1
        r["filled"] = actual
        r["reconciled"] = True
    return len(posted), changed


def model_was_right(r: dict, settled: dict) -> bool | None:
    res = settled.get(r.get("ticker"))
    if res is None:
        return None
    return (res == "yes") if r.get("side") == "yes" else (res == "no")


def pct(a: float, b: float) -> str:
    return f"{a / b * 100:.0f}%" if b else "   -"


def main() -> int:
    ap = argparse.ArgumentParser(description="Maker fill statistics")
    ap.add_argument("--log", default=str(RUNNER_LOG))
    ap.add_argument("--predictions", default=str(PRED_LOG))
    ap.add_argument("--reconcile", action="store_true",
                    help="replace logged fill counts with the exchange's fills")
    args = ap.parse_args()

    by = split_actions(read_jsonl(Path(args.log)))
    posted = by.get("posted", [])
    settled = load_settlements(Path(args.predictions))

    print()
    print("=" * 64)
    decided = sum(len(v) for v in by.values())
    print(f"  {decided} windows decided by the runner")
    for action, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"    {action:<10} {len(rs):>5}")

    refused = by.get("refused", [])
    if refused:
        reasons = Counter()
        for r in refused:
            for reason in r.get("reasons") or ["(none given)"]:
                reasons[reason.split("(")[0].strip()] += 1
        print("  refusals by reason:")
        for reason, n in reasons.most_common():
            print(f"    {n:>5}  {reason}")
    print("=" * 64)

    if not posted:
        print("  No resting orders have been posted yet.")
        if refused:
            print("  Every trade the model wanted was refused by a rail - check")
            print("  the reasons above. A kill switch left set stops everything.")
        print()
        return 0

    if args.reconcile:
        n, changed = reconcile(posted)
        print(f"  reconciled {n} orders against the exchange; "
              f"{changed} had a wrong fill count in the log")
        usable = posted
    else:
        usable = [r for r in posted if trusted(r)]
        flagged = [r for r in posted if not trusted(r)]
        if flagged:
            print(f"  {len(flagged)} of {len(posted)} posted orders carry a "
                  "`problem` and are EXCLUDED:")
            for r in flagged[:5]:
                print(f"    {r.get('window_id')}  {(r.get('problem') or '')[:70]}")
            print("  Their logged fill counts may be wrong. Run with --reconcile")
            print("  to take the exchange's own record instead.")

    print()
    if not usable:
        print("  Nothing usable to measure yet.")
        print()
        return 0

    filled = [r for r in usable if filled_count(r) > 0]
    unfilled = [r for r in usable if filled_count(r) <= 0]
    crossed = [r for r in usable if float(r.get("immediate_fill") or 0) > 0]

    print(f"  FILL RATE   {len(filled)}/{len(usable)} = "
          f"{pct(len(filled), len(usable))}")
    if len(usable) < 30:
        print(f"  Only {len(usable)} orders. Below about 30 this moves several")
        print("  points with a single fill.")
    if crossed:
        print(f"  WARNING: {len(crossed)} post-only order(s) filled on arrival.")
        print("  A post-only order should be rejected, not matched. Either the")
        print("  flag is not being honoured or the book moved mid-send.")
    print()

    waits = sorted(r["first_fill_after_s"] for r in filled
                   if r.get("first_fill_after_s") is not None)
    if waits:
        fast = sum(1 for w in waits if w < 30)
        print("  How long the fills took")
        print("  " + "-" * 60)
        print(f"  fastest {waits[0]:>6.0f}s    median {waits[len(waits) // 2]:>6.0f}s"
              f"    slowest {waits[-1]:>6.0f}s")
        print(f"  filled within 30s: {fast}/{len(waits)} ({pct(fast, len(waits))})")
        if fast / len(waits) > 0.5:
            print("  Most fills came almost immediately: the quote was stale when")
            print("  the order went out. That is a taker fill in disguise.")
        print("  " + "-" * 60)
        print()

    f_known = [x for x in (model_was_right(r, settled) for r in filled) if x is not None]
    u_known = [x for x in (model_was_right(r, settled) for r in unfilled) if x is not None]

    print("  Adverse selection: was the model right more often when")
    print("  nothing filled?")
    print("  " + "-" * 60)
    if len(f_known) < 5 or len(u_known) < 5:
        print(f"  Not enough settled windows yet ({len(f_known)} filled, "
              f"{len(u_known)} unfilled). Run backfill.py, or wait.")
    else:
        fr = sum(f_known) / len(f_known)
        ur = sum(u_known) / len(u_known)
        gap = (ur - fr) * 100
        print(f"  model right on FILLED windows    {fr * 100:>5.0f}%  (n={len(f_known)})")
        print(f"  model right on UNFILLED windows  {ur * 100:>5.0f}%  (n={len(u_known)})")
        print(f"  gap                              {gap:>+5.0f} pts")
        if gap > 8:
            print("  Filled more often when about to be wrong: adverse selection.")
            print("  Subtract it from replay's maker figure.")
        elif gap < -8:
            print("  Fills did BETTER than non-fills. Check the sample size.")
        else:
            print("  No meaningful gap: fills look like a fair sample.")
    print("  " + "-" * 60)
    print()

    bands = defaultdict(lambda: [0, 0])
    for r in usable:
        e = abs(r.get("edge") or 0)
        b = ("5-8%" if e < 0.08 else "8-12%" if e < 0.12
             else "12-20%" if e < 0.20 else "20%+")
        bands[b][1] += 1
        if filled_count(r) > 0:
            bands[b][0] += 1
    print("  Fill rate by size of disagreement")
    print("  " + "-" * 60)
    print(f"  {'edge':>10} {'filled':>8} {'posted':>8} {'rate':>8}")
    for b in ("5-8%", "8-12%", "12-20%", "20%+"):
        if b in bands:
            f, n = bands[b]
            print(f"  {b:>10} {f:>8} {n:>8} {pct(f, n):>8}")
    print("  " + "-" * 60)
    print("  Big disagreements filling more readily than small ones is the")
    print("  market telling you it disagrees for a reason.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
