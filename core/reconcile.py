"""
Turn exchange records into realised P&L.

WHY THIS EXISTS
---------------
Before this module, the daily loss cap depended on you remembering to type
`executor.py pnl --amount -1.25` after every settled window. A rail that
depends on human diligence is not a rail. Forget once and it silently stops
protecting you, while still printing a reassuring number.

This derives the same figure from what the exchange says actually happened.

THE FIELD NAMES ARE NOT CONFIRMED
---------------------------------
Kalshi's settlement and fill objects are read here by trying several plausible
key names for each quantity. That is not defensive programming for its own
sake - it is an admission that this code was written against documentation
rather than against real responses, and the exact schema has not been seen.

The consequence is deliberate: a record whose fields cannot be read is
reported as UNPARSEABLE and contributes nothing. It is never treated as zero.
A P&L tracker that silently scores unknown records as break-even is worse than
one that refuses, because the loss cap would then be computed from a number
that looks complete and is not.

Run `executor.py sync --dry-run` first. It prints the raw records next to what
it believes they mean. Read them. If the mapping is wrong, fix `FIELD_NAMES`
below - it is one dict - and only then run with --apply.

THE ARITHMETIC
--------------
For a settled market:

    pnl = revenue - cost - fees

`revenue` is what the exchange paid out, `cost` is what the contracts cost to
acquire, `fees` come from the fills that built the position. Settlement
revenue is gross, so fees must be pulled separately from the fill records or
losses will be understated - which is the direction that matters, since it is
the direction that keeps the loss cap from firing.

All exchange money values are integer cents. They are converted to dollars
once, at the boundary, and everything downstream is dollars.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

# Candidate key names, most likely first. Edit here if the real API disagrees.
FIELD_NAMES = {
    "ticker": ("ticker", "market_ticker"),
    "revenue": ("revenue", "settlement_revenue", "payout"),
    "yes_cost": ("yes_total_cost", "yes_cost", "total_yes_cost"),
    "no_cost": ("no_total_cost", "no_cost", "total_no_cost"),
    "settled_at": ("settled_time", "settled_at", "determined_time", "ts"),
    "fee": ("fee", "fee_cents", "taker_fee", "fees"),
    "fill_ticker": ("ticker", "market_ticker"),
    "fill_time": ("created_time", "created_at", "ts"),
    "fill_id": ("trade_id", "fill_id", "id"),
}


def pick(record: dict, names: tuple[str, ...]):
    """First present, non-None value among `names`."""
    for n in names:
        if n in record and record[n] is not None:
            return record[n]
    return None


def cents_to_dollars(v) -> float | None:
    """Exchange money is integer cents. Reject anything that is not a number."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return v / 100.0


def parse_time(v) -> datetime | None:
    if not v:
        return None
    if isinstance(v, (int, float)):
        # epoch seconds; milliseconds if it is implausibly large
        secs = v / 1000.0 if v > 1e11 else v
        try:
            return datetime.fromtimestamp(secs, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(v, str):
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def utc_day_of(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d") if dt else None


@dataclass
class Settled:
    """One settled market, parsed."""
    key: str
    ticker: str | None
    revenue: float | None
    cost: float | None
    fees: float
    day: str | None
    raw: dict
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def pnl(self) -> float | None:
        if not self.ok:
            return None
        return round(self.revenue - self.cost - self.fees, 4)

    def describe(self) -> str:
        if not self.ok:
            return f"UNPARSEABLE  {self.ticker or '?'}  ({'; '.join(self.problems)})"
        return (
            f"{self.ticker:<30} revenue {self.revenue:>7.2f}  "
            f"cost {self.cost:>7.2f}  fees {self.fees:>6.2f}  "
            f"pnl {self.pnl:>+7.2f}  [{self.day}]"
        )


def fees_by_ticker(fills: list[dict]) -> dict[str, float]:
    """
    Total fees per ticker, in dollars.

    A fill whose fee cannot be read contributes nothing rather than zero, and
    the caller is told via `unreadable`. Silently dropping fees understates
    losses.
    """
    totals: dict[str, float] = {}
    unreadable = 0
    for f in fills or []:
        ticker = pick(f, FIELD_NAMES["fill_ticker"])
        fee = cents_to_dollars(pick(f, FIELD_NAMES["fee"]))
        if ticker is None:
            unreadable += 1
            continue
        if fee is None:
            unreadable += 1
            continue
        totals[ticker] = round(totals.get(ticker, 0.0) + fee, 4)
    totals["_unreadable"] = unreadable
    return totals


def settlement_key(s: dict) -> str:
    """
    A stable identity for one settlement, so sync can be run twice safely.

    Ticker plus settle time. If a settlement arrives with neither, it cannot be
    deduplicated and is rejected upstream rather than risking a double count.
    """
    ticker = pick(s, FIELD_NAMES["ticker"]) or "?"
    at = pick(s, FIELD_NAMES["settled_at"]) or "?"
    return f"{ticker}@{at}"


def parse_settlement(s: dict, fees: dict[str, float]) -> Settled:
    problems: list[str] = []

    ticker = pick(s, FIELD_NAMES["ticker"])
    if ticker is None:
        problems.append("no ticker field")

    revenue = cents_to_dollars(pick(s, FIELD_NAMES["revenue"]))
    if revenue is None:
        problems.append(f"no readable revenue (tried {FIELD_NAMES['revenue']})")

    yes_cost = cents_to_dollars(pick(s, FIELD_NAMES["yes_cost"]))
    no_cost = cents_to_dollars(pick(s, FIELD_NAMES["no_cost"]))
    if yes_cost is None and no_cost is None:
        problems.append(f"no readable cost (tried {FIELD_NAMES['yes_cost']})")
    cost = (yes_cost or 0.0) + (no_cost or 0.0)

    at = parse_time(pick(s, FIELD_NAMES["settled_at"]))
    day = utc_day_of(at)
    if day is None:
        problems.append("no readable settle time - cannot attribute to a day")

    key = settlement_key(s)
    if key == "?@?":
        problems.append("no identity fields - cannot deduplicate safely")

    return Settled(
        key=key,
        ticker=ticker,
        revenue=revenue,
        cost=cost if revenue is not None else None,
        fees=fees.get(ticker, 0.0) if ticker else 0.0,
        day=day,
        raw=s,
        problems=problems,
    )


@dataclass
class SyncReport:
    parsed: list[Settled] = field(default_factory=list)
    applied: list[Settled] = field(default_factory=list)
    skipped_duplicate: list[Settled] = field(default_factory=list)
    unparseable: list[Settled] = field(default_factory=list)
    unreadable_fills: int = 0

    @property
    def total_applied(self) -> float:
        return round(sum(s.pnl for s in self.applied), 4)

    @property
    def clean(self) -> bool:
        return not self.unparseable and not self.unreadable_fills


def sync(state, settlements: list[dict], fills: list[dict],
         apply: bool = False) -> SyncReport:
    """
    Fold settlements into realised P&L.

    With apply=False nothing is written. That is the default because the field
    mapping above is unconfirmed, and the first run should be read by a human
    before it is trusted to move the number that gates trading.
    """
    fees = fees_by_ticker(fills)
    unreadable = fees.pop("_unreadable", 0)

    report = SyncReport(unreadable_fills=unreadable)

    for raw in settlements or []:
        s = parse_settlement(raw, fees)
        report.parsed.append(s)

        if not s.ok:
            report.unparseable.append(s)
            continue
        if state.already_applied(s.key):
            report.skipped_duplicate.append(s)
            continue

        report.applied.append(s)
        if apply:
            state.mark_applied(s.key, s.pnl, s.day)

    return report
