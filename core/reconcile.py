"""
Turn exchange records into realised P&L.

WHY THIS EXISTS
---------------
Before this module, the daily loss cap depended on you remembering to type
`executor.py pnl --amount -1.25` after every settled window. A rail that
depends on human diligence is not a rail. Forget once and it silently stops
protecting you, while still printing a reassuring number.

This derives the same figure from what the exchange says actually happened.

THE UNITS ARE MIXED, IN THE SAME OBJECT
---------------------------------------
This mapping is now confirmed against a real settlement. The thing to know is
that one settlement record carries money in two different units:

    revenue: 100                      integer CENTS
    yes_total_cost_dollars: "0.85"    STRING DOLLARS
    fee_cost: "0.009000"              STRING DOLLARS

Reading a dollar string as cents understates a cost by 100x; reading cents as
dollars overstates revenue by the same. Both directions flatter or distort the
P&L that gates trading, so the two units get two separate converters and each
field is mapped to exactly one of them. Never add a field to the wrong list.

FEES COME FROM THE SETTLEMENT, NOT THE FILLS
--------------------------------------------
The settlement record carries `fee_cost` covering the fills that built the
position. Fills carry their own `fee_cost` too, so summing both would
double-count every fee. The settlement is the authority; fills are only a
fallback for a settlement that somehow lacks the field.

WHEN A RECORD CANNOT BE READ
----------------------------
It is reported as UNPARSEABLE and contributes nothing. It is never treated as
zero. A P&L tracker that silently scores unknown records as break-even is
worse than one that refuses, because the loss cap would then be computed from
a number that looks complete and is not. This is not hypothetical: the first
version of this module guessed `yes_total_cost` and the real field is
`yes_total_cost_dollars`. Refusing is what surfaced that.

If Kalshi changes the schema again, `executor.py sync --raw` prints the raw
records beside what this code believes they mean. Fix `CENTS_FIELDS` and
`DOLLAR_FIELDS` below, then re-run.

THE ARITHMETIC
--------------
    pnl = revenue - cost - fees

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

# Confirmed against a real KXBTC15M settlement, 2026-09-14.
FIELD_NAMES = {
    "ticker": ("ticker", "market_ticker"),
    "revenue": ("revenue", "value"),
    "yes_cost": ("yes_total_cost_dollars",),
    "no_cost": ("no_total_cost_dollars",),
    "settled_at": ("settled_time", "settled_at", "determined_time"),
    "fee": ("fee_cost",),
    "fill_ticker": ("ticker", "market_ticker"),
    "fill_time": ("created_time", "created_at", "ts"),
    "fill_id": ("trade_id", "fill_id", "id"),
}

# Only field names confirmed against a real response are listed. Plausible
# legacy names like `yes_total_cost` and `fee_cents` are deliberately absent:
# their unit is unknown, and a name in the wrong converter list is a silent
# 100x error. An unrecognised schema should fail loudly as UNPARSEABLE, not
# be guessed at.

# Which converter each field needs. A field in neither list is a bug.
CENTS_FIELDS = {"revenue"}
DOLLAR_FIELDS = {"yes_cost", "no_cost", "fee"}


def pick(record: dict, names: tuple[str, ...]):
    """First present, non-None value among `names`."""
    for n in names:
        if n in record and record[n] is not None:
            return record[n]
    return None


def cents_to_dollars(v) -> float | None:
    """
    Integer cents -> dollars. Used ONLY for fields in CENTS_FIELDS.

    Rejects strings deliberately. A dollar string like "0.85" read as cents
    would silently become $0.0085, and nothing downstream would notice.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return v / 100.0


def dollars_to_dollars(v) -> float | None:
    """
    Dollar string (or number) -> dollars. For fields in DOLLAR_FIELDS.

    Kalshi sends these as fixed-point strings: "0.850000", "0.009000".
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def money(record: dict, field: str) -> float | None:
    """Read one money field using the converter its unit requires."""
    raw = pick(record, FIELD_NAMES[field])
    if field in CENTS_FIELDS:
        return cents_to_dollars(raw)
    if field in DOLLAR_FIELDS:
        return dollars_to_dollars(raw)
    raise KeyError(f"{field} is in neither CENTS_FIELDS nor DOLLAR_FIELDS")


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
    Total fees per ticker, in dollars. FALLBACK ONLY.

    The settlement record carries its own `fee_cost` covering the fills that
    built the position, and that is what parse_settlement uses. This is here
    for a settlement that lacks the field. Using both would double-count.
    """
    totals: dict[str, float] = {}
    unreadable = 0
    for f in fills or []:
        ticker = pick(f, FIELD_NAMES["fill_ticker"])
        fee = dollars_to_dollars(pick(f, FIELD_NAMES["fee"]))
        if ticker is None or fee is None:
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

    revenue = money(s, "revenue")
    if revenue is None:
        problems.append(f"no readable revenue (tried {FIELD_NAMES['revenue']})")

    yes_cost = money(s, "yes_cost")
    no_cost = money(s, "no_cost")
    if yes_cost is None and no_cost is None:
        problems.append(f"no readable cost (tried {FIELD_NAMES['yes_cost']})")
    cost = (yes_cost or 0.0) + (no_cost or 0.0)

    # The settlement's own fee is authoritative; fills are a fallback.
    fee = money(s, "fee")

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
        fees=fee if fee is not None else (fees.get(ticker, 0.0) if ticker else 0.0),
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
