"""
Safety rails.

Every order passes through `check()` before it is sent. A single veto blocks
the order; there is no override flag and no force option, because the whole
value of a rail is that it cannot be argued with at 2am.

THE SIX RAILS
-------------
  1. KILL SWITCH      a file on disk. Create it and everything stops.
  2. DAILY LOSS CAP   realised losses today, against a hard floor.
  3. PER-WINDOW CAP   how many orders one 15-minute window may receive.
  4. SUPPRESSED       the model itself flagged this window as untrustworthy.
  5. STALE BOOK       the quote is older than the freshness bound.
  6. NOTIONAL CAP     the dollar size of this single order.

WHY A FILE FOR THE KILL SWITCH
------------------------------
It works when the process is wedged, when you are on a phone over SSH, and
when you cannot remember any of the CLI flags. `New-Item KILL` stops trading.
That is the entire interface, and it is deliberately the crudest thing in the
codebase.

STATE
-----
Realised P&L and per-window counts live in a small JSON file so the caps
survive a restart. A restart is exactly when you would most like the daily
loss cap to have forgotten - which is why it does not.

Standard library only. Nothing here imports `cryptography` or talks to a
network; the rails must be testable and runnable on any machine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

KILL_FILE = "KILL"
STATE_FILE = "executor_state.json"


@dataclass
class Rails:
    """
    Limits. The defaults are deliberately small.

    These are not tuned numbers - there is no evidence to tune them with. They
    are 'small enough that a bug is affordable' numbers, which is the correct
    basis while the model has never beaten the market price.
    """
    max_daily_loss: float = 10.00       # dollars, realised, per UTC day
    max_orders_per_window: int = 1
    max_notional: float = 5.00          # dollars, one order
    max_book_age_sec: float = 30.0
    allow_suppressed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    """The verdict, and every reason behind it."""
    allowed: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.allowed

    @property
    def why(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "all rails clear"


def utc_day(at: datetime | None = None) -> str:
    at = at or datetime.now(timezone.utc)
    return at.astimezone(timezone.utc).strftime("%Y-%m-%d")


class State:
    """
    Persisted counters: realised P&L per day, orders per window.

    Deliberately dumb. A corrupt or missing file resets to zero rather than
    raising, because a rails failure must never be the thing that stops you
    from flattening a position.
    """

    def __init__(self, path: str | Path = STATE_FILE):
        self.path = Path(path)
        self.data = {"daily_pnl": {}, "window_orders": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(d, dict):
            self.data["daily_pnl"] = d.get("daily_pnl", {}) or {}
            self.data["window_orders"] = d.get("window_orders", {}) or {}

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True))
        except OSError:
            pass

    # -- reads ------------------------------------------------------------

    def pnl_today(self, day: str | None = None) -> float:
        return float(self.data["daily_pnl"].get(day or utc_day(), 0.0))

    def orders_in_window(self, window_id: str) -> int:
        return int(self.data["window_orders"].get(window_id, 0))

    # -- writes -----------------------------------------------------------

    def record_order(self, window_id: str) -> None:
        self.data["window_orders"][window_id] = self.orders_in_window(window_id) + 1
        self.save()

    def record_pnl(self, amount: float, day: str | None = None) -> None:
        day = day or utc_day()
        self.data["daily_pnl"][day] = round(self.pnl_today(day) + amount, 4)
        self.save()


def kill_switch_active(root: str | Path = ".") -> bool:
    return (Path(root) / KILL_FILE).exists()


def book_age_seconds(quoted_at: datetime | str, now: datetime | None = None) -> float:
    """Seconds between when the book was read and now."""
    if isinstance(quoted_at, str):
        quoted_at = datetime.fromisoformat(quoted_at)
    if quoted_at.tzinfo is None:
        quoted_at = quoted_at.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - quoted_at).total_seconds()


def check(
    *,
    rails: Rails,
    state: State,
    window_id: str,
    suppressed: bool,
    quoted_at: datetime | str | None,
    price_cents: int,
    count: int,
    root: str | Path = ".",
    now: datetime | None = None,
) -> Decision:
    """
    Run every rail and collect every failure.

    All rails are evaluated even after the first veto. Being told all four
    reasons an order was refused beats being told the first one and
    rediscovering the rest one run at a time.
    """
    reasons: list[str] = []

    # 1. kill switch
    if kill_switch_active(root):
        reasons.append(f"kill switch present ({Path(root) / KILL_FILE})")

    # 2. daily loss cap
    today = utc_day(now)
    pnl = state.pnl_today(today)
    if pnl <= -abs(rails.max_daily_loss):
        reasons.append(
            f"daily loss cap hit: {pnl:+.2f} on {today}, "
            f"limit {-abs(rails.max_daily_loss):.2f}"
        )

    # 3. per-window cap
    placed = state.orders_in_window(window_id)
    if placed >= rails.max_orders_per_window:
        reasons.append(
            f"window {window_id} already has {placed} order(s), "
            f"limit {rails.max_orders_per_window}"
        )

    # 4. suppressed window
    if suppressed and not rails.allow_suppressed:
        reasons.append("model suppressed this window")

    # 5. stale book
    if quoted_at is None:
        reasons.append("no book timestamp - cannot prove the quote is fresh")
    else:
        age = book_age_seconds(quoted_at, now)
        if age > rails.max_book_age_sec:
            reasons.append(
                f"book is {age:.1f}s old, limit {rails.max_book_age_sec:.0f}s"
            )
        elif age < -5:
            reasons.append(f"book timestamp is {-age:.1f}s in the future - check the clock")

    # 6. notional cap
    if count < 1:
        reasons.append(f"count must be at least 1, got {count}")
    if not (1 <= price_cents <= 99):
        reasons.append(f"price {price_cents}c outside 1-99")
    notional = (price_cents * count) / 100.0
    if notional > rails.max_notional:
        reasons.append(
            f"notional ${notional:.2f} over the ${rails.max_notional:.2f} cap "
            f"({count} x {price_cents}c)"
        )

    return Decision(allowed=not reasons, reasons=reasons)
