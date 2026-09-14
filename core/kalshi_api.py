"""
Read-only Kalshi market data.

    from core.kalshi_api import KalshiMarketData
    md = KalshiMarketData()                 # series from KALSHI_BTC_SERIES or default
    quotes = md.quotes_for_window(close)    # list[Quote] for the window closing at `close`

This talks to the public market-data endpoints only. No credentials, no
order placement. Order placement does not exist in this project yet, on
purpose - nothing should be able to trade until the model has been shown to
beat the price it would pay, and that measurement is what this module makes
possible.

WHY THIS EXISTS
---------------
score.py can tell you whether the model's 70% means 70%. It cannot tell you
whether 70% beats what Kalshi was charging, because until now nothing wrote
the book price down. Being calibrated is necessary. Beating the market's own
probability, after fees, is what makes money. This module lets the logger
record both at the same instant.

SERIES TICKER
-------------
Kalshi groups related markets under a series ticker. The 15-minute BTC series
is configurable because Kalshi renames things:

    set KALSHI_BTC_SERIES=KXBTC15M      (Windows)
    export KALSHI_BTC_SERIES=KXBTC15M   (Mac/Linux)

If the default is wrong the logger will say so and fall back to the synthetic
ladder. Run `python kalshi_book.py discover` to list candidate series.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_SERIES = os.environ.get("KALSHI_BTC_SERIES", "KXBTC15M")


class KalshiError(RuntimeError):
    """Raised when Kalshi market data cannot be read."""


def _get_json(url: str, timeout: int = 10):
    req = urllib.request.Request(
        url, headers={"User-Agent": "btc-scalper/0.1", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise KalshiError(f"kalshi {exc.code} for {url}") from exc
    except Exception as exc:
        raise KalshiError(f"kalshi request failed: {url}") from exc


def _parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _price(market: dict, field: str) -> float | None:
    """
    A price in dollars, 0..1, or None if the book is empty on that side.

    Kalshi returns cents as integers and, on newer payloads, also a
    `<field>_dollars` string. Prefer the dollar form when present; fall back
    to cents. A zero on the bid side means no bid, not a free contract.
    """
    dollars = market.get(f"{field}_dollars")
    if dollars not in (None, ""):
        try:
            v = float(dollars)
            return v if v > 0 else None
        except (TypeError, ValueError):
            pass
    cents = market.get(field)
    if cents in (None, ""):
        return None
    try:
        v = float(cents) / 100.0
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _strike(market: dict) -> tuple[float | None, str | None]:
    """
    (strike, yes_direction) for a threshold market, or (None, None).

    yes_direction says what a YES contract is a bet on:
        "above"  YES pays if price finishes above strike
        "below"  YES pays if price finishes below strike
    Range ("between") markets are skipped - they need two strikes and a
    different probability, and the 15-minute BTC series does not use them.
    """
    st = (market.get("strike_type") or "").lower()
    floor = market.get("floor_strike")
    cap = market.get("cap_strike")
    try:
        if st in ("greater", "greater_or_equal") and floor is not None:
            return float(floor), "above"
        if st in ("less", "less_or_equal") and cap is not None:
            return float(cap), "below"
        # Some payloads omit strike_type but carry exactly one bound.
        if not st:
            if floor is not None and cap is None:
                return float(floor), "above"
            if cap is not None and floor is None:
                return float(cap), "below"
    except (TypeError, ValueError):
        pass
    return None, None


@dataclass(frozen=True)
class Quote:
    """Top of book for one contract, at the moment it was read."""

    ticker: str
    strike: float
    yes_direction: str          # "above" or "below"
    yes_bid: float | None       # dollars 0..1, None if that side is empty
    yes_ask: float | None
    no_bid: float | None
    no_ask: float | None
    last_price: float | None
    volume: int
    open_interest: int
    close_time: datetime | None
    quoted_at: datetime

    @property
    def yes_mid(self) -> float | None:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2
        if self.yes_ask is not None:
            return self.yes_ask
        if self.yes_bid is not None:
            return self.yes_bid
        return None

    @property
    def implied_p_above(self) -> float | None:
        """
        The market's own P(price ends above strike), from the YES mid.

        For a "below" contract, YES pays when price is under the strike, so
        P(above) is one minus the YES mid. Folding this here means the scorer
        never has to think about contract direction.
        """
        m = self.yes_mid
        if m is None:
            return None
        return m if self.yes_direction == "above" else 1 - m

    @property
    def spread(self) -> float | None:
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return None

    def to_dict(self) -> dict:
        r = lambda v: None if v is None else round(v, 4)
        return {
            "ticker": self.ticker,
            "yes_direction": self.yes_direction,
            "yes_bid": r(self.yes_bid),
            "yes_ask": r(self.yes_ask),
            "no_bid": r(self.no_bid),
            "no_ask": r(self.no_ask),
            "last_price": r(self.last_price),
            "implied_p_above": r(self.implied_p_above),
            "volume": self.volume,
            "open_interest": self.open_interest,
            "quoted_at": self.quoted_at.isoformat(),
        }


def quote_from_market(market: dict, quoted_at: datetime | None = None) -> Quote | None:
    strike, direction = _strike(market)
    if strike is None or direction is None:
        return None
    return Quote(
        ticker=str(market.get("ticker", "")),
        strike=strike,
        yes_direction=direction,
        yes_bid=_price(market, "yes_bid"),
        yes_ask=_price(market, "yes_ask"),
        no_bid=_price(market, "no_bid"),
        no_ask=_price(market, "no_ask"),
        last_price=_price(market, "last_price"),
               volume=_count(market, "volume"),
        open_interest=_count(market, "open_interest"),
        close_time=_parse_time(market.get("close_time")),
        quoted_at=quoted_at or datetime.now(timezone.utc),
    )


class KalshiMarketData:
    """
    Public, read-only access to Kalshi markets for one series.

    `fetch` is injectable so tests can hand in canned payloads instead of
    hitting the network.
    """

    def __init__(
        self,
        series: str = DEFAULT_SERIES,
        base: str = BASE,
        fetch=None,
        timeout: int = 10,
    ) -> None:
        self.series = series
        self.base = base.rstrip("/")
        self.timeout = timeout
        self._fetch = fetch or (lambda url: _get_json(url, self.timeout))

    # ---- raw ------------------------------------------------------------

    def markets(self, status: str | None = "open", limit: int = 200,
                series: str | None = None) -> list[dict]:
        """All markets in the series, following pagination cursors."""
        out: list[dict] = []
        cursor = None
        for _ in range(20):  # hard stop; a series never needs this many pages
            params = {"limit": limit, "series_ticker": series or self.series}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            url = f"{self.base}/markets?{urllib.parse.urlencode(params)}"
            data = self._fetch(url)
            out.extend(data.get("markets") or [])
            cursor = data.get("cursor")
            if not cursor:
                break
        return out

    # ---- shaped ---------------------------------------------------------

    def quotes(self, status: str | None = "open") -> list[Quote]:
        now = datetime.now(timezone.utc)
        qs = []
        for m in self.markets(status=status):
            q = quote_from_market(m, now)
            if q:
                qs.append(q)
        return qs

    def quotes_for_window(self, close: datetime, tolerance_s: float = 90) -> list[Quote]:
        """
        Every threshold contract whose close time matches `close`, sorted by
        strike.

        Kalshi's close_time and the logger's window boundary should agree to
        the second, but the tolerance absorbs clock skew and any exchange-side
        rounding. Empty list means no market found for this window - the
        caller should fall back rather than fail.
        """
        if close.tzinfo is None:
            close = close.replace(tzinfo=timezone.utc)
        close = close.astimezone(timezone.utc)
        hits = [
            q for q in self.quotes()
            if q.close_time is not None
            and abs((q.close_time - close).total_seconds()) <= tolerance_s
        ]
        return sorted(hits, key=lambda q: q.strike)

    def discover_series(self, contains: str = "BTC") -> list[str]:
        """
        Best-effort list of series tickers that look like Bitcoin markets.

        Tries the series endpoint first; if that is unavailable, scans open
        markets across all series and collects the distinct tickers.
        """
        found: set[str] = set()
        needle = contains.upper()
        try:
            data = self._fetch(f"{self.base}/series?limit=200")
            for s in data.get("series") or []:
                t = str(s.get("ticker", ""))
                title = str(s.get("title", ""))
                if needle in t.upper() or needle in title.upper():
                    found.add(t)
        except KalshiError:
            pass
        if not found:
            cursor = None
            for _ in range(10):
                params = {"limit": 200, "status": "open"}
                if cursor:
                    params["cursor"] = cursor
                data = self._fetch(f"{self.base}/markets?{urllib.parse.urlencode(params)}")
                for m in data.get("markets") or []:
                    t = str(m.get("series_ticker") or m.get("ticker", ""))
                    title = str(m.get("title", ""))
                    if needle in t.upper() or needle in title.upper():
                        found.add(t.split("-")[0])
                cursor = data.get("cursor")
                if not cursor:
                    break
        return sorted(found)
