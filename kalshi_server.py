"""
Dashboard server for Kalshi BTC contracts.

    python kalshi_server.py
    python kalshi_server.py --multiplier 0.10 --port 8080

Then open http://localhost:8000

Two feeds are kept. One-minute candles drive the live volatility estimate and
refresh constantly. Hourly candles, covering about twelve days, are fetched
once and refreshed every half hour - they exist so the time-of-day agent can
measure Bitcoin's daily rhythm rather than assume one.

The endpoint returns per-minute volatility rather than finished probabilities.
The browser scales it to whatever time is left and redraws every second, so
fair value decays smoothly instead of jumping on each poll.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from core.adjusters import Context, build_estimate
from core.feed import CoinbaseFeed, FeedError
from core.kalshi import KalshiFees, estimate_vol

HERE = Path(__file__).parent
CONFIG: dict = {}


def next_window_close(now: datetime | None = None, minutes: int = 15) -> datetime:
    """The next quarter-hour boundary, where 15-minute windows end."""
    now = now or datetime.now(timezone.utc)
    elapsed = (now.minute % minutes) * 60 + now.second + now.microsecond / 1e6
    return now + timedelta(seconds=minutes * 60 - elapsed)


class MarketCache:
    """
    Shares one set of exchange calls across requests and browser tabs.

    The hourly series has its own long TTL. Refetching twelve days of history
    every fifteen seconds would be pointless and rude to the exchange.
    """

    def __init__(self, minute_ttl: int = 15, hourly_ttl: int = 1800):
        self.minute_ttl = minute_ttl
        self.hourly_ttl = hourly_ttl
        self.feed = CoinbaseFeed()
        self._lock = threading.Lock()
        self._minute = None
        self._minute_at = 0.0
        self._hourly = None
        self._hourly_at = 0.0
        self._error = None

    def _refresh_hourly(self):
        if self._hourly and time.time() - self._hourly_at < self.hourly_ttl:
            return
        try:
            self._hourly = self.feed.candles(granularity=3600, limit=300)
            self._hourly_at = time.time()
        except FeedError:
            pass  # non-fatal; the time-of-day agent simply stays idle

    def get(self):
        with self._lock:
            if self._minute and time.time() - self._minute_at < self.minute_ttl:
                return self._minute, self._hourly, self._error
            try:
                candles = self.feed.candles(granularity=60, limit=120)
                if len(candles) < 30:
                    raise FeedError("too few candles to estimate volatility")
                self._minute = candles
                self._minute_at = time.time()
                self._error = None
                self._refresh_hourly()
            except FeedError as exc:
                self._error = str(exc)
            return self._minute, self._hourly, self._error


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            self._page("kalshi_dashboard.html")
            return
        if path == "/advanced":
            self._page("kalshi_ladder_advanced.html")
            return
        if path == "/api/market":
            self._send(*self._market())
            return
        self._send(404, b"not found", "text/plain")

    def _page(self, name):
        page = HERE / name
        if not page.exists():
            self._send(404, f"{name} is missing".encode(), "text/plain")
            return
        self._send(200, page.read_bytes(), "text/html; charset=utf-8")

    def _market(self):
        cache: MarketCache = CONFIG["cache"]
        fees: KalshiFees = CONFIG["fees"]
        candles, hourly, error = cache.get()

        if candles is None:
            body = json.dumps({"ok": False, "error": error or "no data"})
            return 503, body.encode(), "application/json"

        now = datetime.now(timezone.utc)
        spot = candles[-1].close
        vol = estimate_vol(candles, 1.0)
        minutes_left = max((next_window_close(now) - now).total_seconds() / 60, 0.1)

        ctx = Context(
            spot=spot,
            candles=candles,
            vol=vol,
            minutes_left=minutes_left,
            now=now,
            hourly=hourly,
        )
        est = build_estimate(ctx)

        # Send per-minute sigma so the browser can rescale as the clock runs.
        base_per_min = vol.blended or 0.0
        adjusted_per_min = (
            base_per_min * (est.final_sigma / est.base_sigma)
            if est.base_sigma
            else base_per_min
        )

        payload = {
            "ok": True,
            "stale": error is not None,
            "spot": round(spot, 2),
            "sigma_per_min": adjusted_per_min,
            "base_sigma_per_min": base_per_min,
            "vol_change_pct": est.total_vol_change_pct,
            "drift_pct": est.drift_pct,
            "suppressed": est.suppressed,
            "disagreement": vol.disagreement,
            "window_close": next_window_close(now).isoformat(),
            "server_time": now.isoformat(),
            "taker_multiplier": fees.taker_multiplier,
            "maker_multiplier": fees.maker_multiplier,
            "hourly_loaded": bool(hourly),
            "agents": [a.to_dict() for a in est.adjustments],
            "recent": [round(c.close, 2) for c in candles[-40:]],
        }
        return 200, json.dumps(payload).encode(), "application/json"


def main() -> int:
    p = argparse.ArgumentParser(description="Kalshi BTC dashboard")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--multiplier", type=float, default=0.07,
                   help="Kalshi taker fee multiplier, check your order ticket")
    args = p.parse_args()

    CONFIG["cache"] = MarketCache()
    CONFIG["fees"] = KalshiFees(taker_multiplier=args.multiplier)

    print(f"  Dashboard    http://localhost:{args.port}")
    print(f"  Advanced     http://localhost:{args.port}/advanced")
    print(f"  Fee model    taker {args.multiplier}, maker 0.0175")
    print("  Loading history for the time-of-day agent on first request.")
    print("  Ctrl-C to stop\n")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
