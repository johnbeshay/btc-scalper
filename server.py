"""
Local server. One command, then open a browser.

    python3 server.py
    python3 server.py --port 8080 --taker 0.25
    python3 server.py --mock            # no exchange needed, synthetic data

Serves the dashboard at / and the verdict JSON at /api/verdict.

Candles are cached so that refreshing the browser, or having three tabs
open, does not mean three calls to the exchange. The cache TTL is deliberately
shorter than the candle interval.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from core.feed import BinanceFeed, CoinbaseFeed, FeedError
from core.indicators import FeeModel
from core.orchestrator import Orchestrator

HERE = Path(__file__).parent
CONFIG: dict = {}


class CandleCache:
    """Thread-safe cache so concurrent requests share one exchange call."""

    def __init__(self, feed, granularity: int, ttl: int = 20):
        self.feed = feed
        self.granularity = granularity
        self.ttl = ttl
        self._lock = threading.Lock()
        self._candles = None
        self._fetched_at = 0.0
        self._error = None

    def get(self):
        with self._lock:
            age = time.time() - self._fetched_at
            if self._candles is not None and age < self.ttl:
                return self._candles, self._error

            try:
                self._candles = self.feed.candles(
                    granularity=self.granularity, limit=200
                )
                self._fetched_at = time.time()
                self._error = None
            except FeedError as exc:
                self._error = str(exc)

            return self._candles, self._error


class MockFeed:
    """Synthetic candles for testing the UI without an exchange."""

    def __init__(self):
        self._t = 0

    def candles(self, granularity=300, limit=200):
        import math
        import random
        from datetime import datetime, timedelta, timezone

        from core.feed import Candle

        random.seed(42)
        self._t += 1
        out, price = [], 43000.0
        now = datetime.now(timezone.utc)

        for i in range(limit):
            wave = math.sin((i + self._t) / 14) * 0.004
            price *= 1 + wave / 6 + random.gauss(0, 0.0018)
            o = price
            c = price * (1 + random.gauss(0, 0.0015))
            out.append(
                Candle(
                    ts=now - timedelta(minutes=5 * (limit - i)),
                    open=o,
                    high=max(o, c) * 1.0008,
                    low=min(o, c) * 0.9992,
                    close=c,
                    volume=abs(random.gauss(140, 50)) * (3 if i == limit - 1 else 1),
                )
            )
            price = c
        return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet; the console is for verdicts, not access logs

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            page = HERE / "dashboard.html"
            if not page.exists():
                self._send(404, b"dashboard.html is missing", "text/plain")
                return
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/verdict":
            self._send(*self._verdict())
            return

        self._send(404, b"not found", "text/plain")

    def _verdict(self):
        cache: CandleCache = CONFIG["cache"]
        orch: Orchestrator = CONFIG["orchestrator"]
        fees: FeeModel = CONFIG["fees"]

        candles, error = cache.get()

        if candles is None:
            payload = {
                "ok": False,
                "error": error or "no data yet",
            }
            return 503, json.dumps(payload).encode(), "application/json"

        verdict = orch.run(candles)
        price = candles[-1].close

        payload = verdict.to_dict()
        payload.update(
            {
                "ok": True,
                "stale": error is not None,
                "price": round(price, 2),
                "round_trip_cost_pct": round(fees.round_trip_pct, 3),
                "breakeven_long": round(fees.breakeven_price(price, "long"), 2),
                "breakeven_short": round(fees.breakeven_price(price, "short"), 2),
                "candles": [
                    {
                        "t": c.ts.isoformat(),
                        "o": round(c.open, 2),
                        "h": round(c.high, 2),
                        "l": round(c.low, 2),
                        "c": round(c.close, 2),
                        "v": round(c.volume, 2),
                    }
                    for c in candles[-60:]
                ],
            }
        )
        return 200, json.dumps(payload).encode(), "application/json"


def main() -> int:
    p = argparse.ArgumentParser(description="BTC scalper dashboard server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--exchange", choices=["coinbase", "binance"], default="coinbase")
    p.add_argument("--granularity", type=int, default=300)
    p.add_argument("--taker", type=float, default=0.60)
    p.add_argument("--slippage", type=float, default=0.05)
    p.add_argument("--mock", action="store_true", help="synthetic data, no exchange")
    args = p.parse_args()

    if args.mock:
        feed = MockFeed()
    else:
        feed = CoinbaseFeed() if args.exchange == "coinbase" else BinanceFeed()

    fees = FeeModel(args.taker, args.taker, args.slippage)

    CONFIG["cache"] = CandleCache(feed, args.granularity)
    CONFIG["orchestrator"] = Orchestrator(fees=fees)
    CONFIG["fees"] = fees

    source = "mock data" if args.mock else args.exchange
    print(f"  Dashboard   http://localhost:{args.port}")
    print(f"  Source      {source}, {args.granularity // 60}m candles")
    print(f"  Round trip  {fees.round_trip_pct:.2f}%  (taker {args.taker}% per side)")
    print("  Ctrl-C to stop\n")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
