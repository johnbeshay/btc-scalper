"""
Candle data models and exchange feeds.

Two feeds are provided. Coinbase is the default because Binance.com is
geo-blocked in the US. Neither endpoint needs an API key - these are the
public market-data routes.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Candle:
    """A single OHLCV bar."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def is_green(self) -> bool:
        return self.close >= self.open

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def pct_change(self) -> float:
        if self.open == 0:
            return 0.0
        return (self.close - self.open) / self.open * 100


class FeedError(RuntimeError):
    """Raised when an exchange feed cannot be read."""


def _get_json(url: str, timeout: int = 10):
    req = urllib.request.Request(url, headers={"User-Agent": "btc-scalper/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        raise FeedError(f"feed request failed: {url}") from exc


class CoinbaseFeed:
    """
    Public Coinbase Exchange market data.

    granularity is in seconds: 60, 300, 900, 3600, 21600, 86400.
    Coinbase caps a single response at 300 candles.
    """

    BASE = "https://api.exchange.coinbase.com"

    def __init__(self, product: str = "BTC-USD") -> None:
        self.product = product

    def candles(self, granularity: int = 300, limit: int = 200) -> list[Candle]:
        url = f"{self.BASE}/products/{self.product}/candles?granularity={granularity}"
        raw = _get_json(url)

        # Coinbase returns [time, low, high, open, close, volume], newest first.
        out = [
            Candle(
                ts=datetime.fromtimestamp(row[0], tz=timezone.utc),
                low=float(row[1]),
                high=float(row[2]),
                open=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in raw
        ]
        out.sort(key=lambda c: c.ts)
        return out[-limit:]

    def spot_price(self) -> float:
        url = f"{self.BASE}/products/{self.product}/ticker"
        return float(_get_json(url)["price"])


class BinanceFeed:
    """Public Binance klines. Fallback for non-US users."""

    BASE = "https://api.binance.com"

    def __init__(self, symbol: str = "BTCUSDT") -> None:
        self.symbol = symbol

    def candles(self, granularity: int = 300, limit: int = 200) -> list[Candle]:
        interval = {60: "1m", 300: "5m", 900: "15m", 3600: "1h"}.get(granularity)
        if interval is None:
            raise FeedError(f"unsupported granularity for Binance: {granularity}")

        url = (
            f"{self.BASE}/api/v3/klines?symbol={self.symbol}"
            f"&interval={interval}&limit={limit}"
        )
        raw = _get_json(url)

        return [
            Candle(
                ts=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in raw
        ]

    def spot_price(self) -> float:
        url = f"{self.BASE}/api/v3/ticker/price?symbol={self.symbol}"
        return float(_get_json(url)["price"])
