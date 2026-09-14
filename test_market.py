"""
Tests for the market-price path: Kalshi client, logger ladder, scorer.

Everything here is offline. The Kalshi client takes an injectable fetch, the
logger takes an injectable feed and market, and the scorer reads a temp file.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import score
from core.feed import Candle
from core.kalshi import prob_above
from core.kalshi_api import (
    KalshiError,
    KalshiMarketData,
    Quote,
    _price,
    _strike,
    quote_from_market,
)
from logger import Recorder, window_close, window_id

UTC = timezone.utc


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def market(ticker, strike, close, yes_bid=45, yes_ask=47, strike_type="greater",
           **extra):
    m = {
        "ticker": ticker,
        "series_ticker": "KXBTC15M",
        "strike_type": strike_type,
        "close_time": close.isoformat().replace("+00:00", "Z"),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": 100 - yes_ask if yes_ask else None,
        "no_ask": 100 - yes_bid if yes_bid else None,
        "last_price": yes_bid,
        "volume": 120,
        "open_interest": 300,
    }
    if strike_type in ("greater", "greater_or_equal"):
        m["floor_strike"] = strike
    elif strike_type in ("less", "less_or_equal"):
        m["cap_strike"] = strike
    else:
        m["floor_strike"] = strike
        m["cap_strike"] = strike + 500
    m.update(extra)
    return m


def canned(pages):
    """fetch() that serves pages in order and records the URLs it saw."""
    calls = []

    def fetch(url):
        calls.append(url)
        if not pages:
            raise KalshiError("out of pages")
        return pages.pop(0)

    fetch.calls = calls
    return fetch


class FakeFeed:
    """Enough of CoinbaseFeed for the logger: flat-ish 1-minute candles."""

    def __init__(self, spot=100_000.0, n=120):
        self.spot = spot
        self.n = n

    def candles(self, granularity=60, limit=120):
        random.seed(7)
        now = datetime(2026, 9, 11, 3, 0, tzinfo=UTC)
        out = []
        px = self.spot
        for i in range(limit):
            ts = now - timedelta(seconds=granularity * (limit - i))
            move = px * random.uniform(-0.0004, 0.0004)
            o, c = px, px + move
            out.append(Candle(ts, o, max(o, c) + 5, min(o, c) - 5, c, 10.0))
            px = c
        return out


# --------------------------------------------------------------------------
# client: parsing
# --------------------------------------------------------------------------


class PriceParsing(unittest.TestCase):
    def test_cents_to_dollars(self):
        self.assertAlmostEqual(_price({"yes_ask": 47}, "yes_ask"), 0.47)

    def test_dollar_string_preferred(self):
        self.assertAlmostEqual(
            _price({"yes_ask": 47, "yes_ask_dollars": "0.4800"}, "yes_ask"), 0.48
        )

    def test_zero_is_empty_side(self):
        self.assertIsNone(_price({"yes_bid": 0}, "yes_bid"))

    def test_missing_is_none(self):
        self.assertIsNone(_price({}, "yes_bid"))


class StrikeParsing(unittest.TestCase):
    def test_greater_uses_floor(self):
        self.assertEqual(_strike({"strike_type": "greater", "floor_strike": 100500}),
                         (100500.0, "above"))

    def test_less_uses_cap(self):
        self.assertEqual(_strike({"strike_type": "less", "cap_strike": 99500}),
                         (99500.0, "below"))

    def test_between_is_skipped(self):
        self.assertEqual(
            _strike({"strike_type": "between", "floor_strike": 1, "cap_strike": 2}),
            (None, None),
        )

    def test_missing_type_with_one_bound(self):
        self.assertEqual(_strike({"floor_strike": 42}), (42.0, "above"))
        self.assertEqual(_strike({"cap_strike": 42}), (42.0, "below"))


class QuoteShape(unittest.TestCase):
    def test_implied_folds_direction(self):
        c = datetime(2026, 9, 11, 3, 15, tzinfo=UTC)
        above = quote_from_market(market("A", 100000, c, 30, 34, "greater"))
        below = quote_from_market(market("B", 100000, c, 30, 34, "less"))
        self.assertAlmostEqual(above.implied_p_above, 0.32)
        self.assertAlmostEqual(below.implied_p_above, 0.68)

    def test_one_sided_book_uses_that_side(self):
        c = datetime(2026, 9, 11, 3, 15, tzinfo=UTC)
        q = quote_from_market(market("A", 100000, c, yes_bid=0, yes_ask=90))
        self.assertIsNone(q.yes_bid)
        self.assertAlmostEqual(q.yes_mid, 0.90)

    def test_to_dict_roundtrips_the_fields_score_needs(self):
        c = datetime(2026, 9, 11, 3, 15, tzinfo=UTC)
        d = quote_from_market(market("A", 100000, c)).to_dict()
        for k in ("ticker", "yes_direction", "yes_bid", "yes_ask", "no_ask",
                  "implied_p_above", "quoted_at"):
            self.assertIn(k, d)


# --------------------------------------------------------------------------
# client: fetching
# --------------------------------------------------------------------------


class Fetching(unittest.TestCase):
    def setUp(self):
        self.close = datetime(2026, 9, 11, 3, 15, tzinfo=UTC)
        self.other = self.close + timedelta(minutes=15)

    def test_follows_cursor(self):
        fetch = canned([
            {"markets": [market("A", 99000, self.close)], "cursor": "p2"},
            {"markets": [market("B", 100000, self.close)], "cursor": ""},
        ])
        md = KalshiMarketData(fetch=fetch)
        self.assertEqual(len(md.markets()), 2)
        self.assertEqual(len(fetch.calls), 2)
        self.assertIn("cursor=p2", fetch.calls[1])
        self.assertIn("series_ticker=KXBTC15M", fetch.calls[0])

    def test_window_filter_and_sort(self):
        fetch = canned([{
            "markets": [
                market("C", 101000, self.close),
                market("X", 100000, self.other),      # wrong window
                market("A", 99000, self.close),
                market("R", 99500, self.close, strike_type="between"),  # skipped
            ],
        }])
        qs = KalshiMarketData(fetch=fetch).quotes_for_window(self.close)
        self.assertEqual([q.ticker for q in qs], ["A", "C"])

    def test_tolerance_absorbs_skew(self):
        skewed = self.close + timedelta(seconds=40)
        fetch = canned([{"markets": [market("A", 99000, skewed)]}])
        self.assertEqual(len(KalshiMarketData(fetch=fetch).quotes_for_window(self.close)), 1)
        fetch = canned([{"markets": [market("A", 99000, skewed)]}])
        self.assertEqual(
            len(KalshiMarketData(fetch=fetch).quotes_for_window(self.close, tolerance_s=10)), 0
        )

    def test_naive_close_treated_as_utc(self):
        fetch = canned([{"markets": [market("A", 99000, self.close)]}])
        naive = self.close.replace(tzinfo=None)
        self.assertEqual(len(KalshiMarketData(fetch=fetch).quotes_for_window(naive)), 1)

    def test_network_error_is_kalshi_error(self):
        md = KalshiMarketData(fetch=canned([]))
        with self.assertRaises(KalshiError):
            md.quotes()

    def test_discover_falls_back_to_market_scan(self):
        fetch = canned([
            None,  # /series raises via out-of-pages? no - give it an error page
        ])

        def fetch(url):
            if "/series" in url:
                raise KalshiError("no series endpoint")
            return {"markets": [
                {"ticker": "KXBTC15M-26SEP110315-T99000", "series_ticker": "KXBTC15M",
                 "title": "Bitcoin above 99,000?"},
                {"ticker": "KXETH-1", "series_ticker": "KXETH", "title": "Ethereum"},
            ]}

        found = KalshiMarketData(fetch=fetch).discover_series("BTC")
        self.assertEqual(found, ["KXBTC15M"])


# --------------------------------------------------------------------------
# logger: which ladder gets written
# --------------------------------------------------------------------------


class CannedMarket:
    def __init__(self, quotes=None, error=None):
        self.series = "KXBTC15M"
        self._quotes = quotes or []
        self._error = error

    def quotes_for_window(self, close, tolerance_s=90):
        if self._error:
            raise self._error
        return self._quotes


def quotes_around(spot, close, n=5, step=250):
    out = []
    for i in range(-(n // 2), n // 2 + 1):
        k = spot + i * step
        out.append(quote_from_market(market(f"T{k:.0f}", k, close, 40, 44)))
    return out


class LoggerLadder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "p.jsonl"
        self.now = datetime(2026, 9, 11, 3, 3, tzinfo=UTC)
        self.close = window_close(self.now)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, market):
        rec = Recorder(self.path, horizons=(12,), strikes=9,
                       market=market, use_market=market is not None)
        rec.feed = FakeFeed(100_000.0)
        return rec

    def read(self):
        return [json.loads(l) for l in self.path.read_text().splitlines()]

    def test_kalshi_ladder_when_book_available(self):
        rec = self.make(CannedMarket(quotes_around(100_000, self.close)))
        out = rec.snapshot(self.close, 12, now=self.now)
        self.assertIsNotNone(out)
        self.assertEqual(out["schema"], 3)
        self.assertEqual(out["ladder"], "kalshi")
        self.assertEqual(len(out["predictions"]), 5)
        for item in out["predictions"]:
            self.assertIsNotNone(item["market"])
            self.assertIn("implied_p_above", item["market"])
            self.assertIn("p_yes", item)
            self.assertIn("sigmas_out", item)
        # strikes are the exchange's, not rounded-to-$10 synthetics
        self.assertEqual([i["strike"] for i in out["predictions"]],
                         [99500.0, 99750.0, 100000.0, 100250.0, 100500.0])

    def test_p_yes_follows_contract_direction(self):
        c = self.close
        below = quote_from_market(market("B", 100_000, c, 40, 44, "less"))
        rec = self.make(CannedMarket([below]))
        out = rec.snapshot(c, 12, now=self.now)
        item = out["predictions"][0]
        self.assertAlmostEqual(item["p_yes"], 1 - item["p_above"], places=5)

    def test_synthetic_when_market_errors(self):
        rec = self.make(CannedMarket(error=KalshiError("down")))
        out = rec.snapshot(self.close, 12, now=self.now)
        self.assertEqual(out["ladder"], "synthetic")
        self.assertEqual(len(out["predictions"]), 9)
        self.assertTrue(all(i["market"] is None for i in out["predictions"]))
        self.assertTrue(rec._market_warned)

    def test_synthetic_when_no_contracts(self):
        rec = self.make(CannedMarket([]))
        out = rec.snapshot(self.close, 12, now=self.now)
        self.assertEqual(out["ladder"], "synthetic")

    def test_no_market_at_all(self):
        rec = self.make(None)
        out = rec.snapshot(self.close, 12, now=self.now)
        self.assertEqual(out["ladder"], "synthetic")
        self.assertIsNone(out["market_series"])

    def test_synthetic_ladder_unchanged_from_before(self):
        """The fallback must produce exactly what the old logger produced."""
        rec = self.make(None)
        out = rec.snapshot(self.close, 12, now=self.now)
        spot, sigma, drift = out["spot"], out["sigma"], out["drift_pct"]
        step = spot * sigma * 0.5
        expect = [round((spot + i * step) / 10) * 10 for i in range(-4, 5)]
        self.assertEqual([i["strike"] for i in out["predictions"]], expect)
        for i in out["predictions"]:
            p = prob_above(spot * (1 + drift / 100), i["strike"], sigma)
            self.assertAlmostEqual(i["p_above"], p, places=4)


# --------------------------------------------------------------------------
# scorer: reads both schemas, measures against the market
# --------------------------------------------------------------------------


def write_log(path, windows=30, per_reading=5, with_market=True, schema=2,
              market_noise=0.0, seed=3):
    """
    A synthetic log where the market is (optionally) a perfect predictor
    and the model is a noisy one. Enough to check the plumbing.
    """
    rng = random.Random(seed)
    with path.open("w") as fh:
        for w in range(windows):
            close_t = datetime(2026, 9, 11, 0, 0, tzinfo=UTC) + timedelta(minutes=15 * (w + 1))
            wid = window_id(close_t)
            spot = 100_000.0 + rng.uniform(-300, 300)
            close_px = spot + rng.gauss(0, 150)
            sigma = 0.0015
            for h in (12, 8, 4):
                preds = []
                for i in range(-(per_reading // 2), per_reading // 2 + 1):
                    k = round(spot + i * 100, -1)
                    truth = 1.0 if close_px > k else 0.0
                    p = min(max(prob_above(spot, k, sigma) + rng.gauss(0, 0.15), 0.01), 0.99)
                    item = {"strike": k, "p_above": round(p, 6), "sigmas_out": 0.3}
                    if schema == 2:
                        item["p_yes"] = item["p_above"]
                        if with_market:
                            mp = min(max(truth * 0.8 + 0.1 + rng.gauss(0, market_noise), 0.01), 0.99)
                            item["market"] = {
                                "ticker": f"T{k:.0f}", "yes_direction": "above",
                                "yes_bid": round(mp - 0.01, 4), "yes_ask": round(mp + 0.01, 4),
                                "no_bid": None, "no_ask": round(1 - mp + 0.01, 4),
                                "implied_p_above": round(mp, 4),
                                "quoted_at": close_t.isoformat(),
                            }
                        else:
                            item["market"] = None
                    preds.append(item)
                rec = {
                    "type": "prediction", "window_id": wid, "horizon_min": h,
                    "spot": spot, "sigma": sigma, "drift_pct": 0.02,
                    "suppressed": False, "predictions": preds,
                }
                if schema == 2:
                    rec["schema"] = 2
                    rec["ladder"] = "kalshi" if with_market else "synthetic"
                fh.write(json.dumps(rec) + "\n")
            fh.write(json.dumps({
                "type": "outcome", "window_id": wid, "close_price": close_px,
                "trustworthy": True,
            }) + "\n")


class Scorer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "p.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_old_schema_still_loads(self):
        write_log(self.path, schema=1)
        rows, n, unres = score.load(self.path)
        self.assertEqual(n, 90)
        self.assertEqual(unres, 0)
        self.assertTrue(all(r["mkt_p"] is None for r in rows))
        self.assertTrue(all(r["ladder"] == "synthetic" for r in rows))
        for k in ("window_id", "horizon", "suppressed", "p", "sigmas", "hit",
                  "spot", "strike", "close"):
            self.assertIn(k, rows[0])

    def test_new_schema_carries_market(self):
        write_log(self.path)
        rows, _, _ = score.load(self.path)
        self.assertTrue(all(r["mkt_p"] is not None for r in rows))
        self.assertTrue(all(r["yes_ask"] is not None for r in rows))

    def test_readings_fewer_than_calls(self):
        write_log(self.path, windows=20, per_reading=5)
        rows, _, _ = score.load(self.path)
        self.assertEqual(len(rows), 20 * 3 * 5)
        self.assertEqual(score.n_readings(rows), 20 * 3)
        for c in score.calibration(rows):
            self.assertLessEqual(c["readings"], c["n"])

    def test_clustered_se_is_wider(self):
        write_log(self.path, windows=20, per_reading=9)
        rows, _, _ = score.load(self.path)
        for c in score.calibration(rows):
            naive = math.sqrt(max(c["actual"] * (1 - c["actual"]), 1e-9) / c["n"])
            self.assertGreaterEqual(c["se"], naive)

    def test_market_brier_beats_noisy_model(self):
        write_log(self.path, windows=40)
        rows, _, _ = score.load(self.path)
        self.assertLess(score.brier(rows, "mkt_p"), score.brier(rows, "p"))

    def test_zero_drift_reprices(self):
        write_log(self.path, windows=5)
        base, _, _ = score.load(self.path)
        nod, _, _ = score.load(self.path, zero_drift=True)
        for r in nod:
            self.assertAlmostEqual(r["p"], prob_above(r["spot"], r["strike"], r["sigma"]), places=6)
        # and it actually changed something versus the logged (noisy) p
        self.assertTrue(any(abs(a["p"] - b["p"]) > 1e-6 for a, b in zip(base, nod)))

    def test_report_runs_with_and_without_market(self):
        import contextlib, io
        for with_market in (True, False):
            write_log(self.path, with_market=with_market)
            rows, n, u = score.load(self.path)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                score.report(rows, n, u)
            text = buf.getvalue()
            self.assertIn("Skill vs 50%", text)
            self.assertIn("Skill vs market", text)
            self.assertIn("By distance from the money", text)
            if with_market:
                self.assertIn("Market-mid Brier", text)
            else:
                self.assertIn("no Kalshi prices", text)


if __name__ == "__main__":
    unittest.main()
