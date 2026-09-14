"""
Tests for grading against Kalshi settlement rather than the Coinbase close.

The Coinbase close disagreed with real settlements on 20.6% of windows. If
this layer is wrong the scorer is wrong, so every branch of the outcome
decision is pinned here: settlement wins, direction is honoured, fallback
still works, and the report says which source it used.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import backfill
import score


def pred(wid, ticker, strike, direction="above", schema=3, p=0.6):
    market = None
    if ticker:
        market = {"ticker": ticker, "yes_direction": direction,
                  "implied_p_above": 0.5, "yes_bid": 0.49, "yes_ask": 0.51,
                  "no_ask": 0.51, "no_bid": 0.49, "quoted_at": "z"}
    return {"type": "prediction", "schema": schema, "ladder": "kalshi",
            "window_id": wid, "horizon_min": 4, "at": "x", "window_close": "y",
            "spot": 77000.0, "sigma": 0.001, "base_sigma": 0.001,
            "vol_change_pct": 0, "drift_pct": 0, "suppressed": False,
            "agents": {}, "predictions": [{
                "strike": strike, "p_above": p, "p_yes": p, "sigmas_out": 0.1,
                "market": market}]}


def outcome(wid, close):
    return {"type": "outcome", "window_id": wid, "window_close": "y",
            "close_price": close, "candle_ts": "z", "seconds_off": 0,
            "trustworthy": True}


def settlement(wid, ticker, result):
    return {"type": "settlement", "window_id": wid, "ticker": ticker,
            "result": result, "at": "t"}


def write(lines, path):
    path.write_text("\n".join(json.dumps(l) for l in lines))


class SettlementGrading(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "log.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, lines):
        write(lines, self.path)
        rows, _, _ = score.load(self.path)
        return rows

    # -- the whole point: settlement beats the close --------------------

    def test_settlement_overrides_coinbase_close(self):
        # Coinbase close is ABOVE the strike, but Kalshi settled NO.
        # This is the real 77,185 vs 77,130 case from the log.
        rows = self.load([
            pred("W1", "T1", 77130.0),
            outcome("W1", 77185.46),
            settlement("W1", "T1", "no"),
        ])
        self.assertEqual(rows[0]["hit"], 0)
        self.assertEqual(rows[0]["hit_source"], "kalshi")

    def test_settlement_yes_when_close_says_no(self):
        rows = self.load([
            pred("W1", "T1", 77201.0),
            outcome("W1", 77191.93),
            settlement("W1", "T1", "yes"),
        ])
        self.assertEqual(rows[0]["hit"], 1)
        self.assertEqual(rows[0]["hit_source"], "kalshi")

    # -- direction ----------------------------------------------------------

    def test_below_contract_flips_the_result(self):
        # A "below" contract pays YES when price finishes UNDER the strike.
        # hit tracks P(above), so a YES settlement means hit = 0.
        rows = self.load([
            pred("W1", "T1", 77000.0, direction="below"),
            outcome("W1", 76900.0),
            settlement("W1", "T1", "yes"),
        ])
        self.assertEqual(rows[0]["hit"], 0)

    def test_below_contract_no_means_above(self):
        rows = self.load([
            pred("W1", "T1", 77000.0, direction="below"),
            outcome("W1", 77100.0),
            settlement("W1", "T1", "no"),
        ])
        self.assertEqual(rows[0]["hit"], 1)

    # -- fallback -----------------------------------------------------------

    def test_no_settlement_falls_back_to_close(self):
        rows = self.load([
            pred("W1", "T1", 77000.0),
            outcome("W1", 77050.0),
        ])
        self.assertEqual(rows[0]["hit"], 1)
        self.assertEqual(rows[0]["hit_source"], "close")

    def test_synthetic_ladder_has_no_ticker_and_uses_close(self):
        rows = self.load([
            pred("W1", None, 77000.0),
            outcome("W1", 76950.0),
            settlement("W1", "SOMETHING", "yes"),   # cannot apply, no ticker
        ])
        self.assertEqual(rows[0]["hit"], 0)
        self.assertEqual(rows[0]["hit_source"], "close")

    def test_settlement_for_other_ticker_does_not_leak(self):
        rows = self.load([
            pred("W1", "T1", 77000.0),
            outcome("W1", 77050.0),
            settlement("W2", "T2", "no"),
        ])
        self.assertEqual(rows[0]["hit"], 1)
        self.assertEqual(rows[0]["hit_source"], "close")

    def test_malformed_settlement_is_ignored(self):
        rows = self.load([
            pred("W1", "T1", 77000.0),
            outcome("W1", 77050.0),
            {"type": "settlement", "window_id": "W1", "ticker": "T1",
             "result": "maybe"},
        ])
        self.assertEqual(rows[0]["hit_source"], "close")

    def test_mixed_sources_are_both_counted(self):
        rows = self.load([
            pred("W1", "T1", 77000.0), outcome("W1", 77050.0),
            settlement("W1", "T1", "yes"),
            pred("W2", "T2", 77000.0), outcome("W2", 77050.0),
        ])
        sources = sorted(r["hit_source"] for r in rows)
        self.assertEqual(sources, ["close", "kalshi"])

    # -- the settlement line can arrive after the prediction ----------------

    def test_settlement_line_order_does_not_matter(self):
        rows = self.load([
            settlement("W1", "T1", "no"),          # first in file
            pred("W1", "T1", 77000.0),
            outcome("W1", 77050.0),
        ])
        self.assertEqual(rows[0]["hit"], 0)
        self.assertEqual(rows[0]["hit_source"], "kalshi")


class BackfillScan(unittest.TestCase):

    def test_scan_finds_missing_tickers_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log.jsonl"
            write([
                pred("W1", "T1", 1.0), outcome("W1", 1.0),
                pred("W2", "T2", 1.0), outcome("W2", 1.0),
                settlement("W1", "T1", "yes"),
                pred("W3", None, 1.0), outcome("W3", 1.0),   # synthetic, no ticker
            ], path)
            wanted, have = backfill.scan(path)
            self.assertEqual(set(wanted), {"T1", "T2"})
            self.assertEqual(wanted["T2"], "W2")
            self.assertEqual(have, {"T1"})

    def test_scan_empty_log(self):
        with tempfile.TemporaryDirectory() as d:
            wanted, have = backfill.scan(Path(d) / "nope.jsonl")
            self.assertEqual(wanted, {})
            self.assertEqual(have, set())


if __name__ == "__main__":
    unittest.main()
