"""
Tests for replay.py.

Three synthetic worlds:
  oracle   the model knows the outcome; the book does not. Should profit.
  noise    the model is a coin flip; the book is fair. Should lose to fees.
  book     the book knows the outcome; the model does not. Should lose.
"""

from __future__ import annotations

import contextlib
import io
import json
import random
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import replay
import score
from core.kalshi import KalshiFees
from logger import window_id

UTC = timezone.utc


def write_world(path, kind, windows=60, seed=1):
    rng = random.Random(seed)
    with path.open("w") as fh:
        for w in range(windows):
            close_t = datetime(2026, 9, 12, 0, 0, tzinfo=UTC) + timedelta(minutes=15 * (w + 1))
            wid = window_id(close_t)
            spot = 100_000.0
            k = 100_000.0
            truth = 1 if rng.random() < 0.5 else 0
            close_px = k + 100 if truth else k - 100

            if kind == "oracle":
                p_model = 0.92 if truth else 0.08
                p_book = 0.50
            elif kind == "book":
                p_model = 0.50 + rng.uniform(-0.05, 0.05)
                p_book = 0.90 if truth else 0.10
            else:  # noise
                p_model = rng.uniform(0.35, 0.65)
                p_book = 0.50

            for h in (12, 8, 4):
                item = {
                    "strike": k, "p_above": p_model, "p_yes": p_model,
                    "sigmas_out": 0.1,
                    "market": {
                        "ticker": f"T{w}", "yes_direction": "above",
                        "yes_bid": round(p_book - 0.01, 4),
                        "yes_ask": round(p_book + 0.01, 4),
                        "no_bid": round(1 - p_book - 0.01, 4),
                        "no_ask": round(1 - p_book + 0.01, 4),
                        "implied_p_above": p_book,
                        "quoted_at": close_t.isoformat(),
                    },
                }
                fh.write(json.dumps({
                    "type": "prediction", "schema": 2, "ladder": "kalshi",
                    "window_id": wid, "horizon_min": h, "spot": spot,
                    "sigma": 0.0015, "drift_pct": 0.0, "suppressed": False,
                    "predictions": [item],
                }) + "\n")
            fh.write(json.dumps({
                "type": "outcome", "window_id": wid, "close_price": close_px,
                "trustworthy": True,
            }) + "\n")


class Worlds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "p.jsonl"
        self.fees = KalshiFees()

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self, kind):
        write_world(self.path, kind)
        rows, _, _ = score.load(self.path)
        return rows

    def test_oracle_profits(self):
        rows = self.rows("oracle")
        s = replay.summarise(replay.simulate(rows, 0.10, "ask", 10, self.fees))
        self.assertEqual(s.n, 60)           # one per window
        self.assertEqual(s.win_rate, 1.0)
        self.assertGreater(s.net, 0)
        self.assertGreater(s.fees, 0)

    def test_noise_loses_to_fees(self):
        rows = self.rows("noise")
        s = replay.summarise(replay.simulate(rows, 0.0, "ask", 10, self.fees))
        self.assertGreater(s.n, 0)
        # At-the-money coin flips with a 2c spread and taker fee: negative.
        self.assertLess(s.net, 0)

    def test_book_oracle_means_model_loses(self):
        rows = self.rows("book")
        # Model says ~50%, book says 90/10. Model sees "edge" on the cheap
        # side every time and is wrong every time.
        s = replay.summarise(replay.simulate(rows, 0.10, "ask", 10, self.fees))
        self.assertGreater(s.n, 0)
        self.assertLess(s.win_rate, 0.2)
        self.assertLess(s.net, 0)
        self.assertLess(s.realised, s.claimed)   # inflated edge


class Mechanics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "p.jsonl"
        self.fees = KalshiFees()

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_trade_per_window(self):
        write_world(self.path, "oracle")
        rows, _, _ = score.load(self.path)
        trades = replay.simulate(rows, 0.0, "ask", 10, self.fees)
        self.assertEqual(len({t.window_id for t in trades}), len(trades))

    def test_earliest_horizon_taken_first(self):
        write_world(self.path, "oracle")
        rows, _, _ = score.load(self.path)
        trades = replay.simulate(rows, 0.0, "ask", 10, self.fees)
        self.assertTrue(all(t.horizon == 12 for t in trades))

    def test_horizon_filter(self):
        write_world(self.path, "oracle")
        rows, _, _ = score.load(self.path)
        trades = replay.simulate(rows, 0.0, "ask", 10, self.fees, horizon=4)
        self.assertTrue(all(t.horizon == 4 for t in trades))

    def test_threshold_filters_trades(self):
        write_world(self.path, "noise")
        rows, _, _ = score.load(self.path)
        loose = replay.simulate(rows, 0.0, "ask", 10, self.fees)
        tight = replay.simulate(rows, 0.10, "ask", 10, self.fees)
        self.assertGreater(len(loose), len(tight))

    def test_mid_cheaper_than_ask(self):
        write_world(self.path, "oracle")
        rows, _, _ = score.load(self.path)
        ask = replay.summarise(replay.simulate(rows, 0.0, "ask", 10, self.fees))
        mid = replay.summarise(replay.simulate(rows, 0.0, "mid", 10, self.fees))
        self.assertGreater(mid.net, ask.net)

    def test_fee_uses_real_formula(self):
        write_world(self.path, "oracle")
        rows, _, _ = score.load(self.path)
        t = replay.simulate(rows, 0.0, "ask", 10, self.fees)[0]
        self.assertAlmostEqual(t.fee, self.fees.order_fee(t.price, 10))

    def test_below_contract_settles_correctly(self):
        """A 'below' contract's YES pays when price is UNDER the strike."""
        r = {"hit": 0, "yes_direction": "below"}
        self.assertEqual(replay.yes_pays(r), 1.0)
        r = {"hit": 1, "yes_direction": "below"}
        self.assertEqual(replay.yes_pays(r), 0.0)

    def test_missing_no_side_inferred_from_yes(self):
        r = {"yes_bid": 0.40, "yes_ask": 0.44, "no_bid": None, "no_ask": None}
        ya, na, ym, nm = replay.prices(r)
        self.assertAlmostEqual(na, 0.60)
        self.assertAlmostEqual(nb := 1 - ya, 0.56)

    def test_chronological_split(self):
        write_world(self.path, "noise", windows=100)
        rows, _, _ = score.load(self.path)
        train, test = replay.split_windows(rows, 0.6)
        self.assertEqual(len(train), 60)
        self.assertEqual(len(test), 40)
        self.assertTrue(max(train) < min(test))   # strictly chronological
        self.assertFalse(train & test)


class Report(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "p.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def run_report(self, kind, **kw):
        write_world(self.path, kind, windows=80)
        rows, _, _ = score.load(self.path)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            replay.report(rows, kw.get("fill", "ask"), 10, KalshiFees(),
                          kw.get("horizon"), 0.6)
        return buf.getvalue()

    def test_report_runs_and_labels_holdout(self):
        out = self.run_report("oracle")
        self.assertIn("ON HELD-OUT DATA", out)
        self.assertIn("Threshold sweep", out)
        self.assertIn("By horizon", out)
        self.assertIn("By distance", out)

    def test_report_with_horizon_skips_by_horizon(self):
        out = self.run_report("oracle", horizon=4)
        self.assertNotIn("By horizon", out)

    def test_diagnoses_fee_vs_model(self):
        out = self.run_report("book")
        self.assertIn("Gross negative", out)

    def test_too_little_data(self):
        write_world(self.path, "oracle", windows=8)
        rows, _, _ = score.load(self.path)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            replay.report(rows, "ask", 10, KalshiFees(), None, 0.6)
        self.assertIn("Not enough", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
