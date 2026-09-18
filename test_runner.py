"""
Tests for the maker loop's decision, order-state reading, and order watching.

The watcher is the code that decides whether an order is still ours to worry
about, so it is tested against a fake exchange: fills that arrive late, fills
that land between the last poll and the cancel, and an order that cannot be
read at all.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import runner
from core import rails as R
from core.kalshi_exec import KalshiError

T0 = datetime(2026, 9, 20, 14, 41, 0, tzinfo=timezone.utc)


def pred(p_yes, yb, ya, ticker="KXBTC15M-X-T1", strike=60000.0):
    return {"p_yes": p_yes, "strike": strike,
            "market": {"ticker": ticker, "yes_bid": yb, "yes_ask": ya,
                       "no_bid": None, "no_ask": None,
                       "quoted_at": T0.isoformat()}}


class ChooseTest(unittest.TestCase):
    def test_yes_edge_measured_at_the_bid(self):
        # p=0.55, bid 0.48: edge 0.07 at the bid, only 0.05 at the mid.
        pick = runner.choose({"predictions": [pred(0.55, 0.48, 0.52)]}, 0.06)
        self.assertIsNotNone(pick)
        self.assertEqual((pick["side"], pick["price_cents"]), ("yes", 48))

    def test_no_side_rests_at_complement_of_yes_ask(self):
        # p_yes 0.40 -> p_no 0.60; no_bid = 1 - yes_ask = 0.48; edge 0.12.
        pick = runner.choose({"predictions": [pred(0.40, 0.46, 0.52)]}, 0.05)
        self.assertEqual((pick["side"], pick["price_cents"]), ("no", 48))

    def test_below_threshold_is_no_trade(self):
        self.assertIsNone(
            runner.choose({"predictions": [pred(0.50, 0.48, 0.52)]}, 0.05))

    def test_suppressed_is_no_trade(self):
        rec = {"suppressed": True, "predictions": [pred(0.9, 0.48, 0.52)]}
        self.assertIsNone(runner.choose(rec, 0.05))

    def test_largest_edge_across_strikes_wins(self):
        rec = {"predictions": [pred(0.56, 0.48, 0.52, "A"),
                               pred(0.70, 0.48, 0.52, "B")]}
        self.assertEqual(runner.choose(rec, 0.05)["ticker"], "B")

    def test_empty_bid_is_skipped_not_priced_at_zero(self):
        self.assertIsNone(
            runner.choose({"predictions": [pred(0.9, None, None)]}, 0.05))

    def test_matches_replay_decide(self):
        from replay import decide
        p = pred(0.61, 0.52, 0.55)
        row = {"p_yes": 0.61, **{k: p["market"][k] for k in
               ("yes_bid", "yes_ask", "no_bid", "no_ask")}}
        side, price, _ = decide(row, 0.05, "bid")
        pick = runner.choose({"predictions": [p]}, 0.05)
        self.assertEqual((pick["side"], pick["price_cents"]),
                         (side, int(round(price * 100))))


class ReadOrderTest(unittest.TestCase):
    def test_fixed_point_strings(self):
        st = runner.read_order({"order": {"fill_count_fp": "1.00",
                                          "remaining_count_fp": "0.00",
                                          "status": "executed"}})
        self.assertEqual((st["filled"], st["remaining"]), (1.0, 0.0))

    def test_create_response_shape(self):
        st = runner.read_order({"order_id": "x", "fill_count": "0.00",
                                "remaining_count": "1.00"})
        self.assertEqual(st["filled"], 0.0)

    def test_unreadable_is_none_not_zero(self):
        self.assertIsNone(runner.read_order({"order": {"status": "resting"}})["filled"])


class FakeClient:
    """Serves a scripted sequence of fill counts; records cancels."""

    def __init__(self, fills, after_cancel=None, fail_read=False, fail_cancel=False):
        self.fills = list(fills)
        self.after_cancel = after_cancel
        self.fail_read, self.fail_cancel = fail_read, fail_cancel
        self.cancelled = False

    def order(self, order_id):
        if self.fail_read:
            raise KalshiError("boom")
        if self.cancelled and self.after_cancel is not None:
            f = self.after_cancel
        else:
            f = self.fills.pop(0) if len(self.fills) > 1 else self.fills[0]
        return {"order": {"fill_count_fp": f"{f:.2f}", "status": "resting"}}

    def cancel(self, order_id):
        if self.fail_cancel:
            raise KalshiError("cannot cancel")
        self.cancelled = True
        return {}


class Clock:
    def __init__(self):
        self.t = T0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += timedelta(seconds=s)
        return False


class WatchOrderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.clock = Clock()
        self.deadline = T0 + timedelta(seconds=60)

    def tearDown(self):
        self.tmp.cleanup()

    def watch(self, client):
        return runner.watch_order(client, "oid", self.deadline, 1,
                                  now=self.clock.now, sleep=self.clock.sleep,
                                  root=self.root)

    def kill_set(self):
        return (self.root / R.KILL_FILE).exists()

    def test_fill_on_third_poll_records_time_and_stops(self):
        c = FakeClient([0, 0, 1])
        out = self.watch(c)
        self.assertEqual(out["filled"], 1.0)
        self.assertEqual(out["first_fill_after_s"], 2 * runner.POLL_SEC)
        self.assertFalse(c.cancelled)

    def test_no_fill_is_cancelled_at_deadline(self):
        c = FakeClient([0])
        out = self.watch(c)
        self.assertTrue(c.cancelled)
        self.assertEqual(out["filled"], 0.0)
        self.assertFalse(self.kill_set())

    def test_fill_between_last_poll_and_cancel_is_counted(self):
        c = FakeClient([0], after_cancel=1)
        out = self.watch(c)
        self.assertEqual(out["filled"], 1.0)
        self.assertIsNotNone(out["first_fill_after_s"])

    def test_unreadable_order_sets_kill_switch_and_still_cancels(self):
        c = FakeClient([0], fail_read=True)
        out = self.watch(c)
        self.assertTrue(c.cancelled)
        self.assertTrue(self.kill_set())
        self.assertIn("could not read", out["problem"])

    def test_failed_cancel_sets_kill_switch(self):
        c = FakeClient([0], fail_cancel=True)
        out = self.watch(c)
        self.assertTrue(self.kill_set())
        self.assertIn("cancel failed", out["problem"])


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "live_plan.json"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, **over):
        plan = {"threshold": 0.05, "horizon": 4, "count": 1,
                "max_total_loss": 25.0, "max_daily_loss": 10.0}
        plan.update(over)
        self.path.write_text(json.dumps(plan))

    def test_complete_plan_loads_with_hash(self):
        self.write()
        plan, h = runner.load_plan(self.path)
        self.assertEqual(plan["threshold"], 0.05)
        self.assertEqual(len(h), 12)

    def test_null_threshold_refused(self):
        self.write(threshold=None)
        with self.assertRaises(SystemExit):
            runner.load_plan(self.path)

    def test_size_above_one_refused(self):
        self.write(count=5)
        with self.assertRaises(SystemExit):
            runner.load_plan(self.path)

    def test_missing_file_refused(self):
        with self.assertRaises(SystemExit):
            runner.load_plan(self.path)

    def test_any_edit_changes_the_hash(self):
        self.write()
        _, h1 = runner.load_plan(self.path)
        self.write(threshold=0.06)
        _, h2 = runner.load_plan(self.path)
        self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
