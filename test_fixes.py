"""
Tests for the fixes of 2026-09-21.

Each test pins something that actually went wrong:

  - score.py dropped readings whose Coinbase close was missing, even when
    Kalshi's own settlement for that ticker was in the log
  - core/learning.py split rows, not windows, so one window could straddle
    train and test
  - the runner's order lookup and cancel went to shard 0; crypto is on
    shard 2, so it 404'd, reported an order that had filled as unfilled, and
    tripped the kill switch
  - makerstats.py read a log format the deployed runner never wrote
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import makerstats
import score
from core import learning
from core.kalshi_exec import DemoClient, KalshiError


def pred(wid, ticker, strike=77000.0, h=4):
    return {"type": "prediction", "schema": 3, "ladder": "kalshi",
            "window_id": wid, "horizon_min": h, "spot": 77000.0,
            "sigma": 0.001, "suppressed": False, "agents": {},
            "predictions": [{"strike": strike, "p_above": 0.6, "p_yes": 0.6,
                             "sigmas_out": 0.1,
                             "market": {"ticker": ticker,
                                        "yes_direction": "above",
                                        "implied_p_above": 0.55}}]}


def write(lines, path):
    path.write_text("\n".join(json.dumps(x) for x in lines))


# ---------------------------------------------------------------------------
# score.load
# ---------------------------------------------------------------------------

class SettlementWithoutCloseTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "p.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_settlement_grades_a_reading_with_no_outcome_line(self):
        write([pred("W1", "T1"),
               {"type": "settlement", "ticker": "T1", "result": "yes"}],
              self.path)
        rows, _, unresolved = score.load(self.path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["hit"], 1)
        self.assertEqual(rows[0]["hit_source"], "kalshi")
        self.assertIsNone(rows[0]["close"])
        self.assertEqual(unresolved, 0)

    def test_settlement_grades_a_reading_with_an_untrustworthy_close(self):
        write([pred("W1", "T1"),
               {"type": "outcome", "window_id": "W1", "close_price": 1.0,
                "trustworthy": False},
               {"type": "settlement", "ticker": "T1", "result": "no"}],
              self.path)
        rows, _, unresolved = score.load(self.path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["hit"], 0)
        self.assertEqual(unresolved, 0)

    def test_no_settlement_and_no_close_is_excluded(self):
        write([pred("W1", "T1")], self.path)
        rows, _, unresolved = score.load(self.path)
        self.assertEqual(rows, [])
        self.assertEqual(unresolved, 1)

    def test_close_still_used_as_fallback(self):
        write([pred("W1", "T1", strike=77000.0),
               {"type": "outcome", "window_id": "W1", "close_price": 77100.0,
                "trustworthy": True}], self.path)
        rows, _, _ = score.load(self.path)
        self.assertEqual(rows[0]["hit"], 1)
        self.assertEqual(rows[0]["hit_source"], "close")

    def test_settlement_beats_the_close_when_both_exist(self):
        write([pred("W1", "T1", strike=77000.0),
               {"type": "outcome", "window_id": "W1", "close_price": 77100.0,
                "trustworthy": True},
               {"type": "settlement", "ticker": "T1", "result": "no"}],
              self.path)
        rows, _, _ = score.load(self.path)
        self.assertEqual(rows[0]["hit"], 0)
        self.assertEqual(rows[0]["hit_source"], "kalshi")


# ---------------------------------------------------------------------------
# core.learning.split
# ---------------------------------------------------------------------------

class WindowSplitTest(unittest.TestCase):

    def rows(self, n_windows, per_window=3):
        return [{"window_id": f"W{w:04d}", "h": h}
                for w in range(n_windows) for h in range(per_window)]

    def test_no_window_appears_on_both_sides(self):
        # 11 windows x 3 readings = 33 rows. A 30% cut by ROW lands at index
        # 23, which is the third reading of window 7 - so the old split put
        # two of window 7's readings in training and one in the holdout.
        # (An earlier version of this test used 10 windows, where the row cut
        # happens to fall exactly on a window boundary; it passed against the
        # broken code and so proved nothing.)
        rows = self.rows(11)
        cut = int(len(rows) * 0.7)
        self.assertEqual(rows[cut - 1]["window_id"], rows[cut]["window_id"],
                         "fixture must put the row cut inside a window")
        train, test = learning.split(rows, 0.3)
        self.assertFalse({r["window_id"] for r in train}
                         & {r["window_id"] for r in test})

    def test_every_row_is_kept(self):
        train, test = learning.split(self.rows(17), 0.3)
        self.assertEqual(len(train) + len(test), 51)

    def test_train_precedes_test(self):
        train, test = learning.split(self.rows(20), 0.3)
        self.assertLess(max(r["window_id"] for r in train),
                        min(r["window_id"] for r in test))

    def test_one_reading_per_window_matches_the_old_behaviour(self):
        train, test = learning.split(self.rows(100, 1), 0.3)
        self.assertEqual((len(train), len(test)), (70, 30))

    def test_empty(self):
        self.assertEqual(learning.split([], 0.3), ([], []))


# ---------------------------------------------------------------------------
# core.kalshi_exec order lookup and cancel
# ---------------------------------------------------------------------------

class FakeClient(DemoClient):
    def __init__(self, orders=()):
        self.base = "x"
        self.key_id = "k"
        self._key = None
        self.calls = []
        self._orders = list(orders)

    def _request(self, method, path, body=None, query=None):
        self.calls.append((method, path, dict(query or {})))
        if method == "GET" and path.endswith("/portfolio/orders"):
            t = (query or {}).get("ticker")
            return {"orders": [o for o in self._orders
                               if not t or o["ticker"] == t]}
        if method == "DELETE":
            return {"order_id": path.rsplit("/", 1)[-1], "reduced_by": "1.00"}
        raise AssertionError(f"unexpected call {method} {path}")


SHARD2 = {"order_id": "A", "ticker": "KXBTC15M-X", "exchange_index": 2,
          "status": "resting", "fill_count_fp": "0.00",
          "remaining_count_fp": "1.00"}


class OrderLookupTest(unittest.TestCase):

    def test_found_through_the_list_endpoint_not_by_id(self):
        c = FakeClient([SHARD2])
        got = c.order("A", ticker="KXBTC15M-X")
        self.assertEqual(got["order"]["exchange_index"], 2)
        method, path, q = c.calls[0]
        self.assertEqual(path, "/trade-api/v2/portfolio/orders")
        self.assertNotIn("exchange_index", q)       # all shards
        self.assertEqual(q["ticker"], "KXBTC15M-X")

    def test_without_ticker_scans_recent_orders(self):
        c = FakeClient([SHARD2])
        self.assertEqual(c.order("A")["order"]["order_id"], "A")
        self.assertIn("min_ts", c.calls[0][2])

    def test_missing_order_raises(self):
        with self.assertRaises(KalshiError):
            FakeClient([SHARD2]).order("nope", ticker="KXBTC15M-X")

    def test_response_shape_matches_the_old_endpoint(self):
        got = FakeClient([SHARD2]).order("A", ticker="KXBTC15M-X")
        self.assertIn("order", got)
        self.assertEqual(got["order"]["fill_count_fp"], "0.00")


class CancelRoutingTest(unittest.TestCase):

    def test_uses_the_v2_path(self):
        c = FakeClient([SHARD2])
        c.cancel("A", ticker="KXBTC15M-X")
        self.assertEqual(c.calls[-1][1],
                         "/trade-api/v2/portfolio/events/orders/A")

    def test_ticker_is_sent_for_auto_routing(self):
        c = FakeClient([SHARD2])
        c.cancel("A", ticker="KXBTC15M-X")
        self.assertEqual(c.calls[-1][2], {"market_ticker": "KXBTC15M-X"})

    def test_explicit_shard_is_sent(self):
        c = FakeClient([SHARD2])
        c.cancel("A", exchange_index=2)
        self.assertEqual(c.calls[-1][2], {"exchange_index": 2})

    def test_bare_order_id_is_looked_up_first_never_sent_unrouted(self):
        """
        The failure this guards: an unrouted cancel defaults to shard 0,
        misses a shard-2 order, and leaves it resting.
        """
        c = FakeClient([SHARD2])
        c.cancel("A")
        self.assertEqual(c.calls[0][0], "GET")
        method, path, q = c.calls[-1]
        self.assertEqual(method, "DELETE")
        self.assertEqual(q, {"market_ticker": "KXBTC15M-X"})

    def test_unfindable_order_is_not_cancelled_on_a_guessed_shard(self):
        c = FakeClient([])
        with self.assertRaises(KalshiError):
            c.cancel("ghost")
        self.assertFalse([x for x in c.calls if x[0] == "DELETE"])

    def test_no_v1_paths_remain_for_order_writes(self):
        import inspect
        import core.kalshi_exec as ke
        src = inspect.getsource(ke.DemoClient.cancel)
        self.assertNotIn('"/trade-api/v2/portfolio/orders/{', src)


# ---------------------------------------------------------------------------
# makerstats
# ---------------------------------------------------------------------------

# The actual record the deployed runner wrote on 2026-09-19. Logged as
# unfilled; the account balance shows it filled and won (+$0.35).
REAL_POSTED = {
    "window_id": "20260919T0245", "action": "posted", "plan": "demo",
    "immediate_fill": 0.0, "order_id": "01a0b789-de48-7163-9f17-6ace49b939a0",
    "first_fill_after_s": None, "filled": 0.0, "cancelled": False,
    "problem": "could not read order: HTTP 404 ... cancel failed: HTTP 410",
    "ticker": "KXBTC15M-26SEP182245-45", "side": "no", "price_cents": 65,
    "edge": 0.0909,
}


class MakerStatsFormatTest(unittest.TestCase):

    def test_reads_the_deployed_runners_actions(self):
        by = makerstats.split_actions([
            REAL_POSTED,
            {"action": "refused", "reasons": ["kill switch present (x)"]},
            {"action": "no_trade"}, {"action": "no_trade"},
        ])
        self.assertEqual(len(by["posted"]), 1)
        self.assertEqual(len(by["refused"]), 1)
        self.assertEqual(len(by["no_trade"]), 2)

    def test_record_with_a_problem_is_untrusted(self):
        self.assertFalse(makerstats.trusted(REAL_POSTED))
        clean = dict(REAL_POSTED, problem=None)
        self.assertTrue(makerstats.trusted(clean))

    def test_fill_count_reads_the_logged_field(self):
        self.assertEqual(makerstats.filled_count({"filled": 1.0}), 1.0)
        self.assertEqual(makerstats.filled_count({"filled": None}), 0.0)
        self.assertEqual(makerstats.filled_count({}), 0.0)

    def test_side_and_settlement_decide_rightness(self):
        self.assertTrue(makerstats.model_was_right(
            REAL_POSTED, {"KXBTC15M-26SEP182245-45": "no"}))
        self.assertFalse(makerstats.model_was_right(
            REAL_POSTED, {"KXBTC15M-26SEP182245-45": "yes"}))
        self.assertIsNone(makerstats.model_was_right(REAL_POSTED, {}))


class ReconcileTest(unittest.TestCase):
    """--reconcile must replace the logged count with the exchange's."""

    def test_reconcile_corrects_the_real_misreported_fill(self):
        import core.kalshi_exec as ke

        class FillsOnly:
            def __init__(self, *_a, **_k): pass
            def fills(self, limit=1000):
                return {"fills": [{"order_id": REAL_POSTED["order_id"],
                                   "count_fp": "1.00"}]}

        orig_client, orig_creds = ke.DemoClient, ke.Credentials
        ke.DemoClient = FillsOnly
        ke.Credentials = type("C", (), {"from_file": staticmethod(lambda p: None)})
        try:
            rec = dict(REAL_POSTED)
            n, changed = makerstats.reconcile([rec])
        finally:
            ke.DemoClient, ke.Credentials = orig_client, orig_creds
        self.assertEqual((n, changed), (1, 1))
        self.assertEqual(rec["filled"], 1.0)
        self.assertTrue(rec["reconciled"])


if __name__ == "__main__":
    unittest.main()
