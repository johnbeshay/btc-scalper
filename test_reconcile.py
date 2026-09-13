"""
Tests for deriving realised P&L from exchange records.

The theme of this file is that the module must never invent a number. An
unreadable record has to stay unreadable and visibly so, because the figure it
feeds is the daily loss cap - and a loss cap computed from a partially-guessed
P&L is a loss cap that will not fire when it matters.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core import rails as R
from core import reconcile as RC


def settlement(ticker="KXBTC15M-26SEP13-B100", revenue=100, yes_cost=40,
               no_cost=0, at="2026-09-13T22:45:00Z", **over):
    d = {
        "ticker": ticker,
        "revenue": revenue,
        "yes_total_cost": yes_cost,
        "no_total_cost": no_cost,
        "settled_time": at,
    }
    d.update(over)
    return d


def fill(ticker="KXBTC15M-26SEP13-B100", fee=2, **over):
    d = {"ticker": ticker, "fee": fee, "created_time": "2026-09-13T22:33:00Z"}
    d.update(over)
    return d


class StateFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = R.State(Path(self.tmp.name) / "state.json")

    def tearDown(self):
        self.tmp.cleanup()


class ParsingTest(StateFixture):

    def test_pnl_is_revenue_minus_cost_minus_fees(self):
        r = RC.sync(self.state, [settlement(revenue=100, yes_cost=40)],
                    [fill(fee=2)])
        self.assertEqual(len(r.applied), 1)
        # $1.00 - $0.40 - $0.02
        self.assertAlmostEqual(r.applied[0].pnl, 0.58)

    def test_a_loss_is_negative(self):
        r = RC.sync(self.state, [settlement(revenue=0, yes_cost=40)],
                    [fill(fee=1)])
        self.assertAlmostEqual(r.applied[0].pnl, -0.41)

    def test_cents_are_converted_to_dollars_once(self):
        r = RC.sync(self.state, [settlement(revenue=250, yes_cost=0)], [])
        self.assertAlmostEqual(r.applied[0].revenue, 2.50)

    def test_no_side_cost_is_used_when_present(self):
        r = RC.sync(self.state,
                    [settlement(revenue=100, yes_cost=0, no_cost=35)], [])
        self.assertAlmostEqual(r.applied[0].cost, 0.35)

    def test_day_comes_from_the_settle_time_in_utc(self):
        r = RC.sync(self.state, [settlement(at="2026-09-13T23:59:00Z")], [])
        self.assertEqual(r.applied[0].day, "2026-09-13")

    def test_epoch_seconds_timestamps_parse(self):
        r = RC.sync(self.state, [settlement(settled_time=1789689900)], [])
        self.assertTrue(r.applied[0].ok)
        self.assertIsNotNone(r.applied[0].day)

    def test_alternate_field_names_are_tried(self):
        s = {"market_ticker": "X", "payout": 100, "yes_cost": 30,
             "settled_at": "2026-09-13T22:45:00Z"}
        r = RC.sync(self.state, [s], [])
        self.assertEqual(len(r.applied), 1)
        self.assertAlmostEqual(r.applied[0].pnl, 0.70)


class RefusalTest(StateFixture):
    """The module must decline rather than guess."""

    def test_missing_revenue_is_unparseable_not_zero(self):
        s = settlement()
        del s["revenue"]
        r = RC.sync(self.state, [s], [])
        self.assertEqual(len(r.unparseable), 1)
        self.assertEqual(r.applied, [])
        self.assertIsNone(r.unparseable[0].pnl)

    def test_missing_cost_is_unparseable_not_zero(self):
        s = settlement()
        del s["yes_total_cost"]
        del s["no_total_cost"]
        r = RC.sync(self.state, [s], [])
        self.assertEqual(len(r.unparseable), 1)

    def test_missing_settle_time_is_unparseable(self):
        s = settlement()
        del s["settled_time"]
        r = RC.sync(self.state, [s], [])
        self.assertEqual(len(r.unparseable), 1)
        self.assertTrue(any("day" in p for p in r.unparseable[0].problems))

    def test_unparseable_contributes_nothing_to_the_total(self):
        s = settlement()
        del s["revenue"]
        r = RC.sync(self.state, [s, settlement(revenue=100, yes_cost=0)], [])
        self.assertAlmostEqual(r.total_applied, 1.00)

    def test_non_numeric_money_is_rejected(self):
        r = RC.sync(self.state, [settlement(revenue="100")], [])
        self.assertEqual(len(r.unparseable), 1)

    def test_booleans_are_not_treated_as_numbers(self):
        self.assertIsNone(RC.cents_to_dollars(True))

    def test_report_is_not_clean_when_anything_was_unreadable(self):
        s = settlement()
        del s["revenue"]
        self.assertFalse(RC.sync(self.state, [s], []).clean)

    def test_report_is_clean_when_everything_parsed(self):
        self.assertTrue(RC.sync(self.state, [settlement()], [fill()]).clean)


class FeeTest(StateFixture):

    def test_fees_are_summed_per_ticker(self):
        fills = [fill(fee=2), fill(fee=3)]
        r = RC.sync(self.state, [settlement(revenue=100, yes_cost=0)], fills)
        self.assertAlmostEqual(r.applied[0].fees, 0.05)

    def test_fees_from_another_ticker_are_not_applied(self):
        fills = [fill(ticker="OTHER", fee=50)]
        r = RC.sync(self.state, [settlement(revenue=100, yes_cost=0)], fills)
        self.assertAlmostEqual(r.applied[0].fees, 0.0)

    def test_unreadable_fee_is_counted_and_reported(self):
        bad = {"ticker": "KXBTC15M-26SEP13-B100"}
        r = RC.sync(self.state, [settlement()], [bad])
        self.assertEqual(r.unreadable_fills, 1)
        self.assertFalse(r.clean)

    def test_no_fills_means_zero_fees_not_a_crash(self):
        r = RC.sync(self.state, [settlement(revenue=100, yes_cost=40)], [])
        self.assertAlmostEqual(r.applied[0].fees, 0.0)
        self.assertAlmostEqual(r.applied[0].pnl, 0.60)


class IdempotencyTest(StateFixture):
    """Running sync twice must not double-count."""

    def test_apply_writes_pnl(self):
        RC.sync(self.state, [settlement(revenue=0, yes_cost=40)], [], apply=True)
        self.assertAlmostEqual(self.state.pnl_today("2026-09-13"), -0.40)

    def test_second_sync_skips_what_was_already_applied(self):
        s = [settlement(revenue=0, yes_cost=40)]
        RC.sync(self.state, s, [], apply=True)
        r2 = RC.sync(self.state, s, [], apply=True)
        self.assertEqual(len(r2.skipped_duplicate), 1)
        self.assertEqual(r2.applied, [])
        self.assertAlmostEqual(self.state.pnl_today("2026-09-13"), -0.40)

    def test_dry_run_writes_nothing(self):
        s = [settlement(revenue=0, yes_cost=40)]
        r = RC.sync(self.state, s, [], apply=False)
        self.assertEqual(len(r.applied), 1)
        self.assertAlmostEqual(self.state.pnl_today("2026-09-13"), 0.0)
        self.assertEqual(self.state.applied_count(), 0)

    def test_ledger_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.json"
            s1 = R.State(path)
            RC.sync(s1, [settlement(revenue=0, yes_cost=40)], [], apply=True)
            s2 = R.State(path)
            self.assertEqual(s2.applied_count(), 1)
            self.assertAlmostEqual(s2.pnl_today("2026-09-13"), -0.40)

    def test_distinct_settlements_are_both_counted(self):
        a = settlement(ticker="A", revenue=0, yes_cost=40)
        b = settlement(ticker="B", revenue=0, yes_cost=30)
        RC.sync(self.state, [a, b], [], apply=True)
        self.assertAlmostEqual(self.state.pnl_today("2026-09-13"), -0.70)

    def test_same_ticker_different_time_is_not_a_duplicate(self):
        a = settlement(at="2026-09-13T22:45:00Z", revenue=0, yes_cost=40)
        b = settlement(at="2026-09-13T23:00:00Z", revenue=0, yes_cost=40)
        RC.sync(self.state, [a, b], [], apply=True)
        self.assertAlmostEqual(self.state.pnl_today("2026-09-13"), -0.80)

    def test_mark_applied_is_idempotent_directly(self):
        self.state.mark_applied("k", -1.0, "2026-09-13")
        self.state.mark_applied("k", -1.0, "2026-09-13")
        self.assertAlmostEqual(self.state.pnl_today("2026-09-13"), -1.0)


class LossCapIntegrationTest(StateFixture):
    """The point of all of this: a synced loss must actually block an order."""

    def test_synced_losses_trip_the_daily_cap(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = R.State(root / "s.json")

            losses = [
                settlement(ticker=f"T{i}", revenue=0, yes_cost=400,
                           at="2026-09-13T22:45:00Z")
                for i in range(3)
            ]
            RC.sync(state, losses, [], apply=True)
            self.assertAlmostEqual(state.pnl_today("2026-09-13"), -12.00)

            from datetime import datetime, timezone
            now = datetime(2026, 9, 13, 22, 50, tzinfo=timezone.utc)
            d_ = R.check(
                rails=R.Rails(), state=state, window_id="W1",
                suppressed=False, quoted_at=now.isoformat(),
                price_cents=40, count=1, root=root, now=now,
            )
            self.assertFalse(d_)
            self.assertTrue(any("daily loss cap" in r for r in d_.reasons))


class EmptyInputTest(StateFixture):

    def test_no_settlements(self):
        r = RC.sync(self.state, [], [])
        self.assertEqual(r.applied, [])
        self.assertEqual(r.total_applied, 0)
        self.assertTrue(r.clean)

    def test_none_inputs(self):
        r = RC.sync(self.state, None, None)
        self.assertEqual(r.applied, [])


if __name__ == "__main__":
    unittest.main()
