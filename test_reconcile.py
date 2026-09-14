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
               no_cost=0, at="2026-09-13T22:45:00Z", fee=None, **over):
    """
    Shaped like the real thing: revenue in integer cents, costs and fees as
    fixed-point dollar STRINGS. The helper takes cents for convenience and
    converts, so a test that says yes_cost=40 means 40 cents.
    """
    d = {
        "ticker": ticker,
        "revenue": revenue,
        "yes_total_cost_dollars": f"{yes_cost / 100:.6f}",
        "no_total_cost_dollars": f"{no_cost / 100:.6f}",
        "settled_time": at,
    }
    if fee is not None:
        d["fee_cost"] = f"{fee / 100:.6f}"
    d.update(over)
    return d


def fill(ticker="KXBTC15M-26SEP13-B100", fee=2, **over):
    d = {"ticker": ticker, "fee_cost": f"{fee / 100:.6f}",
         "created_time": "2026-09-13T22:33:00Z"}
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

    def test_units_are_not_confused(self):
        """
        revenue is cents, cost is a dollar string. Reading either with the
        wrong converter is a 100x error that nothing downstream would catch.
        """
        r = RC.sync(self.state, [settlement(revenue=100, yes_cost=85)], [])
        s = r.applied[0]
        self.assertAlmostEqual(s.revenue, 1.00)
        self.assertAlmostEqual(s.cost, 0.85)

    def test_real_settlement_record(self):
        """The actual record from the first live demo fill."""
        real = {
            "event_ticker": "KXBTC15M-26SEP132015",
            "exchange_index": 2,
            "fee_cost": "0.009000",
            "market_result": "yes",
            "no_count_fp": "0.00",
            "no_total_cost_dollars": "0.000000",
            "revenue": 100,
            "settled_time": "2026-09-14T00:15:10.107758Z",
            "ticker": "KXBTC15M-26SEP132015-15",
            "value": 100,
            "yes_count_fp": "1.00",
            "yes_total_cost_dollars": "0.850000",
        }
        r = RC.sync(self.state, [real], [])
        self.assertEqual(len(r.applied), 1, r.unparseable and
                         r.unparseable[0].problems)
        s = r.applied[0]
        self.assertAlmostEqual(s.revenue, 1.00)
        self.assertAlmostEqual(s.cost, 0.85)
        self.assertAlmostEqual(s.fees, 0.009)
        self.assertAlmostEqual(s.pnl, 0.141)
        self.assertEqual(s.day, "2026-09-14")

    def test_day_comes_from_the_settle_time_in_utc(self):
        r = RC.sync(self.state, [settlement(at="2026-09-13T23:59:00Z")], [])
        self.assertEqual(r.applied[0].day, "2026-09-13")

    def test_epoch_seconds_timestamps_parse(self):
        r = RC.sync(self.state, [settlement(settled_time=1789689900)], [])
        self.assertTrue(r.applied[0].ok)
        self.assertIsNotNone(r.applied[0].day)

    def test_alternate_confirmed_names_are_tried(self):
        """`value` mirrors `revenue`; `market_ticker` mirrors `ticker`."""
        s = {"market_ticker": "X", "value": 100,
             "yes_total_cost_dollars": "0.300000",
             "no_total_cost_dollars": "0.000000",
             "settled_at": "2026-09-13T22:45:00Z"}
        r = RC.sync(self.state, [s], [])
        self.assertEqual(len(r.applied), 1)
        self.assertAlmostEqual(r.applied[0].pnl, 0.70)

    def test_unconfirmed_legacy_names_are_not_guessed(self):
        """
        A record using a plausible-but-unseen field name must fail loudly.
        Accepting `yes_total_cost` would mean guessing whether it is cents
        or dollars, and guessing wrong is a 100x error in the P&L that
        gates trading.
        """
        s = {"ticker": "X", "revenue": 100, "yes_total_cost": 30,
             "settled_time": "2026-09-13T22:45:00Z"}
        r = RC.sync(self.state, [s], [])
        self.assertEqual(len(r.unparseable), 1)
        self.assertEqual(r.applied, [])


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
        del s["yes_total_cost_dollars"]
        del s["no_total_cost_dollars"]
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

    def test_revenue_as_a_string_is_rejected(self):
        """revenue is integer cents; a string there means the schema moved."""
        r = RC.sync(self.state, [settlement(revenue="100")], [])
        self.assertEqual(len(r.unparseable), 1)

    def test_unparseable_cost_string_is_rejected(self):
        s = settlement()
        s["yes_total_cost_dollars"] = "not-a-number"
        s["no_total_cost_dollars"] = "also-not"
        r = RC.sync(self.state, [s], [])
        self.assertEqual(len(r.unparseable), 1)

    def test_booleans_are_not_treated_as_numbers(self):
        self.assertIsNone(RC.cents_to_dollars(True))
        self.assertIsNone(RC.dollars_to_dollars(True))

    def test_cents_converter_rejects_strings(self):
        """A dollar string read as cents would become 1/100th of itself."""
        self.assertIsNone(RC.cents_to_dollars("0.85"))

    def test_dollar_converter_reads_fixed_point_strings(self):
        self.assertAlmostEqual(RC.dollars_to_dollars("0.850000"), 0.85)
        self.assertAlmostEqual(RC.dollars_to_dollars("0.009000"), 0.009)

    def test_every_money_field_has_a_unit(self):
        for f in ("revenue", "yes_cost", "no_cost", "fee"):
            self.assertTrue(
                f in RC.CENTS_FIELDS or f in RC.DOLLAR_FIELDS, f
            )

    def test_report_is_not_clean_when_anything_was_unreadable(self):
        s = settlement()
        del s["revenue"]
        self.assertFalse(RC.sync(self.state, [s], []).clean)

    def test_report_is_clean_when_everything_parsed(self):
        self.assertTrue(RC.sync(self.state, [settlement(fee=1)], []).clean)


class FeeTest(StateFixture):

    def test_settlement_fee_wins_over_fills(self):
        """Summing both would double-count every fee."""
        r = RC.sync(self.state,
                    [settlement(revenue=100, yes_cost=0, fee=1)],
                    [fill(fee=99)])
        self.assertAlmostEqual(r.applied[0].fees, 0.01)

    def test_fills_are_used_when_the_settlement_has_no_fee(self):
        fills = [fill(fee=2), fill(fee=3)]
        r = RC.sync(self.state, [settlement(revenue=100, yes_cost=0)], fills)
        self.assertAlmostEqual(r.applied[0].fees, 0.05)

    def test_fees_from_another_ticker_are_not_applied(self):
        fills = [fill(ticker="OTHER", fee=50)]
        r = RC.sync(self.state, [settlement(revenue=100, yes_cost=0)], fills)
        self.assertAlmostEqual(r.applied[0].fees, 0.0)

    def test_unreadable_fee_is_counted_and_reported(self):
        bad = {"ticker": "KXBTC15M-26SEP13-B100"}
        r = RC.sync(self.state, [settlement(fee=None)], [bad])
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
