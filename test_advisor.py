"""
Tests for advisor.assess - the verdict logic of the manual trading helper.

The most important property is in the first test: a fairly priced market
must read FAIR, not SKIP. Buying at the ask always costs the fee and part of
the spread, so an early version of this logic called nearly every market
"overpriced". That is false, and telling a beginner a fair market is a bad
one teaches the wrong lesson.
"""

from __future__ import annotations

import unittest

from advisor import EXTREME, assess, fee


def v(**k):
    return assess(**k)["verdict"]


class VerdictTest(unittest.TestCase):

    def test_fair_market_reads_fair_not_skip(self):
        a = assess(p_model=0.50, yes_bid=0.495, yes_ask=0.505, minutes_left=4)
        self.assertEqual(a["verdict"], "FAIR")
        self.assertIn("fee", a["reason"])

    def test_fair_market_still_loses_on_average_after_fee(self):
        """FAIR must not be read as 'profitable' - the numbers must say so."""
        a = assess(p_model=0.50, yes_bid=0.495, yes_ask=0.505, minutes_left=4)
        self.assertLess(a["sides"]["YES"]["edge"], 0)
        self.assertLess(a["sides"]["NO"]["edge"], 0)

    def test_overpriced_side_is_not_the_one_suggested(self):
        a = assess(p_model=0.40, yes_bid=0.54, yes_ask=0.55, minutes_left=4)
        self.assertEqual(a["best"], "NO")

    def test_clear_discount_is_worth_a_look(self):
        self.assertEqual(v(p_model=0.75, yes_bid=0.59, yes_ask=0.60,
                           minutes_left=4), "WORTH A LOOK")

    def test_both_sides_overpriced_is_skip(self):
        # Wide-ish market where the model sits in the middle: every side
        # costs more than it is worth before the fee.
        self.assertEqual(v(p_model=0.50, yes_bid=0.46, yes_ask=0.54,
                           minutes_left=4), "SKIP")


class RedFlagTest(unittest.TestCase):
    """Any of these overrides an attractive price."""

    GOOD = dict(p_model=0.75, yes_bid=0.59, yes_ask=0.60)

    def test_wide_spread(self):
        self.assertEqual(v(p_model=0.5, yes_bid=0.40, yes_ask=0.60,
                           minutes_left=4), "SKIP")

    def test_under_a_minute(self):
        self.assertEqual(v(**self.GOOD, minutes_left=0.5), "SKIP")

    def test_stale_quote(self):
        self.assertEqual(v(**self.GOOD, minutes_left=4, quote_age_s=90), "SKIP")

    def test_suppressed_window(self):
        self.assertEqual(v(**self.GOOD, minutes_left=4, suppressed=True), "SKIP")

    def test_missing_side_of_the_book(self):
        a = assess(p_model=0.5, yes_bid=0.49, yes_ask=None, minutes_left=4)
        self.assertEqual(a["verdict"], "SKIP")
        self.assertEqual(a["sides"], {})

    def test_near_certain_contract_is_skip_even_if_cheap(self):
        """Risking 97c to make 3c is a bad bet whatever the model says."""
        a = assess(p_model=0.99, yes_bid=0.955, yes_ask=0.965, minutes_left=4)
        self.assertEqual(a["verdict"], "SKIP")
        self.assertGreaterEqual(a["sides"][a["best"]]["price"], EXTREME)

    def test_early_window_warns_but_does_not_block(self):
        a = assess(**self.GOOD, minutes_left=13)
        self.assertEqual(a["verdict"], "WORTH A LOOK")
        self.assertTrue(any("12 minutes" in f for f in a["flags"]))


class ArithmeticTest(unittest.TestCase):

    def test_fee_formula(self):
        self.assertAlmostEqual(fee(0.50), 0.0175)
        self.assertAlmostEqual(fee(0.90), 0.0063)

    def test_no_side_costs_one_minus_the_yes_bid(self):
        a = assess(p_model=0.5, yes_bid=0.62, yes_ask=0.64, minutes_left=4)
        self.assertAlmostEqual(a["sides"]["NO"]["price"], 0.38)

    def test_win_plus_loss_is_one_dollar(self):
        """What you risk and what you can make always sum to the $1 payout."""
        a = assess(p_model=0.6, yes_bid=0.55, yes_ask=0.57, minutes_left=4)
        for side in a["sides"].values():
            self.assertAlmostEqual(side["win"] + side["loss"], 1.0)

    def test_fair_value_is_the_blend(self):
        a = assess(p_model=0.70, yes_bid=0.49, yes_ask=0.51, minutes_left=4)
        self.assertAlmostEqual(a["fair_yes"], 0.60)


if __name__ == "__main__":
    unittest.main()


class MovementPanelTest(unittest.TestCase):
    """The panel describes; it must not invent a direction from noise."""

    def test_gap_and_typical_move(self):
        from advisor import movement
        m = movement(spot=86000.0, strike=85900.0, sigma=0.001, recent_ref=None)
        self.assertAlmostEqual(m["gap"], 100.0)
        self.assertAlmostEqual(m["typical"], 86.0)
        self.assertAlmostEqual(m["gap_in_moves"], 100 / 86.0)

    def test_small_move_reads_flat_not_up(self):
        from advisor import movement
        # $5 against an $86 typical move is noise, not a direction.
        m = movement(spot=86005.0, strike=86000.0, sigma=0.001,
                     recent_ref=86000.0)
        self.assertEqual(m["direction"], "flat")

    def test_real_moves_read_up_and_down(self):
        from advisor import movement
        up = movement(spot=86040.0, strike=86000.0, sigma=0.001, recent_ref=86000.0)
        dn = movement(spot=85960.0, strike=86000.0, sigma=0.001, recent_ref=86000.0)
        self.assertEqual(up["direction"], "up")
        self.assertEqual(dn["direction"], "down")

    def test_missing_inputs_do_not_crash(self):
        from advisor import movement
        m = movement(spot=86000.0, strike=85900.0, sigma=None, recent_ref=None)
        self.assertIsNone(m["typical"])
        self.assertEqual(m["direction"], "unknown")

    def test_gap_descriptions_scale_with_distance(self):
        from advisor import describe_gap
        self.assertIn("coin flip", describe_gap(0.1))
        self.assertIn("big lead", describe_gap(2.0))
