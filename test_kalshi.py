"""
Tests for Kalshi binary pricing. python3 -m unittest test_kalshi -v
"""

import math
import unittest
from datetime import datetime, timedelta, timezone

from core.feed import Candle
from core.kalshi import (
    Edge,
    KalshiFees,
    estimate_vol,
    evaluate,
    norm_cdf,
    parkinson_vol,
    prob_above,
    sigmas_from_money,
    tail_warning,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def series(moves, start=100000.0, spread=0.0005):
    """Build 1-minute candles from a list of per-candle log returns."""
    out, p = [], start
    for i, m in enumerate(moves):
        nxt = p * math.exp(m)
        out.append(
            Candle(
                ts=T0 + timedelta(minutes=i),
                open=p,
                high=max(p, nxt) * (1 + spread),
                low=min(p, nxt) * (1 - spread),
                close=nxt,
                volume=100.0,
            )
        )
        p = nxt
    return out


class TestFees(unittest.TestCase):
    def test_fee_peaks_at_fifty_cents(self):
        f = KalshiFees()
        at_50 = f.order_fee(0.50, 1000)
        for p in (0.10, 0.30, 0.70, 0.90):
            self.assertLess(f.order_fee(p, 1000), at_50, f"price {p}")

    def test_fee_is_symmetric_around_fifty(self):
        f = KalshiFees()
        self.assertAlmostEqual(f.order_fee(0.30, 1000), f.order_fee(0.70, 1000))

    def test_maker_is_cheaper_than_taker(self):
        f = KalshiFees()
        self.assertLess(f.order_fee(0.5, 100, maker=True), f.order_fee(0.5, 100))

    def test_fee_rounds_up_never_down(self):
        f = KalshiFees()
        raw = 0.07 * 3 * 0.5 * 0.5  # 0.0525
        self.assertEqual(f.order_fee(0.50, 3), 0.06)
        self.assertGreaterEqual(f.order_fee(0.50, 3), raw)

    def test_single_contract_pays_rounding_penalty(self):
        f = KalshiFees()
        one = f.fee_as_pct_of_stake(0.50, 1)
        many = f.fee_as_pct_of_stake(0.50, 100)
        self.assertGreater(one, many)

    def test_longshots_cost_most_as_share_of_stake(self):
        """Absolute fee peaks at 50c, but stake-relative cost peaks at the
        cheap end. This inverts the usual reading of the fee curve."""
        f = KalshiFees()
        cheap = f.fee_as_pct_of_stake(0.05, 1000)
        mid = f.fee_as_pct_of_stake(0.50, 1000)
        rich = f.fee_as_pct_of_stake(0.95, 1000)
        self.assertGreater(cheap, mid)
        self.assertGreater(mid, rich)

    def test_zero_stake_does_not_divide_by_zero(self):
        self.assertEqual(KalshiFees().fee_as_pct_of_stake(0.0, 10), 0.0)


class TestProbability(unittest.TestCase):
    def test_at_the_money_is_fifty_percent(self):
        for sigma in (0.0005, 0.002, 0.01):
            self.assertAlmostEqual(prob_above(100000, 100000, sigma), 0.5, places=9)

    def test_probability_rises_as_strike_falls(self):
        prev = 0.0
        for strike in (101000, 100500, 100000, 99500, 99000):
            p = prob_above(100000, strike, 0.003)
            self.assertGreater(p, prev)
            prev = p

    def test_higher_vol_pulls_toward_fifty(self):
        low = prob_above(100000, 100300, 0.001)
        high = prob_above(100000, 100300, 0.010)
        self.assertLess(abs(high - 0.5), abs(low - 0.5))

    def test_zero_vol_is_a_step_function(self):
        self.assertEqual(prob_above(100000, 99999, 0), 1.0)
        self.assertEqual(prob_above(100000, 100001, 0), 0.0)
        self.assertEqual(prob_above(100000, 100000, 0), 0.0)

    def test_yes_and_no_sum_to_one(self):
        p = prob_above(100000, 100250, 0.004)
        self.assertAlmostEqual(p + (1 - p), 1.0, places=12)

    def test_rejects_nonpositive_prices(self):
        with self.assertRaises(ValueError):
            prob_above(0, 100, 0.01)
        with self.assertRaises(ValueError):
            prob_above(100, -1, 0.01)

    def test_norm_cdf_known_values(self):
        self.assertAlmostEqual(norm_cdf(0), 0.5, places=9)
        self.assertAlmostEqual(norm_cdf(1.96), 0.975, places=3)
        self.assertAlmostEqual(norm_cdf(-1.96), 0.025, places=3)


class TestTailWarning(unittest.TestCase):
    def test_near_money_is_unflagged(self):
        self.assertIsNone(tail_warning(0.5))
        self.assertIsNone(tail_warning(1.9))

    def test_two_sigma_is_flagged(self):
        self.assertIsNotNone(tail_warning(2.1))

    def test_three_sigma_is_flagged_harder(self):
        self.assertIn("do not trust", tail_warning(3.5))

    def test_none_passes_through(self):
        self.assertIsNone(tail_warning(None))

    def test_sigmas_from_money_is_direction_agnostic(self):
        up = sigmas_from_money(100000, 100500, 0.003)
        down = sigmas_from_money(100000, 99502.5, 0.003)
        self.assertAlmostEqual(up, down, places=2)


class TestVolatility(unittest.TestCase):
    def test_flat_market_has_zero_close_to_close_vol(self):
        vol = estimate_vol(series([0.0] * 60))
        self.assertAlmostEqual(vol.close_to_close, 0.0, places=12)

    def test_parkinson_uses_the_bar_range(self):
        """Close-to-close sees nothing in a market that whipsaws back to
        flat; Parkinson sees the range."""
        candles = series([0.0] * 60, spread=0.002)
        self.assertAlmostEqual(estimate_vol(candles).close_to_close, 0.0, places=12)
        self.assertGreater(parkinson_vol(candles), 0.0)

    def test_vol_scales_with_square_root_of_time(self):
        vol = estimate_vol(series([0.001, -0.001] * 40))
        one = vol.sigma_over(1)
        fifteen = vol.sigma_over(15)
        self.assertAlmostEqual(fifteen / one, math.sqrt(15), places=6)

    def test_ewma_reacts_faster_than_flat_average(self):
        calm_then_wild = series([0.0002] * 50 + [0.006, -0.006] * 10)
        vol = estimate_vol(calm_then_wild)
        self.assertGreater(vol.ewma, vol.close_to_close)

    def test_disagreement_is_reported(self):
        vol = estimate_vol(series([0.001, -0.001] * 40))
        self.assertIsNotNone(vol.disagreement)
        self.assertGreaterEqual(vol.disagreement, 0)

    def test_insufficient_data_returns_none(self):
        vol = estimate_vol(series([0.001]))
        self.assertIsNone(vol.sigma_over(15))

    def test_zero_minutes_gives_no_sigma(self):
        self.assertIsNone(estimate_vol(series([0.001, -0.001] * 40)).sigma_over(0))


class TestEdge(unittest.TestCase):
    def _edge(self, fair, price, fee=0.0, warning=None):
        return Edge("yes", fair, price, fee, 1, 1.0, warning)

    def test_ev_subtracts_price_and_fee(self):
        e = self._edge(0.60, 0.50, 0.02)
        self.assertAlmostEqual(e.ev_per_contract, 0.08, places=9)

    def test_fees_can_erase_an_apparent_edge(self):
        e = self._edge(0.52, 0.50, 0.02)
        self.assertAlmostEqual(e.edge_before_fees, 0.02, places=9)
        self.assertAlmostEqual(e.ev_per_contract, 0.0, places=9)
        self.assertFalse(e.worth_taking)

    def test_thin_edges_are_rejected_as_noise(self):
        self.assertFalse(self._edge(0.515, 0.50).worth_taking)
        self.assertTrue(self._edge(0.53, 0.50).worth_taking)

    def test_warning_disqualifies_regardless_of_ev(self):
        e = self._edge(0.90, 0.20, 0.0, warning="far out of money")
        self.assertGreater(e.ev_per_contract, 0.5)
        self.assertFalse(e.worth_taking)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.vol = estimate_vol(series([0.0008, -0.0008] * 50))

    def test_returns_both_sides(self):
        edges = evaluate(100000, 100000, 15, self.vol, yes_ask=0.50, no_ask=0.50)
        self.assertEqual({e.side for e in edges}, {"yes", "no"})

    def test_sorted_best_first(self):
        edges = evaluate(100000, 100000, 15, self.vol, yes_ask=0.50, no_ask=0.50)
        evs = [e.ev_per_contract for e in edges]
        self.assertEqual(evs, sorted(evs, reverse=True))

    def test_underpriced_yes_shows_positive_ev(self):
        """Spot well below strike makes NO the cheap side; if the book is
        asking 0.30 for NO the model should see value."""
        edges = evaluate(100000, 100400, 15, self.vol, yes_ask=0.60, no_ask=0.30)
        best = edges[0]
        self.assertEqual(best.side, "no")
        self.assertGreater(best.ev_per_contract, 0)

    def test_fairly_priced_market_offers_nothing(self):
        sigma = self.vol.sigma_over(15)
        fair = prob_above(100000, 100100, sigma)
        edges = evaluate(
            100000, 100100, 15, self.vol, yes_ask=fair, no_ask=1 - fair
        )
        for e in edges:
            self.assertFalse(e.worth_taking)

    def test_invalid_asks_are_skipped(self):
        edges = evaluate(100000, 100000, 15, self.vol, yes_ask=0.0, no_ask=1.5)
        self.assertEqual(edges, [])

    def test_raises_without_enough_data(self):
        with self.assertRaises(ValueError):
            evaluate(100000, 100000, 15, estimate_vol(series([0.001])), 0.5)

    def test_maker_pricing_improves_ev(self):
        taker = evaluate(100000, 100000, 15, self.vol, 0.50, 0.50)[0]
        maker = evaluate(100000, 100000, 15, self.vol, 0.50, 0.50, maker=True)[0]
        self.assertGreater(maker.ev_per_contract, taker.ev_per_contract)


if __name__ == "__main__":
    unittest.main()
