"""
Tests for the replay simulator's guard rails.

These exist because the first version of this tool reported a profit on a
pooled mixture of two different models, selected its threshold in a way that
favoured trading everything, and printed no warning when the two periods
flatly contradicted each other. Each of those is pinned here.
"""

from __future__ import annotations

import unittest

import replay
from replay import Summary, Trade
from core.kalshi import KalshiFees


def summ(n, net, stake=100.0):
    return Summary(n=n, wins=n // 2, stake=stake, gross=net, fees=0.0,
                   net=net, claimed=0.0, realised=0.0)


class AgreementTest(unittest.TestCase):
    """Do the two periods rank thresholds the same way?"""

    def test_identical_rankings_agree(self):
        res = [(t, summ(50, v), summ(50, v))
               for t, v in [(0.0, 10), (0.02, 20), (0.05, 30), (0.08, 40)]]
        self.assertGreater(replay.agreement(res), 0.9)

    def test_opposite_rankings_disagree(self):
        res = [(t, summ(50, a), summ(50, b)) for t, a, b in
               [(0.0, 10, 40), (0.02, 20, 30), (0.05, 30, 20), (0.08, 40, 10)]]
        self.assertLess(replay.agreement(res), 0)

    def test_both_humped_is_not_a_disagreement(self):
        """
        Both columns peaking in the middle is agreement, not conflict.
        An earlier slope-based check read this as the periods disagreeing.
        """
        res = [(t, summ(50, a), summ(50, b)) for t, a, b in
               [(0.0, 10, 12), (0.02, 40, 38), (0.05, 60, 55),
                (0.08, 35, 30), (0.10, 5, 8)]]
        self.assertGreater(replay.agreement(res), 0.9)

    def test_too_few_eligible_rows_returns_none(self):
        res = [(0.0, summ(50, 10), summ(50, 10)),
               (0.02, summ(2, 5), summ(2, 5))]
        self.assertIsNone(replay.agreement(res))

    def test_rows_without_enough_trades_are_ignored(self):
        res = [(t, summ(n, a), summ(n, b)) for t, n, a, b in
               [(0.0, 50, 10, 10), (0.02, 1, 999, -999), (0.05, 50, 20, 20),
                (0.08, 50, 30, 30), (0.10, 50, 40, 40)]]
        self.assertGreater(replay.agreement(res), 0.9)


class BootstrapTest(unittest.TestCase):

    def _trades(self, nets):
        return [Trade(window_id=f"W{i}", horizon=4, side="yes", contracts=10,
                      price=0.5, fee=0.0, claimed_edge=0.1,
                      payoff=1.0 if n > 0 else 0.0, sigmas=0.3)
                for i, n in enumerate(nets)]

    def test_all_winners_gives_a_positive_interval(self):
        lo, hi = replay.bootstrap_net(self._trades([5.0] * 40))
        self.assertGreater(lo, 0)

    def test_mixed_results_span_zero(self):
        lo, hi = replay.bootstrap_net(self._trades([5.0, -5.0] * 20))
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_too_few_trades_returns_none(self):
        self.assertIsNone(replay.bootstrap_net(self._trades([1.0, 2.0])))

    def test_interval_brackets_the_point_estimate(self):
        trades = self._trades([3.0, -1.0, 4.0, -2.0, 5.0] * 8)
        total = sum(t.net for t in trades)
        lo, hi = replay.bootstrap_net(trades)
        self.assertLessEqual(lo, total)
        self.assertGreaterEqual(hi, total)


class SelectionTest(unittest.TestCase):
    """
    The threshold is chosen on return on stake, not total net.

    Total net rewards whichever threshold trades most, which is the lowest
    one. On a model whose edge lives in its strong disagreements, that picks
    'trade everything' and loses to fees.
    """

    def test_roi_prefers_the_efficient_threshold(self):
        # The realistic trap: the wide threshold makes MORE in total while
        # making less per dollar risked. Selecting on net picks it; selecting
        # on ROI does not.
        wide = Summary(n=200, wins=120, stake=2000.0, gross=150.0, fees=110.0,
                       net=40.0, claimed=0.0, realised=0.0)      # +2% ROI
        tight = Summary(n=30, wins=25, stake=150.0, gross=40.0, fees=10.0,
                        net=30.0, claimed=0.0, realised=0.0)     # +20% ROI
        self.assertGreater(wide.net, tight.net)     # net would pick wide
        self.assertLess(wide.roi, tight.roi)        # roi picks tight
        best = max([(0.0, wide, wide), (0.10, tight, tight)],
                   key=lambda x: x[1].roi)
        self.assertEqual(best[0], 0.10)

    def test_min_train_trades_blocks_a_lucky_few(self):
        lucky = Summary(n=3, wins=3, stake=15.0, gross=10.0, fees=0.5,
                        net=9.5, claimed=0.0, realised=0.0)
        solid = Summary(n=40, wins=25, stake=200.0, gross=30.0, fees=10.0,
                        net=20.0, claimed=0.0, realised=0.0)
        results = [(0.05, solid, solid), (0.30, lucky, lucky)]
        eligible = [x for x in results if x[1].n >= replay.MIN_TRAIN_TRADES]
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0][0], 0.05)


class PriceInferenceTest(unittest.TestCase):

    def test_no_side_inferred_from_yes(self):
        ya, na, ym, nm = replay.prices(
            {"yes_bid": 0.40, "yes_ask": 0.44, "no_bid": None, "no_ask": None})
        self.assertAlmostEqual(na, 0.60)   # 1 - yes_bid
        self.assertAlmostEqual(nb := nm * 2 - na, 0.56, places=4)

    def test_yes_direction_flips_payoff(self):
        above = {"hit": 1, "yes_direction": "above"}
        below = {"hit": 1, "yes_direction": "below"}
        self.assertEqual(replay.yes_pays(above), 1.0)
        self.assertEqual(replay.yes_pays(below), 0.0)


class FeeTest(unittest.TestCase):

    def test_fee_is_largest_near_fifty_cents(self):
        f = KalshiFees(taker_multiplier=0.07)
        mid = f.order_fee(0.50, 10)
        edge_ = f.order_fee(0.05, 10)
        self.assertGreater(mid, edge_)


if __name__ == "__main__":
    unittest.main()
