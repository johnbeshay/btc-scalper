"""
Test suite. Run with: python3 -m unittest discover -v

These are written to probe edges, not to confirm the happy path. A test
that only passes when everything is already right is not earning its keep.
"""

import unittest
from datetime import datetime, timedelta, timezone

from core.agents import (
    LevelsAgent,
    PriceActionAgent,
    TrendAgent,
    VolatilityAgent,
    VolumeAgent,
)
from core.base import Confidence, Direction
from core.feed import Candle
from core.indicators import (
    FeeModel,
    atr,
    cluster_levels,
    rsi,
    sma,
    swing_points,
    zscore,
)
from core.orchestrator import Orchestrator

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(o, h, l, c, v=100.0, i=0):
    return Candle(T0 + timedelta(minutes=5 * i), o, h, l, c, v)


def flat_series(n=80, price=43000.0, vol=100.0):
    return [candle(price, price, price, price, vol, i) for i in range(n)]


def ramp_series(n=80, start=43000.0, step_pct=0.001, vol=100.0):
    out, p = [], start
    for i in range(n):
        nxt = p * (1 + step_pct)
        out.append(candle(p, max(p, nxt), min(p, nxt), nxt, vol, i))
        p = nxt
    return out


class TestFeeModel(unittest.TestCase):
    def test_round_trip_sums_both_sides_and_slippage(self):
        m = FeeModel(0.60, 0.60, 0.05)
        self.assertAlmostEqual(m.round_trip_pct, 1.30, places=6)

    def test_zero_fees_means_zero_cost(self):
        m = FeeModel(0, 0, 0)
        self.assertEqual(m.round_trip_pct, 0)

    def test_long_breakeven_nets_exactly_zero(self):
        """breakeven_price and net_pct must agree. If they disagree, one lies."""
        m = FeeModel(0.60, 0.60, 0.05)
        entry = 43000.0
        be = m.breakeven_price(entry, "long")
        self.assertAlmostEqual(m.net_pct(entry, be, "long"), 0.0, places=6)

    def test_short_breakeven_nets_exactly_zero(self):
        m = FeeModel(0.60, 0.60, 0.05)
        entry = 43000.0
        be = m.breakeven_price(entry, "short")
        self.assertAlmostEqual(m.net_pct(entry, be, "short"), 0.0, places=6)

    def test_net_pct_sign_flips_for_short(self):
        m = FeeModel(0, 0, 0)
        self.assertAlmostEqual(m.net_pct(100, 110, "long"), 10.0, places=6)
        self.assertAlmostEqual(m.net_pct(100, 110, "short"), -10.0, places=6)

    def test_clears_costs_requires_margin_not_just_parity(self):
        m = FeeModel(0.5, 0.5, 0.0)  # 1.0% round trip
        self.assertFalse(m.clears_costs(1.0))   # exactly break even is not enough
        self.assertFalse(m.clears_costs(1.49))
        self.assertTrue(m.clears_costs(1.5))

    def test_clears_costs_ignores_direction(self):
        m = FeeModel(0.5, 0.5, 0.0)
        self.assertTrue(m.clears_costs(-2.0))


class TestIndicators(unittest.TestCase):
    def test_sma_returns_none_below_period(self):
        self.assertIsNone(sma([1, 2], 5))

    def test_sma_uses_only_the_last_period_values(self):
        self.assertEqual(sma([100, 100, 100, 1, 2, 3], 3), 2.0)

    def test_atr_none_without_enough_history(self):
        self.assertIsNone(atr(flat_series(5), 14))

    def test_atr_of_flat_market_is_zero(self):
        self.assertEqual(atr(flat_series(40), 14), 0.0)

    def test_rsi_all_gains_is_100(self):
        self.assertEqual(rsi(ramp_series(40, step_pct=0.001), 14), 100.0)

    def test_rsi_bounded_zero_to_hundred(self):
        value = rsi(ramp_series(40, step_pct=-0.001), 14)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 100.0)

    def test_zscore_none_when_no_variance(self):
        self.assertIsNone(zscore(5, [3, 3, 3]))

    def test_zscore_none_on_tiny_population(self):
        self.assertIsNone(zscore(5, [3]))

    def test_zscore_sign_and_magnitude(self):
        self.assertAlmostEqual(zscore(3, [1, 2, 3, 4, 5]), 0.0, places=6)
        self.assertGreater(zscore(10, [1, 2, 3, 4, 5]), 0)

    def test_swing_points_finds_an_obvious_peak(self):
        series = [candle(100, 100, 100, 100, i=i) for i in range(9)]
        series[4] = candle(100, 150, 100, 100, i=4)
        highs, _ = swing_points(series, lookback=2)
        self.assertIn(150, highs)

    def test_swing_points_empty_on_short_input(self):
        highs, lows = swing_points(flat_series(3), lookback=2)
        self.assertEqual((highs, lows), ([], []))

    def test_cluster_levels_groups_near_prices(self):
        levels = cluster_levels([43000, 43010, 43005, 45000], tolerance_pct=0.15)
        top = levels[0]
        self.assertEqual(top["touches"], 3)
        self.assertAlmostEqual(top["price"], 43005, places=0)

    def test_cluster_levels_handles_empty(self):
        self.assertEqual(cluster_levels([]), [])

    def test_cluster_levels_sorted_by_touches(self):
        levels = cluster_levels([1000, 1000, 1000, 2000, 3000])
        self.assertEqual(levels[0]["touches"], 3)


class TestAgentWarmup(unittest.TestCase):
    def test_every_agent_idles_below_warmup(self):
        for cls in (
            PriceActionAgent,
            VolumeAgent,
            LevelsAgent,
            TrendAgent,
            VolatilityAgent,
        ):
            agent = cls()
            signal = agent.run(flat_series(3))
            self.assertEqual(
                signal.direction, Direction.NEUTRAL, f"{cls.__name__} not neutral"
            )
            self.assertEqual(signal.headline, "Standing by", cls.__name__)

    def test_no_agent_raises_on_a_dead_flat_market(self):
        """Flat markets are where division-by-zero hides."""
        series = flat_series(120)
        for cls in (
            PriceActionAgent,
            VolumeAgent,
            LevelsAgent,
            TrendAgent,
            VolatilityAgent,
        ):
            with self.subTest(agent=cls.__name__):
                cls().run(series)


class TestPriceActionAgent(unittest.TestCase):
    def test_flat_market_reads_neutral(self):
        signal = PriceActionAgent().run(flat_series(60))
        self.assertEqual(signal.direction, Direction.NEUTRAL)

    def test_rising_market_reads_bullish(self):
        signal = PriceActionAgent().run(ramp_series(60, step_pct=0.004))
        self.assertEqual(signal.direction, Direction.BULLISH)

    def test_falling_market_reads_bearish(self):
        signal = PriceActionAgent().run(ramp_series(60, step_pct=-0.004))
        self.assertEqual(signal.direction, Direction.BEARISH)


class TestVolumeAgent(unittest.TestCase):
    def test_steady_volume_is_not_a_spike(self):
        signal = VolumeAgent().run(flat_series(60))
        self.assertEqual(signal.headline, "Normal volume")

    def test_spike_on_green_candle_is_bullish(self):
        series = ramp_series(60, step_pct=0.001)
        last = series[-1]
        series[-1] = Candle(
            last.ts, last.open, last.high, last.low, last.close, last.volume * 5
        )
        signal = VolumeAgent().run(series)
        self.assertEqual(signal.direction, Direction.BULLISH)
        self.assertEqual(signal.confidence, Confidence.HIGH)

    def test_spike_on_red_candle_is_bearish(self):
        series = ramp_series(60, step_pct=-0.001)
        last = series[-1]
        series[-1] = Candle(
            last.ts, last.open, last.high, last.low, last.close, last.volume * 5
        )
        self.assertEqual(VolumeAgent().run(series).direction, Direction.BEARISH)


class TestVolatilityAgent(unittest.TestCase):
    def test_dead_market_is_vetoed(self):
        signal = VolatilityAgent().run(flat_series(120))
        self.assertEqual(signal.headline, "Too quiet to trade")

    def test_zero_fees_removes_the_veto(self):
        """The veto must come from the fee model, not a hardcoded threshold."""
        agent = VolatilityAgent(fees=FeeModel(0, 0, 0))
        signal = agent.run(flat_series(120))
        self.assertNotEqual(signal.headline, "Too quiet to trade")

    def test_volatility_agent_never_takes_a_side(self):
        for series in (flat_series(120), ramp_series(120, step_pct=0.005)):
            self.assertEqual(
                VolatilityAgent().run(series).direction, Direction.NEUTRAL
            )


class TestOrchestrator(unittest.TestCase):
    def test_veto_forces_stand_down(self):
        verdict = Orchestrator().run(flat_series(120))
        self.assertTrue(verdict.vetoed)
        self.assertEqual(verdict.headline, "Stand down")
        self.assertEqual(verdict.direction, Direction.NEUTRAL)

    def test_veto_survives_unanimous_agreement(self):
        """Agreement must not be able to outvote the cost floor."""
        series = ramp_series(120, start=43000, step_pct=0.00004)
        verdict = Orchestrator().run(series)
        bullish = [s for s in verdict.signals if s.direction is Direction.BULLISH]
        self.assertTrue(bullish, "precondition: some agents should be bullish")
        self.assertTrue(verdict.vetoed)

    def test_strong_move_with_no_fees_produces_a_direction(self):
        orch = Orchestrator(fees=FeeModel(0, 0, 0))
        verdict = orch.run(ramp_series(120, step_pct=0.004))
        self.assertFalse(verdict.vetoed)
        self.assertEqual(verdict.direction, Direction.BULLISH)

    def test_score_is_bounded(self):
        for series in (flat_series(120), ramp_series(120, step_pct=0.004)):
            verdict = Orchestrator().run(series)
            self.assertGreaterEqual(verdict.score, 0)
            self.assertLessEqual(verdict.score, 100)

    def test_every_agent_reports_once(self):
        verdict = Orchestrator().run(ramp_series(120))
        self.assertEqual(len(verdict.signals), 5)
        self.assertEqual(len({s.agent for s in verdict.signals}), 5)

    def test_verdict_serialises_to_json_safe_types(self):
        import json

        payload = Orchestrator().run(ramp_series(120)).to_dict()
        json.dumps(payload)  # raises if anything is not serialisable
        self.assertIn("vetoed", payload)
        self.assertIn("signals", payload)


if __name__ == "__main__":
    unittest.main()
