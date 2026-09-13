"""
Tests for the adjusters. python3 -m unittest test_adjusters -v
"""

import math
import unittest
from datetime import datetime, timedelta, timezone

from core.adjusters import (
    ALL_ADJUSTERS,
    Adjuster,
    Adjustment,
    Context,
    JumpDetector,
    MomentumSkew,
    RoundNumberPull,
    TimeOfDayVol,
    VolUncertainty,
    build_estimate,
)
from core.feed import Candle
from core.kalshi import estimate_vol

NOON = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def bars(moves, start=76888.0, minutes=1, t0=NOON, spread=0.0004):
    out, p = [], start
    for i, m in enumerate(moves):
        n = p * math.exp(m)
        out.append(
            Candle(
                t0 + timedelta(minutes=minutes * i),
                p,
                max(p, n) * (1 + spread),
                min(p, n) * (1 - spread),
                n,
                100.0,
            )
        )
        p = n
    return out


def ctx_for(candles, minutes=12, now=None, hourly=None, spot=None):
    return Context(
        spot=spot if spot else candles[-1].close,
        candles=candles,
        vol=estimate_vol(candles, 1.0),
        minutes_left=minutes,
        now=now or NOON,
        hourly=hourly,
    )


def alternating(n=120, size=0.0004):
    return [size if i % 2 else -size for i in range(n)]


class TestJumpDetector(unittest.TestCase):
    def test_calm_market_is_not_flagged(self):
        a = JumpDetector().run(ctx_for(bars(alternating())))
        self.assertFalse(a.suppress)
        self.assertEqual(a.headline, "Calm")

    def test_recent_jump_suppresses(self):
        moves = alternating(120)
        moves[-1] = 0.02
        a = JumpDetector().run(ctx_for(bars(moves)))
        self.assertTrue(a.suppress)
        self.assertGreater(a.vol_multiplier, 1.0)

    def test_older_jump_widens_without_suppressing(self):
        moves = alternating(120)
        moves[-4] = 0.02
        a = JumpDetector().run(ctx_for(bars(moves)))
        self.assertFalse(a.suppress)
        self.assertGreater(a.vol_multiplier, 1.0)

    def test_jump_cannot_hide_in_its_own_baseline(self):
        """The baseline must exclude the recent window, or a single huge
        candle inflates the yardstick it is measured against."""
        moves = alternating(120)
        moves[-1] = 0.05
        a = JumpDetector().run(ctx_for(bars(moves)))
        self.assertTrue(a.suppress)
        self.assertGreater(a.metrics["max_sd"], 3.5)

    def test_reports_how_long_ago(self):
        moves = alternating(120)
        moves[-3] = 0.02
        a = JumpDetector().run(ctx_for(bars(moves)))
        self.assertEqual(a.metrics["minutes_ago"], 3)

    def test_short_history_is_idle(self):
        a = JumpDetector().run(ctx_for(bars(alternating(10))))
        self.assertFalse(a.suppress)


class TestTimeOfDayVol(unittest.TestCase):
    def _hourly(self, quiet_hours, quiet=0.0005, loud=0.006, days=14):
        moves, stamps = [], []
        t = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
        for i in range(days * 24):
            h = (t + timedelta(hours=i)).hour
            moves.append(quiet if h in quiet_hours else loud)
            stamps.append(h)
        return bars(moves, minutes=60, t0=datetime(2026, 8, 1, tzinfo=timezone.utc))

    def test_quiet_hour_lowers_the_multiplier(self):
        hourly = self._hourly(quiet_hours={3, 4, 5})
        a = TimeOfDayVol().run(
            ctx_for(bars(alternating()), now=NOON.replace(hour=4), hourly=hourly)
        )
        self.assertLess(a.vol_multiplier, 1.0)

    def test_busy_hour_raises_the_multiplier(self):
        hourly = self._hourly(quiet_hours={3, 4, 5})
        a = TimeOfDayVol().run(
            ctx_for(bars(alternating()), now=NOON.replace(hour=14), hourly=hourly)
        )
        self.assertGreater(a.vol_multiplier, 1.0)

    def test_multiplier_is_clamped_both_ways(self):
        """A thin bucket must not be able to produce an extreme multiplier."""
        hourly = self._hourly(quiet_hours=set(range(1, 24)), quiet=0.00001, loud=0.05)
        for hour in (0, 5):
            a = TimeOfDayVol().run(
                ctx_for(bars(alternating()), now=NOON.replace(hour=hour), hourly=hourly)
            )
            self.assertGreaterEqual(a.vol_multiplier, TimeOfDayVol.CLAMP_LO)
            self.assertLessEqual(a.vol_multiplier, TimeOfDayVol.CLAMP_HI)

    def test_extreme_input_does_not_pin_the_multiplier_at_the_clamp(self):
        """
        The failure this agent was fixed for.

        38% of logged windows sat exactly on the old clamp, because a bucket
        of ~12 samples can produce a 1.4 ratio from noise alone. Shrinkage
        must keep even an absurd input well inside the bounds, so the clamp
        is a backstop rather than the thing setting the number.
        """
        # A realistic hour-of-day spread, not a pathological one: busy hours
        # moving about 1.5x the quiet ones. The old code clamped on input like
        # this; shrinkage should leave it comfortably inside the bounds.
        hourly = self._hourly(quiet_hours={3, 4, 5}, quiet=0.004, loud=0.006)
        # Hour 4 is one of the three quiet hours, so it is genuinely unusual
        # against an average dominated by the other 21.
        a = TimeOfDayVol().run(
            ctx_for(bars(alternating()), now=NOON.replace(hour=4), hourly=hourly)
        )
        self.assertGreater(a.vol_multiplier, TimeOfDayVol.CLAMP_LO)
        self.assertLess(a.vol_multiplier, 1.0)

    def test_thin_bucket_moves_less_than_a_thick_one(self):
        """Same pattern, more evidence, bigger correction."""
        def mult_for(days):
            hourly = self._hourly(quiet_hours={3, 4, 5}, quiet=0.004,
                                  loud=0.006, days=days)
            return TimeOfDayVol().run(
                ctx_for(bars(alternating()), now=NOON.replace(hour=4), hourly=hourly)
            ).vol_multiplier

        thin, thick = mult_for(7), mult_for(40)
        self.assertLess(thick, thin)      # both below 1; thick is further
        self.assertLess(thick, 1.0)

    def test_no_history_means_no_effect(self):
        a = TimeOfDayVol().run(ctx_for(bars(alternating()), hourly=None))
        self.assertEqual(a.vol_multiplier, 1.0)

    def test_too_little_history_means_no_effect(self):
        a = TimeOfDayVol().run(
            ctx_for(bars(alternating()), hourly=bars([0.001] * 40, minutes=60))
        )
        self.assertEqual(a.vol_multiplier, 1.0)


class TestVolUncertainty(unittest.TestCase):
    def test_agreement_leaves_vol_alone(self):
        a = VolUncertainty().run(ctx_for(bars(alternating(), spread=0.0002)))
        self.assertEqual(a.vol_multiplier, 1.0)
        self.assertFalse(a.suppress)

    def test_disagreement_does_not_widen(self):
        """
        Widening is deliberately off.

        Ablation on 1,278 logged calls showed removing this multiplier
        improved Brier by 0.9%: the model is underconfident near the money,
        and widening pushes probabilities toward 50%, which made it worse.
        """
        a = VolUncertainty().run(ctx_for(bars(alternating(), spread=0.0005)))
        self.assertEqual(a.vol_multiplier, 1.0)

    def test_widening_can_be_re_enabled_and_is_capped(self):
        """The mechanism is intact, just switched off."""
        original = VolUncertainty.WIDEN
        try:
            VolUncertainty.WIDEN = 0.5
            a = VolUncertainty().run(ctx_for(bars(alternating(), spread=0.05)))
            self.assertGreater(a.vol_multiplier, 1.0)
            self.assertLessEqual(a.vol_multiplier, VolUncertainty.WIDEN_CAP)
        finally:
            VolUncertainty.WIDEN = original

    def test_suppression_survives_the_widening_change(self):
        """The safety flag is the part worth keeping."""
        a = VolUncertainty().run(ctx_for(bars(alternating(), spread=0.05)))
        self.assertTrue(a.suppress)
        self.assertEqual(a.vol_multiplier, 1.0)

    def test_extreme_disagreement_suppresses(self):
        a = VolUncertainty().run(ctx_for(bars(alternating(), spread=0.05)))
        self.assertTrue(a.suppress)


class TestMomentumSkew(unittest.TestCase):
    def test_random_walk_shows_no_momentum(self):
        import random

        random.seed(11)
        a = MomentumSkew().run(ctx_for(bars([random.gauss(0, 0.0004) for _ in range(120)])))
        self.assertEqual(a.drift_pct, 0.0)
        self.assertEqual(a.headline, "No momentum")

    def test_alternating_series_detects_reversal(self):
        a = MomentumSkew().run(ctx_for(bars(alternating())))
        self.assertLess(a.metrics["autocorrelation"], 0)

    def test_drift_stays_tiny_even_when_detected(self):
        """This is the weakest agent. It must never move the number much."""
        a = MomentumSkew().run(ctx_for(bars(alternating())))
        self.assertLess(abs(a.drift_pct), 0.05)

    def test_never_suppresses(self):
        for moves in (alternating(), [0.001] * 120):
            self.assertFalse(MomentumSkew().run(ctx_for(bars(moves))).suppress)


class TestRoundNumberPull(unittest.TestCase):
    def test_far_from_round_number_is_idle(self):
        a = RoundNumberPull().run(ctx_for(bars(alternating()), spot=76888.0))
        self.assertEqual(a.vol_multiplier, 1.0)

    def test_sitting_on_a_thousand_narrows_slightly(self):
        a = RoundNumberPull().run(ctx_for(bars(alternating()), spot=77000.0))
        self.assertLess(a.vol_multiplier, 1.0)
        self.assertGreater(a.vol_multiplier, 0.9)

    def test_never_suppresses(self):
        a = RoundNumberPull().run(ctx_for(bars(alternating()), spot=77000.0))
        self.assertFalse(a.suppress)


class TestComposition(unittest.TestCase):
    def test_every_adjuster_reports(self):
        est = build_estimate(ctx_for(bars(alternating())))
        self.assertEqual(len(est.adjustments), len(ALL_ADJUSTERS))
        self.assertEqual(
            len({a.agent for a in est.adjustments}), len(ALL_ADJUSTERS)
        )

    def test_multipliers_compound(self):
        class Half(Adjuster):
            name = "half"
            def analyze(self, ctx):
                return Adjustment("half", "strong", vol_multiplier=0.5)

        est = build_estimate(ctx_for(bars(alternating())), [Half(), Half()])
        self.assertAlmostEqual(est.final_sigma, est.base_sigma * 0.25, places=9)

    def test_one_suppressor_stops_everything(self):
        class Stop(Adjuster):
            name = "stop"
            def analyze(self, ctx):
                return Adjustment("stop", "strong", suppress=True)

        class Fine(Adjuster):
            name = "fine"
            def analyze(self, ctx):
                return Adjustment("fine", "strong")

        est = build_estimate(ctx_for(bars(alternating())), [Fine(), Stop(), Fine()])
        self.assertTrue(est.suppressed)
        self.assertEqual(len(est.suppressors), 1)

    def test_a_broken_adjuster_does_not_crash_the_estimate(self):
        class Broken(Adjuster):
            name = "broken"
            def analyze(self, ctx):
                raise RuntimeError("boom")

        est = build_estimate(ctx_for(bars(alternating())), [Broken()])
        self.assertEqual(est.final_sigma, est.base_sigma)
        self.assertIn("boom", est.adjustments[0].detail)

    def test_estimate_serialises(self):
        import json

        json.dumps(build_estimate(ctx_for(bars(alternating()))).to_dict())

    def test_no_adjuster_produces_a_negative_or_zero_sigma(self):
        est = build_estimate(ctx_for(bars(alternating())))
        self.assertGreater(est.final_sigma, 0)

    def test_evidence_ratings_are_valid(self):
        for a in build_estimate(ctx_for(bars(alternating()))).adjustments:
            self.assertIn(a.evidence, ("strong", "moderate", "weak"))


if __name__ == "__main__":
    unittest.main()
