"""
Tests for agent ablation: row alignment, and the chronological holdout.

The alignment tests exist because the previous implementation built its
baseline from `rows[:len(adjusted_p)]` - the first N rows of the log rather
than the N rows that were actually recomputed. Any skipped row shifted the two
series against each other and every reported number became meaningless. These
tests fail loudly if that ever comes back.
"""

from __future__ import annotations

import math
import unittest

from core.learning import MIN_BUCKET, ablate_agents, brier, split


def p_of(spot: float, strike: float, sigma: float) -> float:
    return 0.5 * (1 + math.erf(math.log(spot / strike) / sigma / math.sqrt(2)))


def row(i: int, *, hit: int, mult: float, sigma: float = 0.004,
        spot: float = 100_000.0, strike: float = 100_000.0,
        agent: str = "jump_detector", with_agent: bool = True) -> dict:
    """One synthetic resolved window."""
    r = {
        "window_id": f"W{i:05d}",
        "sigma": sigma,
        "spot": spot,
        "strike": strike,
        "hit": hit,
        "p": p_of(spot, strike, sigma),
    }
    if with_agent:
        r["agents"] = {agent: {"vol_multiplier": mult}}
    else:
        r["agents"] = {}
    return r


class TestAblationAlignment(unittest.TestCase):
    """brier_with must be measured on exactly the rows that were recomputed."""

    def test_skipped_rows_do_not_shift_the_baseline(self):
        rows = []
        # Interleave recomputable and non-recomputable rows so that any
        # positional slicing misaligns the series.
        for i in range(80):
            recomputable = (i % 2 == 0)
            rows.append(row(
                i,
                hit=1 if i % 3 else 0,
                mult=1.15,
                spot=100_000.0 + (i * 40),
                with_agent=recomputable,
            ))

        out = ablate_agents(rows, holdout=None)
        s = out["jump_detector"]

        kept = [r for r in rows if r.get("agents", {}).get("jump_detector")]
        expected = brier([r["p"] for r in kept], [r["hit"] for r in kept])

        self.assertEqual(s["n"], len(kept))
        self.assertAlmostEqual(s["brier_with"], round(expected, 6), places=6)

    def test_missing_sigma_rows_are_excluded_from_both_series(self):
        rows = [row(i, hit=i % 2, mult=1.1, spot=100_000.0 + i * 50)
                for i in range(60)]
        for i in (3, 9, 14, 27):
            rows[i]["sigma"] = None

        out = ablate_agents(rows, holdout=None)
        s = out["jump_detector"]

        self.assertEqual(s["n"], 56)

        kept = [r for r in rows if r["sigma"]]
        expected = brier([r["p"] for r in kept], [r["hit"] for r in kept])
        self.assertAlmostEqual(s["brier_with"], round(expected, 6), places=6)

    def test_neutral_multiplier_changes_nothing(self):
        rows = [row(i, hit=i % 2, mult=1.0, spot=100_000.0 + i * 50)
                for i in range(60)]
        s = ablate_agents(rows, holdout=None)["jump_detector"]

        self.assertAlmostEqual(s["brier_with"], s["brier_without"], places=6)
        self.assertEqual(s["windows_affected"], 0)
        self.assertAlmostEqual(s["delta_pct"], 0.0, places=2)

    def test_below_min_bucket_is_not_reported(self):
        rows = [row(i, hit=i % 2, mult=1.1, spot=100_000.0 + i * 50)
                for i in range(MIN_BUCKET - 1)]
        self.assertEqual(ablate_agents(rows, holdout=None), {})


class TestChronologicalHoldout(unittest.TestCase):

    def test_holdout_block_is_reported(self):
        rows = [row(i, hit=i % 2, mult=1.1, spot=100_000.0 + i * 50)
                for i in range(100)]
        out = ablate_agents(rows, holdout=0.3)

        self.assertIn("_holdout", out)
        self.assertEqual(out["_holdout"]["n_train"], 70)
        self.assertEqual(out["_holdout"]["n_test"], 30)

    def test_split_is_chronological_not_random(self):
        rows = [row(i, hit=i % 2, mult=1.1) for i in range(100)]
        rows.reverse()
        train, test = split(rows, 0.3)

        self.assertEqual(train[0]["window_id"], "W00000")
        self.assertEqual(test[-1]["window_id"], "W00099")
        self.assertTrue(
            all(a["window_id"] < b["window_id"] for a, b in zip(train, train[1:]))
        )

    def test_consistent_agent_is_confirmed(self):
        # Widening sigma is right in both halves: the strike is far from spot
        # and the outcome keeps landing on the near side.
        rows = []
        for i in range(120):
            rows.append(row(i, hit=1, mult=1.30, spot=101_200.0, sigma=0.004))

        s = ablate_agents(rows, holdout=0.3)["jump_detector"]

        self.assertIsNotNone(s["train"])
        self.assertIsNotNone(s["test"])
        self.assertTrue(s["confirmed"])
        self.assertEqual(s["verdict"], s["train"]["helps"] and "helps" or "hurts")
        self.assertEqual(s["train"]["helps"], s["test"]["helps"])

    def test_agent_that_flips_sign_is_unproven(self):
        # First half: hits land where the widened sigma is the better call.
        # Second half: the same multiplier is now wrong.
        rows = []
        for i in range(70):
            rows.append(row(i, hit=1, mult=1.35, spot=101_200.0))
        for i in range(70, 100):
            rows.append(row(i, hit=0, mult=1.35, spot=101_200.0))

        s = ablate_agents(rows, holdout=0.3)["jump_detector"]

        self.assertNotEqual(s["train"]["helps"], s["test"]["helps"])
        self.assertFalse(s["confirmed"])
        self.assertEqual(s["verdict"], "unproven")

    def test_too_few_rows_in_a_half_yields_no_holdout_verdict(self):
        rows = [row(i, hit=i % 2, mult=1.1, spot=100_000.0 + i * 50)
                for i in range(40)]
        # 30% of 40 is 12 rows, under MIN_BUCKET, so the late half cannot score.
        s = ablate_agents(rows, holdout=0.3)["jump_detector"]

        self.assertIsNone(s["test"])
        self.assertFalse(s["confirmed"])
        self.assertEqual(s["verdict"], "no holdout")

    def test_holdout_none_omits_the_split_entirely(self):
        rows = [row(i, hit=i % 2, mult=1.1, spot=100_000.0 + i * 50)
                for i in range(100)]
        out = ablate_agents(rows, holdout=None)

        self.assertNotIn("_holdout", out)
        self.assertNotIn("train", out["jump_detector"])

    def test_full_sample_numbers_are_unchanged_by_the_holdout_flag(self):
        rows = [row(i, hit=i % 2, mult=1.1, spot=100_000.0 + i * 50)
                for i in range(100)]
        a = ablate_agents(rows, holdout=None)["jump_detector"]
        b = ablate_agents(rows, holdout=0.3)["jump_detector"]

        for k in ("brier_with", "brier_without", "helps", "delta_pct", "n"):
            self.assertEqual(a[k], b[k], k)


class TestMultipleAgents(unittest.TestCase):

    def test_each_agent_scored_on_its_own_rows(self):
        rows = []
        for i in range(80):
            r = row(i, hit=i % 2, mult=1.1, spot=100_000.0 + i * 50)
            r["agents"] = {
                "jump_detector": {"vol_multiplier": 1.10},
                "time_of_day": {"vol_multiplier": 0.95},
            }
            rows.append(r)

        out = ablate_agents(rows, holdout=None)
        self.assertIn("jump_detector", out)
        self.assertIn("time_of_day", out)
        self.assertEqual(out["jump_detector"]["n"], 80)
        self.assertEqual(out["time_of_day"]["n"], 80)
        self.assertNotEqual(
            out["jump_detector"]["brier_without"],
            out["time_of_day"]["brier_without"],
        )

    def test_empty_and_agentless_logs(self):
        self.assertEqual(ablate_agents([]), {})
        rows = [{"window_id": f"W{i}", "p": 0.5, "hit": 0, "agents": {}}
                for i in range(50)]
        self.assertEqual(ablate_agents(rows), {})


if __name__ == "__main__":
    unittest.main()
