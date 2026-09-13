"""
Tests for the learning layer. python3 -m unittest test_learning -v

The important tests here are the refusals. A learner that always produces a
correction is worse than none, because it launders noise into something that
looks authoritative.
"""

import random
import unittest

from core.learning import (
    MIN_SAMPLES,
    IsotonicCalibrator,
    brier,
    learn_calibration,
    pava,
    split,
)


def rows_from(fn, n=2000, seed=1):
    """Rows whose true probability is fn(p) when the model says p."""
    random.seed(seed)
    out = []
    for i in range(n):
        p = random.random()
        out.append({
            "window_id": f"{i:06d}",
            "p": p,
            "hit": 1 if random.random() < fn(p) else 0,
            "sigmas": 0.5,
            "horizon": 12,
            "suppressed": False,
        })
    return out


class TestPava(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(pava([], []), [])

    def test_already_monotonic_is_unchanged(self):
        blocks = pava([0.1, 0.5, 0.9], [0.0, 0.5, 1.0])
        self.assertEqual([round(v, 6) for _, v in blocks], [0.0, 0.5, 1.0])

    def test_violations_get_pooled(self):
        blocks = pava([0.1, 0.2, 0.3], [1.0, 0.0, 1.0])
        self.assertTrue(all(
            blocks[i][1] <= blocks[i + 1][1] + 1e-9 for i in range(len(blocks) - 1)
        ))

    def test_output_is_monotonic_on_random_input(self):
        random.seed(3)
        xs = [random.random() for _ in range(200)]
        ys = [random.choice([0.0, 1.0]) for _ in range(200)]
        blocks = pava(xs, ys)
        vals = [v for _, v in blocks]
        self.assertEqual(vals, sorted(vals))


class TestCalibrator(unittest.TestCase):
    def test_unfitted_is_the_identity(self):
        c = IsotonicCalibrator()
        for p in (0.1, 0.5, 0.9):
            self.assertEqual(c.apply(p), p)

    def test_output_stays_inside_bounds(self):
        c = IsotonicCalibrator().fit([0.1, 0.9], [0, 1])
        for p in (0.0, 0.5, 1.0):
            self.assertGreaterEqual(c.apply(p), 0.001)
            self.assertLessEqual(c.apply(p), 0.999)

    def test_never_inverts_order(self):
        c = IsotonicCalibrator().fit(
            [i / 100 for i in range(100)],
            [1 if random.Random(2).random() < i / 100 else 0 for i in range(100)],
        )
        prev = -1
        for i in range(0, 101, 5):
            v = c.apply(i / 100)
            self.assertGreaterEqual(v, prev - 1e-9)
            prev = v

    def test_round_trips_through_dict(self):
        c = IsotonicCalibrator().fit([0.2, 0.4, 0.8], [0, 1, 1])
        back = IsotonicCalibrator.from_dict(c.to_dict())
        for p in (0.1, 0.3, 0.9):
            self.assertAlmostEqual(c.apply(p), back.apply(p), places=9)


class TestSplit(unittest.TestCase):
    def test_split_is_chronological(self):
        train, test = split([{"window_id": f"{i:04d}"} for i in range(100)])
        self.assertLess(train[-1]["window_id"], test[0]["window_id"])

    def test_no_row_appears_in_both(self):
        rows = [{"window_id": f"{i:04d}"} for i in range(100)]
        train, test = split(rows)
        self.assertEqual(
            set(r["window_id"] for r in train) & set(r["window_id"] for r in test),
            set(),
        )

    def test_split_covers_everything(self):
        rows = [{"window_id": f"{i:04d}"} for i in range(100)]
        train, test = split(rows)
        self.assertEqual(len(train) + len(test), 100)


class TestLearning(unittest.TestCase):
    def test_refuses_below_the_minimum(self):
        res = learn_calibration(rows_from(lambda p: p ** 1.6, n=MIN_SAMPLES - 1))
        self.assertFalse(res.accepted)
        self.assertIn("minimum", res.reason)

    def test_refuses_when_already_calibrated(self):
        """The most important test. No spurious corrections."""
        res = learn_calibration(rows_from(lambda p: p, n=2000, seed=6))
        self.assertFalse(res.accepted)
        self.assertIsNone(res.calibrator)

    def test_learns_a_real_miscalibration(self):
        res = learn_calibration(rows_from(lambda p: p ** 1.6, n=2000, seed=5))
        self.assertTrue(res.accepted)
        self.assertGreater(res.improvement_pct, 0)

    def test_learned_mapping_moves_toward_the_truth(self):
        res = learn_calibration(rows_from(lambda p: p ** 1.6, n=3000, seed=5))
        for p in (0.3, 0.5, 0.7):
            truth = p ** 1.6
            corrected = res.calibrator.apply(p)
            self.assertLess(abs(corrected - truth), abs(p - truth))

    def test_validation_uses_unseen_data(self):
        res = learn_calibration(rows_from(lambda p: p ** 1.6, n=2000, seed=5))
        self.assertGreater(res.n_test, 0)
        self.assertGreater(res.n_train, res.n_test)

    def test_worthless_model_collapses_to_the_base_rate(self):
        """
        When outcomes are independent of the prediction, the right correction
        is to ignore the prediction entirely and return the base rate. The
        learner should discover that on its own - and the flat mapping is the
        signal that the underlying model knows nothing.
        """
        random.seed(21)
        rows = [{
            "window_id": f"{i:06d}", "p": random.random(),
            "hit": random.choice([0, 1]), "sigmas": 0.5,
            "horizon": 12, "suppressed": False,
        } for i in range(2000)]
        res = learn_calibration(rows)
        self.assertTrue(res.accepted, "collapsing to base rate is a real improvement")

        outputs = [res.calibrator.apply(p / 10) for p in range(1, 10)]
        spread = max(outputs) - min(outputs)
        self.assertLess(spread, 0.25, "mapping should be nearly flat on noise")
        for o in outputs:
            self.assertLess(abs(o - 0.5), 0.2, "should sit near the base rate")

    def test_result_serialises(self):
        import json
        res = learn_calibration(rows_from(lambda p: p ** 1.6, n=2000, seed=5))
        json.dumps(res.to_dict())


class TestBrier(unittest.TestCase):
    def test_perfect_prediction_scores_zero(self):
        self.assertEqual(brier([1.0, 0.0], [1, 0]), 0.0)

    def test_always_half_scores_quarter(self):
        self.assertAlmostEqual(brier([0.5] * 100, [1, 0] * 50), 0.25, places=9)

    def test_confidently_wrong_scores_one(self):
        self.assertEqual(brier([0.0, 1.0], [1, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
