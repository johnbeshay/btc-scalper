"""
Tests for the settlement basis: it recovers a known beta, it refuses to ship
without out-of-sample support, and the logger only uses it when validated.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import basis


def synth_obs(beta_true, n=400, seed=1):
    rng = random.Random(seed)
    obs = []
    for i in range(n):
        z = rng.uniform(-0.003, 0.003)
        settle = z + rng.gauss(0, beta_true)
        obs.append({"z": z, "hit": 1 if settle > 0 else 0,
                    "window_id": f"W{i:05d}", "ticker": f"T{i:05d}"})
    return obs


class FitTest(unittest.TestCase):

    def test_recovers_a_known_beta(self):
        for bt in (0.0004, 0.0008, 0.0015):
            b = basis.fit_beta(synth_obs(bt, n=1500))
            self.assertAlmostEqual(b, bt, delta=bt * 0.25, msg=f"true {bt}")

    def test_larger_basis_gives_larger_estimate(self):
        small = basis.fit_beta(synth_obs(0.0004, n=800, seed=2))
        large = basis.fit_beta(synth_obs(0.0016, n=800, seed=2))
        self.assertGreater(large, small * 2)

    def test_refuses_too_few_observations(self):
        with self.assertRaises(ValueError):
            basis.fit_beta(synth_obs(0.001, n=10))


class RepriceTest(unittest.TestCase):

    def test_zero_basis_leaves_p_unchanged(self):
        row = {"p": 0.9, "sigma": 0.001, "spot": 77000.0, "strike": 76900.0,
               "drift_pct": 0.0}
        z = math.log(77000.0 / 76900.0) / 0.001
        expect = basis.phi(z)
        self.assertAlmostEqual(basis.reprice(row, 0.0), expect, places=6)

    def test_basis_pulls_far_strike_toward_half(self):
        row = {"p": 0.99, "sigma": 0.001, "spot": 77000.0, "strike": 76800.0,
               "drift_pct": 0.0}
        p0 = basis.reprice(row, 0.0)
        p1 = basis.reprice(row, 0.001)
        self.assertGreater(p0, p1)
        self.assertGreater(p1, 0.5)

    def test_at_the_money_is_unaffected(self):
        row = {"p": 0.5, "sigma": 0.001, "spot": 77000.0, "strike": 77000.0,
               "drift_pct": 0.0}
        self.assertAlmostEqual(basis.reprice(row, 0.002), 0.5, places=9)

    def test_missing_inputs_return_original_p(self):
        row = {"p": 0.42, "sigma": None, "spot": None, "strike": None}
        self.assertEqual(basis.reprice(row, 0.001), 0.42)


class OnePerWindowTest(unittest.TestCase):

    def test_dedupes_by_ticker_and_requires_kalshi_source(self):
        rows = [
            {"hit_source": "kalshi", "ticker": "A", "close": 100.0,
             "strike": 99.0, "hit": 1, "window_id": "W1"},
            {"hit_source": "kalshi", "ticker": "A", "close": 100.0,
             "strike": 99.0, "hit": 1, "window_id": "W1"},   # 2nd reading
            {"hit_source": "close", "ticker": "B", "close": 100.0,
             "strike": 99.0, "hit": 1, "window_id": "W2"},    # proxy: excluded
            {"hit_source": "kalshi", "ticker": None, "close": 100.0,
             "strike": 99.0, "hit": 1, "window_id": "W3"},    # no ticker
        ]
        obs = basis.one_per_window(rows)
        self.assertEqual([o["ticker"] for o in obs], ["A"])


class LoggerGateTest(unittest.TestCase):
    """The logger must price on raw sigma unless a validated file exists."""

    def _load_basis(self, path):
        import importlib.util, sys, types
        # import logger.load_basis without pulling in its network deps
        src = Path("logger.py").read_text()
        start = src.index("def load_basis(")
        end = src.index("\n\n\n", start)
        ns = {"json": json, "Path": Path, "BASIS_FILE": Path("unused")}
        exec(src[start:end], ns)
        return ns["load_basis"](path)

    def test_missing_file_means_zero(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self._load_basis(Path(d) / "nope.json"), 0.0)

    def test_valid_file_is_read(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "basis.json"
            p.write_text(json.dumps({"beta": 0.00081}))
            self.assertAlmostEqual(self._load_basis(p), 0.00081)

    def test_corrupt_or_nonpositive_means_zero(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "basis.json"
            p.write_text("{not json")
            self.assertEqual(self._load_basis(p), 0.0)
            p.write_text(json.dumps({"beta": -0.001}))
            self.assertEqual(self._load_basis(p), 0.0)
            p.write_text(json.dumps({"beta": "abc"}))
            self.assertEqual(self._load_basis(p), 0.0)


if __name__ == "__main__":
    unittest.main()
