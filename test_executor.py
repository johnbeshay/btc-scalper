"""
Tests for the executor's safety rails and request signing.

The rails are the only thing standing between a bug and a filled order, so
they are tested harder than the code they guard. Every rail gets a test that
it fires, and a test that it does not fire when it should not.

The signing tests cover the payload construction only - the parts that fail
silently with a 401 and no explanation. They do not require `cryptography`.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core import rails as R
from core.kalshi_auth import signing_payload

NOW = datetime(2026, 9, 13, 22, 40, 0, tzinfo=timezone.utc)
FRESH = (NOW - timedelta(seconds=3)).isoformat()


def base_kwargs(state, root, **over):
    kw = dict(
        rails=R.Rails(),
        state=state,
        window_id="20260913T2245",
        suppressed=False,
        quoted_at=FRESH,
        price_cents=40,
        count=1,
        root=root,
        now=NOW,
    )
    kw.update(over)
    return kw


class RailsTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = R.State(self.root / "state.json")

    def tearDown(self):
        self.tmp.cleanup()

    # -- the happy path ---------------------------------------------------

    def test_clean_order_is_allowed(self):
        d = R.check(**base_kwargs(self.state, self.root))
        self.assertTrue(d)
        self.assertEqual(d.reasons, [])
        self.assertEqual(d.why, "all rails clear")

    # -- 1. kill switch ---------------------------------------------------

    def test_kill_switch_blocks(self):
        (self.root / R.KILL_FILE).write_text("")
        d = R.check(**base_kwargs(self.state, self.root))
        self.assertFalse(d)
        self.assertTrue(any("kill switch" in r for r in d.reasons))

    def test_kill_switch_absent_by_default(self):
        self.assertFalse(R.kill_switch_active(self.root))

    # -- 2. daily loss cap ------------------------------------------------

    def test_daily_loss_cap_blocks_at_the_limit(self):
        self.state.record_pnl(-10.00, day=R.utc_day(NOW))
        d = R.check(**base_kwargs(self.state, self.root))
        self.assertFalse(d)
        self.assertTrue(any("daily loss cap" in r for r in d.reasons))

    def test_loss_under_the_cap_is_allowed(self):
        self.state.record_pnl(-9.99, day=R.utc_day(NOW))
        self.assertTrue(R.check(**base_kwargs(self.state, self.root)))

    def test_yesterdays_loss_does_not_block_today(self):
        self.state.record_pnl(-50.0, day="2026-09-12")
        self.assertTrue(R.check(**base_kwargs(self.state, self.root)))

    def test_pnl_survives_a_restart(self):
        path = self.root / "state.json"
        s1 = R.State(path)
        s1.record_pnl(-4.25, day=R.utc_day(NOW))
        s2 = R.State(path)
        self.assertAlmostEqual(s2.pnl_today(R.utc_day(NOW)), -4.25)

    def test_corrupt_state_file_resets_rather_than_raising(self):
        path = self.root / "bad.json"
        path.write_text("{not json")
        s = R.State(path)
        self.assertEqual(s.pnl_today(), 0.0)

    # -- 3. per-window cap ------------------------------------------------

    def test_second_order_in_a_window_is_blocked(self):
        self.state.record_order("20260913T2245")
        d = R.check(**base_kwargs(self.state, self.root))
        self.assertFalse(d)
        self.assertTrue(any("already has" in r for r in d.reasons))

    def test_a_different_window_is_unaffected(self):
        self.state.record_order("20260913T2230")
        self.assertTrue(R.check(**base_kwargs(self.state, self.root)))

    # -- 4. suppressed ----------------------------------------------------

    def test_suppressed_window_is_refused(self):
        d = R.check(**base_kwargs(self.state, self.root, suppressed=True))
        self.assertFalse(d)
        self.assertTrue(any("suppressed" in r for r in d.reasons))

    def test_suppressed_can_be_allowed_only_by_changing_rails(self):
        d = R.check(**base_kwargs(
            self.state, self.root, suppressed=True,
            rails=R.Rails(allow_suppressed=True),
        ))
        self.assertTrue(d)

    # -- 5. stale book ----------------------------------------------------

    def test_stale_book_is_refused(self):
        old = (NOW - timedelta(seconds=31)).isoformat()
        d = R.check(**base_kwargs(self.state, self.root, quoted_at=old))
        self.assertFalse(d)
        self.assertTrue(any("old" in r for r in d.reasons))

    def test_book_just_inside_the_bound_is_fine(self):
        ok = (NOW - timedelta(seconds=29)).isoformat()
        self.assertTrue(
            R.check(**base_kwargs(self.state, self.root, quoted_at=ok))
        )

    def test_missing_timestamp_is_refused(self):
        d = R.check(**base_kwargs(self.state, self.root, quoted_at=None))
        self.assertFalse(d)
        self.assertTrue(any("no book timestamp" in r for r in d.reasons))

    def test_future_timestamp_is_refused(self):
        ahead = (NOW + timedelta(seconds=60)).isoformat()
        d = R.check(**base_kwargs(self.state, self.root, quoted_at=ahead))
        self.assertFalse(d)
        self.assertTrue(any("future" in r for r in d.reasons))

    def test_naive_timestamp_is_treated_as_utc(self):
        naive = (NOW - timedelta(seconds=5)).replace(tzinfo=None).isoformat()
        self.assertTrue(
            R.check(**base_kwargs(self.state, self.root, quoted_at=naive))
        )

    # -- 6. notional ------------------------------------------------------

    def test_notional_cap_blocks_a_large_order(self):
        d = R.check(**base_kwargs(self.state, self.root, count=20, price_cents=40))
        self.assertFalse(d)
        self.assertTrue(any("notional" in r for r in d.reasons))

    def test_notional_exactly_at_the_cap_is_allowed(self):
        # 10 x 50c = $5.00, the default cap
        self.assertTrue(
            R.check(**base_kwargs(self.state, self.root,
                                  count=10, price_cents=50))
        )

    def test_price_outside_one_to_ninety_nine_is_refused(self):
        for bad in (0, 100, -5):
            d = R.check(**base_kwargs(self.state, self.root, price_cents=bad))
            self.assertFalse(d, bad)

    def test_zero_count_is_refused(self):
        d = R.check(**base_kwargs(self.state, self.root, count=0))
        self.assertFalse(d)

    # -- reporting --------------------------------------------------------

    def test_every_failing_rail_is_reported_not_just_the_first(self):
        (self.root / R.KILL_FILE).write_text("")
        self.state.record_pnl(-99.0, day=R.utc_day(NOW))
        self.state.record_order("20260913T2245")
        d = R.check(**base_kwargs(
            self.state, self.root, suppressed=True, quoted_at=None,
            count=99, price_cents=99,
        ))
        self.assertFalse(d)
        self.assertGreaterEqual(len(d.reasons), 6)

    def test_decision_is_falsy_when_blocked(self):
        (self.root / R.KILL_FILE).write_text("")
        self.assertFalse(bool(R.check(**base_kwargs(self.state, self.root))))


class SigningPayloadTest(unittest.TestCase):
    """The parts of auth that fail as an unexplained 401."""

    def test_payload_is_timestamp_method_path(self):
        self.assertEqual(
            signing_payload(1757800000000, "GET", "/trade-api/v2/portfolio/balance"),
            "1757800000000GET/trade-api/v2/portfolio/balance",
        )

    def test_method_is_upper_cased(self):
        self.assertEqual(
            signing_payload(1, "post", "/x"), signing_payload(1, "POST", "/x")
        )

    def test_query_string_is_stripped(self):
        self.assertEqual(
            signing_payload(1, "GET", "/trade-api/v2/markets?limit=10"),
            "1GET/trade-api/v2/markets",
        )

    def test_path_without_query_is_untouched(self):
        self.assertEqual(
            signing_payload(1, "GET", "/trade-api/v2/markets"),
            "1GET/trade-api/v2/markets",
        )


class EndpointTest(unittest.TestCase):
    """The endpoint constant is load-bearing; assert it points at demo."""

    def test_base_is_the_demo_host(self):
        from core.kalshi_exec import BASE, IS_DEMO
        self.assertIn("demo", BASE)
        self.assertTrue(IS_DEMO)

    def test_no_flag_can_reach_production(self):
        import executor
        src = Path(executor.__file__).read_text()
        self.assertNotIn("--prod", src)
        self.assertNotIn("elections.kalshi.com", src)


class CredentialsTest(unittest.TestCase):

    def test_missing_file_explains_itself(self):
        from core.kalshi_exec import Credentials, KalshiError
        with self.assertRaises(KalshiError) as ctx:
            Credentials.from_file("/nonexistent/creds.json")
        self.assertIn("no credentials", str(ctx.exception))

    def test_incomplete_file_names_what_is_missing(self):
        from core.kalshi_exec import Credentials, KalshiError
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"key_id": "abc"}))
            with self.assertRaises(KalshiError) as ctx:
                Credentials.from_file(p)
            self.assertIn("private_key_path", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
