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


class V2OrderShapeTest(unittest.TestCase):
    """
    The V2 endpoint quotes the YES leg only. Getting the translation
    backwards would buy the opposite of what was intended at a
    plausible-looking price, and nothing would raise.
    """

    def test_buy_yes_is_a_bid_at_the_same_price(self):
        from core.kalshi_exec import to_v2
        self.assertEqual(to_v2("yes", "buy", 23), ("bid", 23))

    def test_sell_yes_is_an_ask_at_the_same_price(self):
        from core.kalshi_exec import to_v2
        self.assertEqual(to_v2("yes", "sell", 23), ("ask", 23))

    def test_buy_no_is_an_ask_at_the_complement(self):
        from core.kalshi_exec import to_v2
        self.assertEqual(to_v2("no", "buy", 23), ("ask", 77))

    def test_sell_no_is_a_bid_at_the_complement(self):
        from core.kalshi_exec import to_v2
        self.assertEqual(to_v2("no", "sell", 23), ("bid", 77))

    def test_complement_round_trips(self):
        from core.kalshi_exec import to_v2
        for p in range(1, 100):
            _, yes_p = to_v2("no", "buy", p)
            self.assertEqual(yes_p, 100 - p)

    def test_order_path_is_the_v2_endpoint(self):
        from pathlib import Path
        import core.kalshi_exec as ke
        src = Path(ke.__file__).read_text()
        self.assertIn("/trade-api/v2/portfolio/events/orders", src)

    def test_count_and_price_are_fixed_point_strings(self):
        """V2 rejects integer cents; count and price must be dollar strings."""
        sent = {}

        class FakeClient(ke_module.DemoClient):
            def __init__(self):
                self.base = "https://demo-api.kalshi.co/trade-api/v2"
                self.key_id = "k"
                self._key = None

            def _request(self, method, path, body=None, query=None):
                sent["method"], sent["path"], sent["body"] = method, path, body
                return {}

        FakeClient().place_limit(
            ticker="T", side="yes", action="buy", count=2,
            price_cents=23, client_order_id="cid",
        )
        self.assertEqual(sent["path"], "/trade-api/v2/portfolio/events/orders")
        self.assertEqual(sent["body"]["count"], "2.00")
        self.assertEqual(sent["body"]["price"], "0.2300")
        self.assertEqual(sent["body"]["side"], "bid")
        self.assertEqual(sent["body"]["time_in_force"], "good_till_canceled")
        self.assertEqual(
            sent["body"]["self_trade_prevention_type"], "taker_at_cross"
        )

    def test_buying_no_sends_the_complement_price(self):
        sent = {}

        class FakeClient(ke_module.DemoClient):
            def __init__(self):
                self.base = "x"
                self.key_id = "k"
                self._key = None

            def _request(self, method, path, body=None, query=None):
                sent["body"] = body
                return {}

        FakeClient().place_limit(
            ticker="T", side="no", action="buy", count=1,
            price_cents=23, client_order_id="cid",
        )
        self.assertEqual(sent["body"]["side"], "ask")
        self.assertEqual(sent["body"]["price"], "0.7700")

    def test_bad_time_in_force_is_rejected(self):
        import core.kalshi_exec as ke

        class FakeClient(ke.DemoClient):
            def __init__(self):
                self.base = "x"
                self.key_id = "k"
                self._key = None

        with self.assertRaises(ke.KalshiError):
            FakeClient().place_limit(
                ticker="T", side="yes", action="buy", count=1,
                price_cents=23, client_order_id="c", time_in_force="GTT",
            )


import core.kalshi_exec as ke_module  # noqa: E402


class ShardRoutingTest(unittest.TestCase):
    """
    Collateral is per shard and does not follow the order. Getting the
    centicent conversion or the routing index wrong produces a rejection
    that reads exactly like having no money.
    """

    def test_dollars_to_centicents(self):
        from core.kalshi_exec import to_centicents
        self.assertEqual(to_centicents(1.00), 10000)
        self.assertEqual(to_centicents(0.01), 100)
        self.assertEqual(to_centicents(15.00), 150000)
        self.assertEqual(to_centicents(20.0), 200000)

    def test_centicents_rounds_rather_than_truncates(self):
        from core.kalshi_exec import to_centicents
        self.assertEqual(to_centicents(0.0001), 1)
        self.assertEqual(to_centicents(1.2345), 12345)

    def test_transfer_body_shape(self):
        sent = {}

        class FakeClient(ke_module.DemoClient):
            def __init__(self):
                self.base = "x"
                self.key_id = "k"
                self._key = None

            def _request(self, method, path, body=None, query=None):
                sent["path"], sent["body"] = path, body
                return {"transfer_id": "t1"}

        FakeClient().transfer(dollars=15.0, src_shard=0, dst_shard=2)
        self.assertEqual(
            sent["path"],
            "/trade-api/v2/portfolio/intra_exchange_instance_transfer",
        )
        self.assertEqual(sent["body"]["amount"], 150000)
        self.assertEqual(sent["body"]["source_exchange_shard"], 0)
        self.assertEqual(sent["body"]["destination_exchange_shard"], 2)
        self.assertEqual(sent["body"]["source"], "event_contract")
        self.assertEqual(sent["body"]["destination"], "event_contract")

    def test_transfer_rejects_non_positive(self):
        import core.kalshi_exec as ke

        class FakeClient(ke.DemoClient):
            def __init__(self):
                self.base = "x"
                self.key_id = "k"
                self._key = None

        for bad in (0, -5.0):
            with self.assertRaises(ke.KalshiError):
                FakeClient().transfer(dollars=bad, src_shard=0, dst_shard=2)

    def test_exchange_index_is_sent_on_the_order(self):
        sent = {}

        class FakeClient(ke_module.DemoClient):
            def __init__(self):
                self.base = "x"
                self.key_id = "k"
                self._key = None

            def _request(self, method, path, body=None, query=None):
                sent["body"] = body
                return {}

        FakeClient().place_limit(
            ticker="T", side="yes", action="buy", count=1, price_cents=50,
            client_order_id="c", exchange_index=2,
        )
        self.assertEqual(sent["body"]["exchange_index"], 2)

    def test_exchange_index_omitted_when_not_known(self):
        sent = {}

        class FakeClient(ke_module.DemoClient):
            def __init__(self):
                self.base = "x"
                self.key_id = "k"
                self._key = None

            def _request(self, method, path, body=None, query=None):
                sent["body"] = body
                return {}

        FakeClient().place_limit(
            ticker="T", side="yes", action="buy", count=1, price_cents=50,
            client_order_id="c",
        )
        self.assertNotIn("exchange_index", sent["body"])

    def test_exchange_index_read_off_the_market(self):
        class FakeClient(ke_module.DemoClient):
            def __init__(self):
                self.base = "x"
                self.key_id = "k"
                self._key = None

            def _request(self, method, path, body=None, query=None):
                return {"market": {"ticker": "T", "exchange_index": 2}}

        self.assertEqual(FakeClient().exchange_index_for("T"), 2)

    def test_shard_zero_is_a_real_answer_not_a_missing_one(self):
        class FakeClient(ke_module.DemoClient):
            def __init__(self):
                self.base = "x"
                self.key_id = "k"
                self._key = None

            def _request(self, method, path, body=None, query=None):
                return {"market": {"exchange_index": 0}}

        self.assertEqual(FakeClient().exchange_index_for("T"), 0)
