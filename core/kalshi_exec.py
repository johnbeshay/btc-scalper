"""
Authenticated Kalshi client. Demo environment only.

THE ENDPOINT IS NOT CONFIGURABLE
--------------------------------
`BASE` below is the demo host, hardcoded. There is no --prod flag, no
environment variable, and no config key that points this at production. That
is deliberate.

Flags get typed by accident. A flag that moves real money is one shell-history
arrow-up away from being used by mistake at the wrong moment, and the moment
you would reach for it is exactly the moment your judgement is worst - after a
good run, or chasing a bad one.

Switching to production should require opening this file, reading the comment,
and changing a constant. That is a deliberate act with a diff attached, which
is the correct weight for the decision.

Before you make that edit, the bar from the project plan is: Phase A positive
out-of-sample AND Phase B running clean for days. As of the last scoring run
that bar is nowhere close - skill versus market was negative in every distance
band, the agent ablation was measuring correlated ladder rows as if they were
independent windows, and the adjuster fixes had two windows of evidence behind
them. None of that is a reason to be discouraged. It is a reason for this
constant to stay as it is.

Demo accounts are separate from production: separate signup, separate API key,
separate RSA keypair. Demo credentials do not work against production and vice
versa, which is a second layer of protection under this one.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.kalshi_auth import SigningError, auth_headers, load_private_key

# --- the line between practice and money ----------------------------------
BASE = "https://external-api.demo.kalshi.co/trade-api/v2"
# Production is https://external-api.kalshi.com/trade-api/v2
# Read the module docstring before you change this.
# --------------------------------------------------------------------------

IS_DEMO = "demo" in BASE
TIMEOUT = 15

# Credentials follow the endpoint, never a flag. Editing BASE is the only way
# to reach production, and it also switches which key file gets loaded, so a
# demo key can never be sent to production or the other way round by accident.
CREDS_FILENAME = ("kalshi-demo-credentials.json" if IS_DEMO
                  else "kalshi-prod-credentials.json")


class KalshiError(RuntimeError):
    """An API call failed."""


def to_centicents(dollars: float) -> int:
    """
    Dollars to centicents (hundredths of a cent). $1.00 -> 10000.

    Kalshi's transfer endpoint is the only place in this codebase using this
    unit; everywhere else is cents or dollars. One conversion, one test.
    """
    return int(round(dollars * 10000))


def to_v2(side: str, action: str, price_cents: int) -> tuple[str, int]:
    """
    Translate yes/no + buy/sell into the V2 single-book form.

    V2 quotes the YES leg only, so NO exposure becomes the opposite action
    on YES at the complement price:

        buy  yes  ->  bid at p
        sell yes  ->  ask at p
        buy  no   ->  ask at 100 - p    (buying NO at p == selling YES at 1-p)
        sell no   ->  bid at 100 - p

    Returns (book_side, yes_price_cents).
    """
    if side == "yes":
        return ("bid" if action == "buy" else "ask", price_cents)
    return ("ask" if action == "buy" else "bid", 100 - price_cents)


@dataclass
class Credentials:
    key_id: str
    private_key_path: str

    @classmethod
    def from_file(cls, path: str | Path) -> "Credentials":
        """
        Load credentials from a JSON file:

            {"key_id": "...", "private_key_path": "kalshi-demo.pem"}

        Keep this file and the .pem out of git. Both belong in .gitignore.
        """
        path = Path(path)
        if not path.exists():
            raise KalshiError(
                f"no credentials at {path}.\n"
                f'  Create it as: {{"key_id": "...", '
                f'"private_key_path": "kalshi-demo.pem"}}\n'
                "  Demo credentials come from the demo site, not production."
            )
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise KalshiError(f"{path} is not valid JSON: {exc}") from exc

        missing = [k for k in ("key_id", "private_key_path") if not d.get(k)]
        if missing:
            raise KalshiError(f"{path} is missing: {', '.join(missing)}")
        return cls(key_id=d["key_id"], private_key_path=d["private_key_path"])


class DemoClient:
    """
    Minimal authenticated client. Reads, one order type, one cancel.

    No retries and no backoff. A failed order should surface immediately and
    let a human decide, rather than being re-sent by a loop that cannot see
    why it failed the first time.
    """

    def __init__(self, creds: Credentials, base: str = BASE):
        self.base = base.rstrip("/")
        self.key_id = creds.key_id
        self._key = load_private_key(creds.private_key_path)

    # -- plumbing ---------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None,
                 query: dict | None = None) -> dict:
        """
        `path` is the full API path, e.g. /trade-api/v2/portfolio/balance.

        `path` is what gets signed and must never carry a query string. Pass
        query parameters via `query` instead: they are appended to the URL
        after signing. Signing the query string is the single most common
        cause of an unexplained 401 here.
        """
        url = self.base.replace("/trade-api/v2", "") + path
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        payload = json.dumps(body).encode("utf-8") if body is not None else None

        try:
            headers = auth_headers(self.key_id, self._key, method, path)
        except SigningError as exc:
            raise KalshiError(str(exc)) from exc

        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"

        req = urllib.request.Request(
            url, data=payload, headers=headers, method=method.upper()
        )

        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code == 401:
                raise KalshiError(
                    f"401 unauthorized on {method} {path}.\n"
                    "  Usual causes, in order of likelihood:\n"
                    "    - system clock is off (the timestamp is signed)\n"
                    "    - production credentials against the demo host\n"
                    "    - the query string was included in the signed path\n"
                    f"  {detail}"
                ) from exc
            raise KalshiError(f"HTTP {exc.code} on {method} {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise KalshiError(f"could not reach {url}: {exc.reason}") from exc

        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise KalshiError(f"non-JSON reply from {path}: {raw[:200]}") from exc

    # -- reads ------------------------------------------------------------

    def balance(self) -> dict:
        return self._request("GET", "/trade-api/v2/portfolio/balance")

    def positions(self) -> dict:
        return self._request("GET", "/trade-api/v2/portfolio/positions")

    def orders(self) -> dict:
        return self._request("GET", "/trade-api/v2/portfolio/orders")

    def fills(self, limit: int = 200) -> dict:
        """
        Recent fills. Carries the fees that settlement revenue omits.

        The limit is a query parameter, which must NOT appear in the signed
        path - see kalshi_auth.signing_payload.
        """
        return self._request(
            "GET", "/trade-api/v2/portfolio/fills", query={"limit": limit}
        )

    def settlements(self, limit: int = 200) -> dict:
        """Settled markets, the source of realised P&L."""
        return self._request(
            "GET", "/trade-api/v2/portfolio/settlements", query={"limit": limit}
        )

    def market(self, ticker: str) -> dict:
        """
        One market. Carries `exchange_index`, which the docs call the
        authoritative source of truth for routing - read it off the market
        rather than inferring it from the category.
        """
        return self._request("GET", f"/trade-api/v2/markets/{ticker}")

    def order(self, order_id: str) -> dict:
        """One order's current state: fills so far, what is still resting."""
        return self._request("GET", f"/trade-api/v2/portfolio/orders/{order_id}")

    def series(self, series_ticker: str) -> dict:
        """
        Series metadata. Carries the fee terms for the series (fee type and
        multiplier) - the only fee source that is specific to KXBTC15M rather
        than to the exchange in general. Read it; do not assume it.
        """
        return self._request("GET", f"/trade-api/v2/series/{series_ticker}")

    def exchange_index_for(self, ticker: str) -> int | None:
        m = self.market(ticker)
        return (m.get("market", m) or {}).get("exchange_index")

    def transfer(self, *, dollars: float, src_shard: int, dst_shard: int,
                 instance: str = "event_contract") -> dict:
        """
        Move collateral between exchange shards.

        Collateral does not follow an order. Funds on shard 0 cannot back an
        order routed to shard 2; the rejection is `insufficient_shard_balance`
        and reads identically to having no money at all.

        `amount` is in CENTICENTS - one hundredth of a cent. $1.00 is 10,000.
        Getting this wrong by a factor of 100 is the obvious failure here, so
        the conversion happens in one place and is tested.

        Kalshi warns that cross-shard transfers run in up to three non-atomic
        steps and that completed steps are not rolled back if a later one
        fails, which can strand funds on either side. Check the balance
        breakdown after every transfer rather than assuming it landed.
        """
        if dollars <= 0:
            raise KalshiError(f"amount must be positive, got {dollars}")
        body = {
            "source": instance,
            "destination": instance,
            "amount": to_centicents(dollars),
            "source_exchange_shard": src_shard,
            "destination_exchange_shard": dst_shard,
        }
        return self._request(
            "POST", "/trade-api/v2/portfolio/intra_exchange_instance_transfer",
            body,
        )

    def whoami(self) -> dict:
        """Cheapest call that proves auth works end to end."""
        return self.balance()

    # -- writes -----------------------------------------------------------

    def place_limit(self, *, ticker: str, side: str, action: str,
                    count: int, price_cents: int,
                    client_order_id: str,
                    time_in_force: str = "good_till_canceled",
                    exchange_index: int | None = None,
                    post_only: bool = False) -> dict:
        """
        One limit order, via the V2 endpoint.

        THE V2 SHAPE
        ------------
        V2 quotes everything from the YES leg. `side` is bid or ask:

            bid  = buy YES
            ask  = sell YES

        and selling YES is economically the same as buying NO at 1 - price.
        So a NO position is expressed as an ask on YES at the complement
        price. `to_v2()` below does that translation and is tested on all
        four combinations, because getting it backwards buys the opposite
        of what you meant at a plausible-looking price - a mistake that
        would not raise anything.

        Count and price are fixed-point STRINGS in dollars, not integer
        cents: "1.00" and "0.23". Sending ints here is accepted by json and
        rejected by the exchange.

        Limit only, never market. The limit price is the one guarantee
        available.

        POST ONLY
        ---------
        `post_only=True` asks the exchange to reject the order rather than
        let it match immediately. That is what makes a maker order a maker
        order: without it, a book that moves between reading and sending can
        turn a resting bid into a taker fill that pays the fee. The field
        name was taken from the documentation; confirm on demo that an order
        priced through the book is REJECTED, not filled, before trusting it.
        """
        if side not in ("yes", "no"):
            raise KalshiError(f"side must be yes or no, got {side!r}")
        if action not in ("buy", "sell"):
            raise KalshiError(f"action must be buy or sell, got {action!r}")
        if not (1 <= price_cents <= 99):
            raise KalshiError(f"price must be 1-99 cents, got {price_cents}")
        if count < 1:
            raise KalshiError(f"count must be >= 1, got {count}")
        if time_in_force not in ("fill_or_kill", "good_till_canceled",
                                 "immediate_or_cancel"):
            raise KalshiError(f"bad time_in_force: {time_in_force!r}")

        book_side, yes_price_cents = to_v2(side, action, price_cents)

        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": book_side,
            "count": f"{count}.00",
            "price": f"{yes_price_cents / 100:.4f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": "taker_at_cross",
        }
        if exchange_index is not None:
            body["exchange_index"] = exchange_index
        if post_only:
            body["post_only"] = True
        return self._request("POST", "/trade-api/v2/portfolio/events/orders", body)

    def cancel(self, order_id: str) -> dict:
        """
        Cancel a resting order.

        NOTE: this is still the V1 path. The V2 migration notice named the
        create endpoint specifically; whether cancel moved too has not been
        confirmed against the API. If this returns HTTP 410 with a
        deprecated_v1 code, the same migration applies here and the path
        needs updating - do not assume it works because create does.
        """
        return self._request(
            "DELETE", f"/trade-api/v2/portfolio/orders/{order_id}"
        )


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
