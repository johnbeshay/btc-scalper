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
BASE = "https://demo-api.kalshi.co/trade-api/v2"
# Production is https://api.elections.kalshi.com/trade-api/v2
# Read the module docstring before you change this.
# --------------------------------------------------------------------------

IS_DEMO = "demo" in BASE
TIMEOUT = 15


class KalshiError(RuntimeError):
    """An API call failed."""


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

    def whoami(self) -> dict:
        """Cheapest call that proves auth works end to end."""
        return self.balance()

    # -- writes -----------------------------------------------------------

    def place_limit(self, *, ticker: str, side: str, action: str,
                    count: int, price_cents: int,
                    client_order_id: str) -> dict:
        """
        One limit order.

        Limit only, never market. A market order on a book with no depth -
        and KXBTC15M has shown zero volume every time it has been checked -
        can fill anywhere. The limit price is the one guarantee available.
        """
        if side not in ("yes", "no"):
            raise KalshiError(f"side must be yes or no, got {side!r}")
        if action not in ("buy", "sell"):
            raise KalshiError(f"action must be buy or sell, got {action!r}")
        if not (1 <= price_cents <= 99):
            raise KalshiError(f"price must be 1-99 cents, got {price_cents}")
        if count < 1:
            raise KalshiError(f"count must be >= 1, got {count}")

        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": side,
            "action": action,
            "count": count,
            "type": "limit",
            f"{side}_price": price_cents,
        }
        return self._request("POST", "/trade-api/v2/portfolio/orders", body)

    def cancel(self, order_id: str) -> dict:
        return self._request(
            "DELETE", f"/trade-api/v2/portfolio/orders/{order_id}"
        )


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
