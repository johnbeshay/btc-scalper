"""
Request signing for the Kalshi trade API.

This is the ONLY module in the project that imports `cryptography`. Everything
in the logging and scoring path - logger.py, score.py, replay.py, core.learning
- must stay standard-library only so it keeps running on a bare Python. If you
find yourself importing this module from any of those, stop: the dependency
has leaked.

THE SIGNATURE
-------------
Kalshi authenticates each request with an RSA-PSS signature over:

    <timestamp_ms><HTTP_METHOD><path>

The path excludes the query string. `/trade-api/v2/portfolio/orders?foo=1` is
signed as `/trade-api/v2/portfolio/orders`. Signing the query string is the
most common way to get a 401 here.

Three headers go on every authenticated request:

    KALSHI-ACCESS-KEY         your API key id
    KALSHI-ACCESS-SIGNATURE   base64 of the PSS signature
    KALSHI-ACCESS-TIMESTAMP   the same millisecond timestamp that was signed

The timestamp is signed as well as sent, so a stale clock produces a signature
mismatch rather than a clear error. If you get persistent 401s with a key you
believe is good, check the system clock before anything else.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path


class SigningError(RuntimeError):
    """Raised when a key cannot be loaded or a signature cannot be produced."""


def _require_cryptography():
    """
    Import cryptography lazily and fail with something readable.

    Lazy so that merely importing this module - which some test collectors do
    automatically - does not hard-fail on a machine that never intends to
    trade.
    """
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SigningError(
            "the `cryptography` package is required for authenticated Kalshi "
            "requests. Install it in a virtualenv so the logger and scorer "
            "stay dependency-free:\n"
            "    python -m venv .venv\n"
            "    .venv\\Scripts\\pip install cryptography"
        ) from exc
    return hashes, serialization, padding, rsa


def load_private_key(path: str | Path, password: bytes | None = None):
    """
    Load an RSA private key from a PEM file.

    Kalshi issues the key once, at API-key creation, and never shows it again.
    Treat the file as a secret: it is equivalent to your trading password.
    """
    hashes, serialization, padding, rsa = _require_cryptography()

    path = Path(path)
    if not path.exists():
        raise SigningError(f"private key not found at {path}")

    try:
        key = serialization.load_pem_private_key(
            path.read_bytes(), password=password
        )
    except Exception as exc:
        raise SigningError(
            f"could not read {path} as a PEM private key. If the key is "
            f"passphrase-protected, pass the password. ({exc})"
        ) from exc

    if not isinstance(key, rsa.RSAPrivateKey):
        raise SigningError(
            "the key is not an RSA key. Kalshi issues RSA keys; an EC or "
            "Ed25519 key means you are using the wrong file."
        )
    return key


def signing_payload(timestamp_ms: int, method: str, path: str) -> str:
    """
    Build the exact string Kalshi expects to have been signed.

    Separated out and tested on its own because every field here is a silent
    failure if it is wrong: a lowercase method, a trailing query string, or
    seconds instead of milliseconds all produce a 401 with no hint as to
    which.
    """
    if "?" in path:
        path = path.split("?", 1)[0]
    return f"{timestamp_ms}{method.upper()}{path}"


def sign(private_key, timestamp_ms: int, method: str, path: str) -> str:
    """Return the base64 PSS signature for one request."""
    hashes, serialization, padding, rsa = _require_cryptography()

    message = signing_payload(timestamp_ms, method, path).encode("utf-8")
    try:
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256().digest_size,
            ),
            hashes.SHA256(),
        )
    except Exception as exc:
        raise SigningError(f"signing failed: {exc}") from exc

    return base64.b64encode(signature).decode("utf-8")


def auth_headers(key_id: str, private_key, method: str, path: str,
                 timestamp_ms: int | None = None) -> dict[str, str]:
    """
    Build the three auth headers for one request.

    `timestamp_ms` is injectable so tests can pin it; in normal use it is the
    current time and must be within Kalshi's tolerance of their clock.
    """
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)

    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sign(private_key, timestamp_ms, method, path),
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
    }
