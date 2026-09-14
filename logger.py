"""
Record what the model predicts, then what actually happened.

    python logger.py                  # run it and leave it running
    python logger.py --once           # one window, for testing
    python logger.py --horizons 12,8,4

Writes newline-delimited JSON to predictions.jsonl. Two kinds of line:

    {"type": "prediction", ...}   written partway through a window
    {"type": "outcome", ...}      written after that window closes

They join on window_id. Two lines instead of one because the process may die
between prediction and close, and a half-written record that never resolves
is still evidence - it tells the scorer to ignore that window rather than
silently losing the prediction.

No Kalshi credentials needed. Market data is public and read-only.

Each prediction line carries a strike ladder. When Kalshi is reachable, the
ladder is the exchange's real listed strikes for that window, and every
strike records the top of book at the same instant the model priced it.
That is what lets score.py answer the question that matters: does the model
beat the price it would have paid? When Kalshi is unreachable the logger
falls back to a synthetic ladder around spot, marks the record accordingly,
and keeps going - calibration can still be measured, edge cannot.

SPOT COMES FROM THE TICKER, NOT THE CANDLES
-------------------------------------------
Coinbase's /candles route serves cached completed bars and runs several
minutes behind the market. Measured directly: the newest bar was 5.3 minutes
old and $88 away from the live price, against a typical 15-minute sigma of
about $60. Pricing a window from that bar means answering a question about a
price that no longer exists - and worse, the exchange has seen the move that
the model has not, so every disagreement with the book is contaminated by
information the model simply did not have.

Every record written before schema 3 was priced this way. Treat pre-schema-3
windows as measuring a model reading a five-minute-old price, which is not
the model anyone would choose to run.

Candles are still used for volatility. Lag does not matter there: the
question is how much price has been moving, not where it is right now.

THE OUTCOME COMES FROM KALSHI, NOT COINBASE
-------------------------------------------
There is a third record type:

    {"type": "settlement", "ticker": ..., "result": "yes"|"no", ...}

It carries how the exchange actually settled the market. Kalshi resolves
these on CF Benchmarks' BRTI as a 60-second average, not on a Coinbase
candle close. Compared against real settlements, the candle-close proxy was
wrong on one window in five, and the errors sat near the money. The scorer
now grades against the settlement line when one exists and only falls back
to the close when it does not.

The outcome line is still written - it still records where price went - but
it no longer decides who was right.

Records carry "schema": 3. Older lines are still readable; the schema number
is what distinguishes stale-spot records from live-spot ones.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.adjusters import Context, build_estimate
from core.feed import CoinbaseFeed, FeedError
from core.kalshi import estimate_vol, prob_above
from core.kalshi_api import DEFAULT_SERIES, KalshiError, KalshiMarketData

LOG = Path(__file__).parent / "predictions.jsonl"
SCHEMA = 3
_stop = False


def _handle_stop(signum, frame):
    global _stop
    _stop = True
    print("\n  finishing current window, then stopping...")


def window_close(now: datetime, minutes: int = 15) -> datetime:
    elapsed = (now.minute % minutes) * 60 + now.second + now.microsecond / 1e6
    return now + timedelta(seconds=minutes * 60 - elapsed)


def window_id(close: datetime) -> str:
    return close.strftime("%Y%m%dT%H%M")


def append(path: Path, record: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


class Recorder:
    def __init__(self, path: Path = LOG, horizons=(12, 8, 4), strikes: int = 9,
                 market: KalshiMarketData | None = None, use_market: bool = True):
        self.path = path
        self.horizons = sorted(horizons, reverse=True)
        self.strikes = strikes
        self.feed = CoinbaseFeed()
        self.market = market if market is not None else (
            KalshiMarketData() if use_market else None
        )
        self._hourly = None
        self._hourly_at = 0.0
        self._market_warned = False
        self._spot_warned = False
        self._settle_warned = False
        self._tickers: dict[str, set[str]] = {}   # window_id -> tickers seen
        self.last_ladder = None  # "kalshi" or "synthetic", for the console line

    def _book(self, close: datetime, now: datetime):
        """
        The Kalshi ladder for this window, or None to fall back.

        A failure here is never fatal. It is printed once per process so the
        user knows edge is not being measured, then the logger carries on
        with the synthetic ladder.
        """
        if self.market is None:
            return None
        try:
            quotes = self.market.quotes_for_window(close)
        except KalshiError as exc:
            if not self._market_warned:
                print(f"  kalshi unavailable, using synthetic ladder: {exc}",
                      file=sys.stderr)
                print("  (calibration still measured; edge vs market will not be)",
                      file=sys.stderr)
                self._market_warned = True
            return None
        if not quotes:
            if not self._market_warned:
                print(f"  no {self.market.series} contracts found for this window; "
                      "using synthetic ladder", file=sys.stderr)
                print("  (check the series ticker: python kalshi_book.py discover)",
                      file=sys.stderr)
                self._market_warned = True
            return None
        return quotes

    def spot(self, candles) -> tuple[float, str]:
        """
        The live price, and where it came from.

        /ticker is real time. /candles is cached and several minutes behind,
        so it is only a fallback - a stale spot is better than no reading at
        all, but the record says which was used so the scorer can separate
        them rather than silently mixing two different models.

        AttributeError is caught alongside FeedError because an injected feed
        (a test double, or some future source) may legitimately have no
        ticker endpoint. That is a fallback, not a crash. It is never silent:
        the record carries spot_source="candle" and the console prints
        STALE SPOT, so a real feed that lost its spot_price would be visible
        within one window rather than quietly degrading every prediction.
        """
        try:
            return self.feed.spot_price(), "ticker"
        except (FeedError, AttributeError) as exc:
            if not self._spot_warned:
                print(f"  ticker unavailable, falling back to the (stale) "
                      f"candle close: {exc}", file=sys.stderr)
                self._spot_warned = True
            return candles[-1].close, "candle"

    def hourly(self):
        if self._hourly and time.time() - self._hourly_at < 1800:
            return self._hourly
        try:
            self._hourly = self.feed.candles(granularity=3600, limit=300)
            self._hourly_at = time.time()
        except FeedError:
            pass
        return self._hourly

    def snapshot(self, close: datetime, horizon: int,
                 now: datetime | None = None) -> dict | None:
        """
        Take one reading and write a prediction line.

        `now` is injectable so this can be driven by a simulated clock. Taking
        it from the wall clock unconditionally made the function impossible to
        test and, worse, let the real remaining time drift away from the
        horizon whenever the loop ran late - which silently shrinks sigma and
        makes the model look far more confident than it should be.
        """
        try:
            candles = self.feed.candles(granularity=60, limit=120)
        except FeedError as exc:
            print(f"  feed error, skipping this reading: {exc}", file=sys.stderr)
            return None

        if len(candles) < 40:
            return None

        now = now or datetime.now(timezone.utc)
        minutes_left = max((close - now).total_seconds() / 60, 0.1)

        # If the loop was late the real gap no longer matches the horizon we
        # meant to sample. Record both so the scorer can group honestly.
        actual_horizon = round(minutes_left, 2)

        # Spot from the live ticker; candles only for volatility. See the
        # module docstring - the candle feed runs minutes behind and pricing
        # from it hands the exchange a free information advantage.
        spot, spot_source = self.spot(candles)
        candle_spot = candles[-1].close
        candle_age_min = round((now - candles[-1].ts).total_seconds() / 60, 2)

        vol = estimate_vol(candles, 1.0)

        ctx = Context(
            spot=spot,
            candles=candles,
            vol=vol,
            minutes_left=minutes_left,
            now=now,
            hourly=self.hourly(),
        )
        est = build_estimate(ctx)
        sigma = est.final_sigma
        if sigma <= 0:
            return None

        def price(strike: float) -> tuple[float, float]:
            p = prob_above(spot * (1 + est.drift_pct / 100), strike, sigma)
            sd = abs(math.log(spot / strike) / sigma) if strike != spot else 0.0
            return p, sd

        # Preferred: the exchange's own strikes, each with the book at this
        # instant. Only contracts that actually exist can be compared to a
        # price, so a synthetic strike is worthless for measuring edge.
        quotes = self._book(close, now)
        preds = []
        if quotes:
            ladder = "kalshi"
            wid = window_id(close)
            self._tickers.setdefault(wid, set()).update(q.ticker for q in quotes)
            for q in quotes:
                p, sd = price(q.strike)
                p_yes = p if q.yes_direction == "above" else 1 - p
                preds.append(
                    {
                        "strike": q.strike,
                        "p_above": round(p, 6),
                        "p_yes": round(p_yes, 6),
                        "sigmas_out": round(sd, 3),
                        "market": q.to_dict(),
                    }
                )
        else:
            # Fallback: a ladder spanning roughly plus or minus two standard
            # deviations, so the log covers near-money and far strikes. The
            # fat-tail claim is only testable if the far ones are recorded.
            ladder = "synthetic"
            step = spot * sigma * 0.5
            half = self.strikes // 2
            for i in range(-half, half + 1):
                strike = round((spot + i * step) / 10) * 10
                p, sd = price(strike)
                preds.append(
                    {
                        "strike": strike,
                        "p_above": round(p, 6),
                        "p_yes": round(p, 6),
                        "sigmas_out": round(sd, 3),
                        "market": None,
                    }
                )
        self.last_ladder = ladder

        record = {
            "type": "prediction",
            "schema": SCHEMA,
            "ladder": ladder,
            "market_series": self.market.series if self.market else None,
            "window_id": window_id(close),
            "horizon_min": horizon,
            "actual_minutes_left": actual_horizon,
            "at": now.isoformat(),
            "window_close": close.isoformat(),
            "spot": round(spot, 2),
            "spot_source": spot_source,
            # Kept so the size of the old bug stays measurable in the log
            # itself rather than only in this docstring.
            "candle_spot": round(candle_spot, 2),
            "candle_age_min": candle_age_min,
            "sigma": round(sigma, 8),
            "base_sigma": round(est.base_sigma, 8),
            "vol_change_pct": round(est.total_vol_change_pct, 2),
            "drift_pct": round(est.drift_pct, 5),
            "suppressed": est.suppressed,
            "agents": {
                a.agent: {
                    "headline": a.headline,
                    "vol_multiplier": round(a.vol_multiplier, 4),
                    "suppress": a.suppress,
                }
                for a in est.adjustments
            },
            "predictions": preds,
        }
        append(self.path, record)
        return record

    def resolve(self, close: datetime, now: datetime | None = None) -> dict | None:
        """After the window closes, record where price actually ended."""
        try:
            candles = self.feed.candles(granularity=60, limit=10)
        except FeedError as exc:
            print(f"  feed error, cannot resolve window: {exc}", file=sys.stderr)
            return None

        # The candle whose timestamp is the close minute, or the closest
        # available. Using "latest price now" would drift if the loop is late.
        #
        # Note this path deliberately still uses candles: resolving asks where
        # price WAS at a past instant, which is exactly what a completed bar
        # records. The lag that ruins a live spot is harmless here, and the
        # `trustworthy` flag already catches a bar too far from the close.
        target = close - timedelta(minutes=1)
        best = min(candles, key=lambda c: abs((c.ts - target).total_seconds()))
        drift = abs((best.ts - target).total_seconds())

        record = {
            "type": "outcome",
            "window_id": window_id(close),
            "window_close": close.isoformat(),
            "close_price": round(best.close, 2),
            "candle_ts": best.ts.isoformat(),
            "seconds_off": round(drift),
            "trustworthy": drift <= 300,
        }
        append(self.path, record)
        return record

    def settle(self, close: datetime, attempts: int = 4,
               wait_s: float = 15.0) -> list[dict]:
        """
        Fetch how Kalshi settled this window's market(s) and record it.

        Kalshi has settled these within about ten seconds of close in
        practice, but that is not guaranteed, so this retries a few times.
        A miss is not fatal: `python backfill.py` fills any gap later, as
        long as the market is still in the API's window.
        """
        wid = window_id(close)
        tickers = self._tickers.pop(wid, set())
        if not tickers or self.market is None:
            return []

        written = []
        pending = set(tickers)
        for i in range(attempts):
            try:
                found = self.market.results_for(pending)
            except Exception as exc:
                if not self._settle_warned:
                    print(f"  settlement lookup failed: {exc}", file=sys.stderr)
                    self._settle_warned = True
                found = {}
            for t, res in found.items():
                rec = {
                    "type": "settlement",
                    "window_id": wid,
                    "ticker": t,
                    "result": res,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
                append(self.path, rec)
                written.append(rec)
            pending -= set(found)
            if not pending:
                break
            if i < attempts - 1 and self._sleep(wait_s):
                break
        return written

    def run_window(self) -> None:
        now = datetime.now(timezone.utc)
        close = window_close(now)
        wid = window_id(close)
        print(f"\n  window {wid}  (closes {close.strftime('%H:%M:%S')} UTC)")

        for h in self.horizons:
            fire_at = close - timedelta(minutes=h)
            wait = (fire_at - datetime.now(timezone.utc)).total_seconds()
            if wait < -30:
                continue  # we started mid-window and missed this reading
            if wait > 0:
                if self._sleep(wait):
                    return
            rec = self.snapshot(close, h)
            if rec:
                flag = "  SUPPRESSED" if rec["suppressed"] else ""
                book = (f"  book {len(rec['predictions'])} strikes"
                        if rec["ladder"] == "kalshi" else "  no book")
                stale = "" if rec["spot_source"] == "ticker" else "  STALE SPOT"
                print(
                    f"    T-{h:<2}  spot ${rec['spot']:>10,.2f}   "
                    f"sigma {rec['sigma'] * 100:.3f}%{book}{flag}{stale}"
                )

        wait = (close - datetime.now(timezone.utc)).total_seconds() + 20
        if wait > 0 and self._sleep(wait):
            return

        out = self.resolve(close)
        if out:
            mark = "" if out["trustworthy"] else "   (timing off, will be excluded)"
            print(f"    close ${out['close_price']:,.2f}{mark}")

        settled = self.settle(close)
        if settled:
            print("    kalshi settled " + ", ".join(
                f"{r['result'].upper()}" for r in settled))
        elif self._tickers.get(wid) is None and self.market is not None:
            print("    settlement not available yet (backfill.py will catch it)")

    def _sleep(self, seconds: float) -> bool:
        """Sleep in slices so Ctrl-C is responsive. True means stop."""
        end = time.time() + seconds
        while time.time() < end:
            if _stop:
                return True
            time.sleep(min(1.0, end - time.time()))
        return False


def main() -> int:
    p = argparse.ArgumentParser(description="Record predictions and outcomes")
    p.add_argument("--once", action="store_true", help="one window then exit")
    p.add_argument("--horizons", default="12,8,4",
                   help="minutes before close to take readings")
    p.add_argument("--strikes", type=int, default=9,
                   help="synthetic ladder size, used only when Kalshi is unreachable")
    p.add_argument("--out", default=str(LOG))
    p.add_argument("--series", default=DEFAULT_SERIES,
                   help="Kalshi series ticker for 15-minute BTC markets")
    p.add_argument("--no-market", action="store_true",
                   help="skip Kalshi entirely and log the synthetic ladder")
    args = p.parse_args()

    signal.signal(signal.SIGINT, _handle_stop)

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    market = None if args.no_market else KalshiMarketData(series=args.series)
    rec = Recorder(Path(args.out), horizons, args.strikes,
                   market=market, use_market=not args.no_market)

    print(f"  Logging to {args.out}")
    print(f"  Readings at T-{', T-'.join(str(h) for h in rec.horizons)} minutes")
    print("  Spot from the live ticker; candles for volatility only")
    print("  Outcomes from Kalshi settlement, not the Coinbase close")
    if market:
        print(f"  Kalshi book from series {market.series} (read-only, no login)")
    else:
        print("  Kalshi book OFF - calibration only, edge will not be measurable")
    print("  Leave this running. Ctrl-C to stop after the current window.")
    print("  Check progress any time with:  python score.py")

    try:
        while not _stop:
            rec.run_window()
            if args.once:
                break
    except KeyboardInterrupt:
        pass

    print("\n  stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
