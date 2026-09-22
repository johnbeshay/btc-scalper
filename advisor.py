"""
Is this Kalshi 15-minute BTC contract priced fairly right now?

    python advisor.py            check the current window once
    python advisor.py --watch    refresh every 20 seconds (Ctrl-C to stop)

A side tool for trading by hand. It does not place orders, does not log
anything, and needs no login - it reads Kalshi's public prices and the live
BTC price.

WHAT IT CAN AND CANNOT DO
-------------------------
It cannot tell you when you will win. Measured over 257 windows, the model
behind it matches the market's accuracy but does not beat it (skill versus
market -0.6%, 95% CI -3.9% to +2.7%). Nothing here predicts BTC better than
the people already trading it.

What it can do is stop you overpaying. Every contract you buy costs the
ask price plus a fee, and that price is often a few cents more than the
contract is worth. On a 15-minute market those few cents are most of the
difference between a small loss and a large one over many trades.

So it estimates what each side is actually worth, compares that to what
you would pay after the fee, and says plainly whether the price is fair.

HOW "FAIR VALUE" IS ESTIMATED
-----------------------------
Half the model, half the market's own mid price. Over the measured windows
the two were almost exactly as accurate as each other (Brier 0.1703 vs
0.1693), so neither deserves more weight than the other. Averaging two
estimates of roughly equal quality is better than trusting either alone.

A consequence worth understanding: because fair value leans on the market,
it rarely says a side is cheap. That is the tool being honest, not broken.
Most of the time the verdict will be SKIP or FAIR.

WHAT THE VERDICTS MEAN
----------------------
    SKIP          one or both sides cost more than they are worth, or
                  something about this window makes it a bad bet
    FAIR          the price is right - but buying at the ask still costs the
                  fee and part of the spread, so on average a trade here loses
                  a cent or two. This will be the most common verdict, and it
                  is the honest one: a fairly priced market is not one you can
                  profit from by buying.
    WORTH A LOOK  one side costs noticeably less than its estimated value,
                  even after the fee. Not a guarantee - the model's past big
                  disagreements with the market were usually the market being
                  right.

Standard library only. Uses the same pricing code as logger.py, so the
number it shows is the number the evaluation measured.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone

from core.adjusters import Context, build_estimate
from core.feed import CoinbaseFeed, FeedError
from core.kalshi import estimate_vol, prob_above
from core.kalshi_api import DEFAULT_SERIES, KalshiError, KalshiMarketData
from logger import window_close

MODEL_WEIGHT = 0.5          # fair = 0.5 * model + 0.5 * market mid
TAKER_FEE = 0.07            # Kalshi: 0.07 x P x (1-P) per contract
WORTH_A_LOOK = 0.02         # must beat cost by at least 2 cents
FAIR_BAND = 0.01            # within 1 cent either way counts as fair

# Red flags. Any one of these makes the verdict SKIP regardless of price.
MAX_SPREAD = 0.05           # bid-ask wider than 5c: you lose a lot to it
MIN_MINUTES = 1.0           # under a minute left: price is nearly decided
MAX_QUOTE_AGE_S = 30        # prices older than this may have moved
EXTREME = 0.92              # buying at 92c+ risks a lot to make very little


def fee(price: float) -> float:
    """Kalshi taker fee per contract, in dollars."""
    return TAKER_FEE * price * (1 - price)


def assess(*, p_model: float, yes_bid: float | None, yes_ask: float | None,
           minutes_left: float, quote_age_s: float | None = None,
           suppressed: bool = False, model_weight: float = MODEL_WEIGHT
           ) -> dict:
    """
    The whole decision, as a pure function so it can be tested.

    Returns fair value, the all-in cost and edge for each side, any red
    flags, and a verdict with a one-line reason.
    """
    flags: list[str] = []

    if yes_bid is None or yes_ask is None:
        return {"verdict": "SKIP", "reason": "the market has no two-sided price",
                "flags": ["no bid or no ask - you could not get a fair fill"],
                "sides": {}}

    mid = (yes_bid + yes_ask) / 2
    fair_yes = model_weight * p_model + (1 - model_weight) * mid
    fair_no = 1 - fair_yes

    # What you actually pay to buy each side. A NO contract costs one minus
    # the YES bid, because buying NO is the same trade as selling YES.
    cost_yes = yes_ask
    cost_no = 1 - yes_bid

    sides = {}
    for name, fair, price in (("YES", fair_yes, cost_yes),
                              ("NO", fair_no, cost_no)):
        f = fee(price)
        sides[name] = {
            "price": price,
            "fee": f,
            "fair": fair,
            "edge": fair - price - f,        # expected profit per contract
            "win": 1 - price - f,            # if it pays out
            "loss": price + f,               # if it does not
        }

    spread = yes_ask - yes_bid
    if spread > MAX_SPREAD:
        flags.append(f"wide spread ({spread * 100:.0f}c) - you lose that much "
                     "just getting in and out")
    if minutes_left < MIN_MINUTES:
        flags.append("less than a minute left - the price is nearly settled")
    if quote_age_s is not None and quote_age_s > MAX_QUOTE_AGE_S:
        flags.append(f"prices are {quote_age_s:.0f}s old and may have moved")
    if suppressed:
        flags.append("the model flagged unusual conditions in this window")
    if minutes_left > 12:
        flags.append("more than 12 minutes left - the model was tested mostly "
                     "around 4 minutes and is less reliable this early")

    best_name = max(sides, key=lambda k: sides[k]["edge"])
    best = sides[best_name]

    if flags and not (len(flags) == 1 and "12 minutes" in flags[0]):
        verdict = "SKIP"
        reason = "conditions are poor - see the warnings"
    elif best["price"] >= EXTREME:
        verdict = "SKIP"
        reason = (f"{best_name} is the better side but costs "
                  f"{best['price'] * 100:.0f}c - you would risk "
                  f"{best['loss'] * 100:.0f}c to make {best['win'] * 100:.0f}c")
    elif best["edge"] >= WORTH_A_LOOK:
        verdict = "WORTH A LOOK"
        reason = (f"{best_name} costs about {best['edge'] * 100:.1f}c less than "
                  "its estimated value, even after the fee")
    elif best["fair"] - best["price"] >= -FAIR_BAND:
        # The price itself is right; what makes the trade lose on average is
        # the fee. That is a different message from "overpriced", and a
        # beginner needs to hear the difference: nothing is wrong with this
        # market, trading it simply costs money.
        verdict = "FAIR"
        reason = (f"priced about right, but buying still costs you about "
                  f"{best['fee'] * 100:.1f}c per contract in fees on average")
    else:
        verdict = "SKIP"
        reason = ("both sides cost more than they are worth, before the fee "
                  "is even counted")

    return {"verdict": verdict, "reason": reason, "flags": flags,
            "sides": sides, "best": best_name, "mid": mid,
            "fair_yes": fair_yes, "spread": spread}


# ---------------------------------------------------------------------------
# what BTC is doing - descriptive, not a prediction
# ---------------------------------------------------------------------------

def movement(*, spot: float, strike: float, sigma: float | None,
             recent_ref: float | None) -> dict:
    """
    Where BTC stands against the line, in plain terms.

    This describes; it does not forecast. The market price already reflects
    all of it - a contract $15 below the line with minutes to go is priced
    near 50c precisely because a $15 gap is small against a normal move.
    The point of showing it is to make the price make sense, so a beginner
    can see WHY a contract costs what it does.
    """
    gap = spot - strike
    out = {"gap": gap, "typical": None, "gap_in_moves": None,
           "recent": None, "direction": "unknown"}

    if sigma and sigma > 0:
        typical = spot * sigma             # one typical move to the close, $
        out["typical"] = typical
        out["gap_in_moves"] = abs(gap) / typical if typical else None

    if recent_ref:
        recent = spot - recent_ref
        out["recent"] = recent
        # "Flat" is anything under a tenth of a typical move - small enough
        # that calling it a direction would be reading noise.
        small = (out["typical"] or spot * 0.001) * 0.1
        out["direction"] = ("up" if recent > small
                            else "down" if recent < -small else "flat")
    return out


def describe_gap(gap_in_moves: float | None) -> str:
    if gap_in_moves is None:
        return ""
    if gap_in_moves < 0.25:
        return "basically on the line - close to a coin flip"
    if gap_in_moves < 0.75:
        return "leaning one way, but one ordinary move could flip it"
    if gap_in_moves < 1.5:
        return "a real lead, though a bigger-than-usual move could still flip it"
    return "a big lead - it would take an unusually large move to flip it"


# ---------------------------------------------------------------------------
# live data - the same pricing path as logger.py
# ---------------------------------------------------------------------------

def price_now(feed: CoinbaseFeed, market: KalshiMarketData):
    now = datetime.now(timezone.utc)
    close = window_close(now)
    minutes_left = max((close - now).total_seconds() / 60, 0.1)

    quotes = market.quotes_for_window(close)
    if not quotes:
        raise KalshiError("no contract listed for this window yet")
    q = quotes[0]

    candles = feed.candles(granularity=60, limit=120)
    try:
        spot = feed.spot_price()
    except FeedError:
        spot = candles[-1].close

    vol = estimate_vol(candles, 1.0)
    est = build_estimate(Context(spot=spot, candles=candles, vol=vol,
                                 minutes_left=minutes_left, now=now,
                                 hourly=None))
    # Drift is zeroed: measured on the log it contributed nothing.
    p_above = prob_above(spot, q.strike, est.final_sigma)
    p_yes = p_above if q.yes_direction == "above" else 1 - p_above

    age = (now - q.quoted_at).total_seconds() if q.quoted_at else None

    # For the "what BTC is doing" panel. The candle feed runs a few minutes
    # behind the live ticker, so the gap between them is roughly the move
    # over the last five minutes. `sigma` is the model's uncertainty to the
    # close, as a fraction of price, so spot * sigma is a typical move in
    # dollars over the time that is left.
    recent_ref = candles[-1].close if candles else None
    return {"close": close, "minutes_left": minutes_left, "quote": q,
            "spot": spot, "p_yes": p_yes, "suppressed": est.suppressed,
            "quote_age_s": age, "sigma": est.final_sigma,
            "recent_ref": recent_ref}


def show(live: dict) -> None:
    q = live["quote"]
    a = assess(p_model=live["p_yes"], yes_bid=q.yes_bid, yes_ask=q.yes_ask,
               minutes_left=live["minutes_left"],
               quote_age_s=live["quote_age_s"], suppressed=live["suppressed"])

    m, s = divmod(int(live["minutes_left"] * 60), 60)
    gap = live["spot"] - q.strike
    word = "above" if q.yes_direction == "above" else "below"

    print()
    print("=" * 64)
    print(f"  BTC 15-min   closes {live['close'].strftime('%H:%M')} UTC   "
          f"{m}m {s:02d}s left")
    print(f"  Question: will BTC finish {word} ${q.strike:,.2f}?")
    print(f"  BTC now:  ${live['spot']:,.2f}   "
          f"(${abs(gap):,.0f} {'above' if gap >= 0 else 'below'} the line)")
    print("=" * 64)

    mv = movement(spot=live["spot"], strike=q.strike,
                  sigma=live.get("sigma"), recent_ref=live.get("recent_ref"))
    print()
    print("  What BTC is doing")
    if mv["recent"] is not None:
        arrow = {"up": "UP", "down": "DOWN", "flat": "FLAT"}[mv["direction"]]
        print(f"    last ~5 min      {arrow:<5} ({mv['recent']:+,.0f})")
    if mv["typical"]:
        print(f"    normal move      about ${mv['typical']:,.0f} either way "
              f"before the close")
        print(f"    distance         ${abs(gap):,.0f} = "
              f"{mv['gap_in_moves']:.1f} normal moves - "
              f"{describe_gap(mv['gap_in_moves'])}")
    print("    (this explains the price; it does not predict the next move -")
    print("     recent direction has not been shown to, on the logged data)")

    if not a["sides"]:
        print(f"\n  SKIP - {a['reason']}\n")
        return

    print()
    print(f"  {'':6} {'you pay':>9} {'fee':>7} {'worth':>8} {'after fee':>11}")
    for name in ("YES", "NO"):
        sd = a["sides"][name]
        print(f"  {name:6} {sd['price'] * 100:>8.0f}c {sd['fee'] * 100:>6.1f}c "
              f"{sd['fair'] * 100:>7.0f}c {sd['edge'] * 100:>+10.1f}c")
    print("  'after fee' is the average profit per contract if the fair")
    print("  value is right. Negative means you are overpaying.")

    print()
    print(f"  VERDICT: {a['verdict']}")
    print(f"  {a['reason']}")

    if a["verdict"] == "WORTH A LOOK":
        # The one verdict that invites a trade gets the strongest evidence
        # against over-reading it. In replay, when the model claimed an edge
        # the realised edge came out far smaller - on the held-out taker run,
        # claimed +19.3 points per contract against realised -5.6.
        print("  Caution: in the logged data, the model's claimed edge in cases")
        print("  like this was overstated - on held-out windows it claimed about")
        print("  +19 points and realised about -6. The market was usually right.")

    if a["flags"]:
        print()
        for f in a["flags"]:
            print(f"  ! {f}")

    b = a["sides"][a["best"]]
    print()
    print(f"  If you bought {a['best']} (1 contract):")
    print(f"    you pay        {b['loss'] * 100:.0f}c  (including the fee)")
    print(f"    if it wins     you get $1.00  -> profit {b['win'] * 100:.0f}c")
    print(f"    if it loses    you get $0.00  -> loss   {b['loss'] * 100:.0f}c")
    print()
    print("  This model has matched the market, never beaten it. Treat this")
    print("  as a check on price, not a prediction.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Is the current window fairly priced?")
    ap.add_argument("--watch", action="store_true", help="refresh repeatedly")
    ap.add_argument("--every", type=int, default=20, help="seconds between refreshes")
    ap.add_argument("--series", default=DEFAULT_SERIES)
    args = ap.parse_args()

    feed = CoinbaseFeed()
    market = KalshiMarketData(series=args.series)

    while True:
        try:
            show(price_now(feed, market))
        except (KalshiError, FeedError) as exc:
            print(f"\n  could not get a price right now: {exc}\n", file=sys.stderr)
        if not args.watch:
            return 0
        try:
            time.sleep(args.every)
        except KeyboardInterrupt:
            print("\n  stopped\n")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
