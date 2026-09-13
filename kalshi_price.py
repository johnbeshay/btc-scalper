"""
Price a Kalshi BTC contract against live market data.

    python kalshi_price.py --strike 100200 --minutes 12 --yes-ask 38

Prices are in cents. Spot and volatility are pulled from Coinbase 1-minute
candles unless you override with --spot.

    --strike     the contract's strike price in dollars
    --minutes    minutes until the window closes
    --yes-ask    what the book wants for YES, in cents
    --no-ask     what the book wants for NO, in cents (optional)
    --contracts  order size, affects fee rounding (default 10)
    --maker      price it as a resting limit order instead of a market order
    --multiplier Kalshi fee multiplier, check your order ticket (default 0.07)
    --spot       override the live price, for testing
    --table      show fair value across a range of strikes instead
"""

from __future__ import annotations

import argparse
import sys

from core.feed import CoinbaseFeed, FeedError
from core.kalshi import (
    KalshiFees,
    estimate_vol,
    evaluate,
    prob_above,
    sigmas_from_money,
    tail_warning,
)


def get_market(spot_override: float | None):
    """Return (spot, VolEstimate) from 1-minute Coinbase candles."""
    feed = CoinbaseFeed()
    candles = feed.candles(granularity=60, limit=120)
    if len(candles) < 30:
        raise FeedError("not enough candles returned to estimate volatility")
    spot = spot_override if spot_override else candles[-1].close
    return spot, estimate_vol(candles, candle_minutes=1.0)


def print_header(spot, vol, minutes, fees):
    sigma = vol.sigma_over(minutes)
    disagree = vol.disagreement
    print()
    print(f"  BTC spot          ${spot:,.2f}")
    print(f"  Window            {minutes:g} minutes")
    print(
        f"  Volatility        {sigma * 100:.3f}% over the window "
        f"(about ${spot * sigma:,.0f})"
    )
    if disagree is not None:
        note = "  <- estimators disagree, treat numbers as soft" if disagree > 0.5 else ""
        print(f"  Estimator spread  {disagree * 100:.0f}%{note}")
    print(f"  Fee multiplier    {fees.taker_multiplier}")
    print()
    return sigma


def show_table(spot, vol, minutes, fees):
    sigma = print_header(spot, vol, minutes, fees)
    step = spot * sigma * 0.5
    print("  Fair value by strike")
    print("  " + "-" * 54)
    print(f"  {'strike':>12}  {'YES':>7}  {'NO':>7}   {'distance':>9}")
    for k in range(-4, 5):
        strike = round((spot + k * step) / 10) * 10
        p = prob_above(spot, strike, sigma)
        sig = sigmas_from_money(spot, strike, sigma)
        warn = "  fat-tail zone" if tail_warning(sig) else ""
        print(
            f"  {strike:>12,}  {p * 100:>6.1f}c  {(1 - p) * 100:>6.1f}c   "
            f"{sig:>6.1f} sd{warn}"
        )
    print()
    print("  Compare these against the book. Any strike the market prices")
    print("  more than a couple of cents away from fair value is worth a look.")
    print()


def show_contract(spot, vol, minutes, fees, args):
    sigma = print_header(spot, vol, minutes, fees)

    yes_ask = args.yes_ask / 100
    no_ask = args.no_ask / 100 if args.no_ask is not None else None

    edges = evaluate(
        spot=spot,
        strike=args.strike,
        minutes_left=minutes,
        vol=vol,
        yes_ask=yes_ask,
        no_ask=no_ask,
        contracts=args.contracts,
        fees=fees,
        maker=args.maker,
    )

    if not edges:
        print("  No valid prices to evaluate. Asks must be between 1c and 99c.")
        return

    order_type = "resting limit order" if args.maker else "market order"
    print(f"  Strike ${args.strike:,}   {args.contracts} contracts, {order_type}")
    print("  " + "-" * 60)

    for e in edges:
        verdict = "TAKE" if e.worth_taking else "skip"
        print()
        print(f"  {e.side.upper()}")
        print(f"    fair value      {e.fair_prob * 100:.1f}c")
        print(f"    book asks       {e.market_price * 100:.0f}c")
        print(f"    fee             {e.fee_per_contract * 100:.2f}c per contract")
        print(f"    edge before fee {e.edge_before_fees * 100:+.1f}c")
        print(
            f"    expected value  {e.ev_per_contract * 100:+.2f}c per contract  "
            f"({e.ev_pct_of_stake:+.1f}% of stake)"
        )
        total = e.ev_per_contract * args.contracts
        print(f"    on {args.contracts} contracts  ${total:+.2f} expected")
        print(f"    verdict         {verdict}")
        if e.warning:
            print(f"    WARNING         {e.warning}")

    best = edges[0]
    print()
    if best.worth_taking:
        print(f"  Best side is {best.side.upper()}.")
        if not args.maker:
            maker_edges = evaluate(
                spot, args.strike, minutes, vol, yes_ask, no_ask,
                args.contracts, fees, maker=True,
            )
            gain = maker_edges[0].ev_per_contract - best.ev_per_contract
            print(
                f"  A resting limit order would add {gain * 100:.2f}c per "
                f"contract if it fills."
            )
    else:
        print("  Nothing here beats the noise floor. Sitting out is a position.")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description="Price a Kalshi BTC contract")
    p.add_argument("--strike", type=float, help="strike price in dollars")
    p.add_argument("--minutes", type=float, default=15, help="minutes to close")
    p.add_argument("--yes-ask", type=float, help="YES ask in cents")
    p.add_argument("--no-ask", type=float, default=None, help="NO ask in cents")
    p.add_argument("--contracts", type=int, default=10)
    p.add_argument("--maker", action="store_true")
    p.add_argument("--multiplier", type=float, default=0.07)
    p.add_argument("--spot", type=float, default=None)
    p.add_argument("--table", action="store_true", help="fair value across strikes")
    args = p.parse_args()

    fees = KalshiFees(taker_multiplier=args.multiplier)

    try:
        spot, vol = get_market(args.spot)
    except FeedError as exc:
        print(f"Could not reach the price feed: {exc}", file=sys.stderr)
        print("Pass --spot to price without live data.", file=sys.stderr)
        return 1

    if vol.sigma_over(args.minutes) is None:
        print("Not enough data to estimate volatility.", file=sys.stderr)
        return 1

    if args.table or args.strike is None:
        show_table(spot, vol, args.minutes, fees)
        if args.strike is None and not args.table:
            print("  Add --strike and --yes-ask to price a specific contract.")
            print()
        return 0

    if args.yes_ask is None:
        print("Pass --yes-ask to price a contract, or use --table.", file=sys.stderr)
        return 1

    show_contract(spot, vol, args.minutes, fees, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
