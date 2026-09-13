"""
Pricing for Kalshi binary BTC contracts on a 15-minute horizon.

The question every contract asks is: will BTC be above strike K when the
window closes? Fifteen minutes out, drift is negligible - at any plausible
annual drift, the expected move over 15 minutes is a rounding error next to
the noise. So the whole problem reduces to volatility versus strike distance.

Fair value = P(price ends above K).
Edge = fair value - what you pay - fees.

READ THIS BEFORE TRUSTING A NUMBER
----------------------------------
The model assumes log returns are normally distributed. They are not. Crypto
has fat tails: big moves happen far more often than a normal distribution
predicts. The practical consequence is directional and predictable:

    Near the money  -> the model is roughly right.
    Far from money  -> the model UNDERSTATES the true probability.

So a far strike the model calls "2% likely, market wants 6%, great short"
may simply be the market pricing tail risk correctly and the model missing
it. Treat far-from-money signals with suspicion. `tail_warning()` flags them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev

from .feed import Candle


# --------------------------------------------------------------------------
# Fees
# --------------------------------------------------------------------------


class KalshiFees:
    """
    Kalshi's trading fee.

        fee = roundup(multiplier * contracts * price * (1 - price))

    Two things invert normal trading intuition:

    1. The fee peaks at 50c, where price*(1-price) hits its maximum of 0.25.
       Coin flips are the most expensive contracts on the exchange. A 90c
       contract costs about a quarter of what a 50c one does.

    2. Resting limit orders pay the maker rate, roughly a quarter of taker.
       If you can wait to be filled, waiting is worth real money.

    The crypto multiplier may be higher than the 0.07 standard rate. Check
    your order ticket and set `taker_multiplier` to what it actually says.
    """

    def __init__(
        self,
        taker_multiplier: float = 0.07,
        maker_multiplier: float = 0.0175,
    ) -> None:
        self.taker_multiplier = taker_multiplier
        self.maker_multiplier = maker_multiplier

    def order_fee(
        self, price: float, contracts: int = 1, maker: bool = False
    ) -> float:
        """
        Total fee in dollars for the whole order, rounded up to the cent.

        The raw product is snapped to 9 decimals before the ceiling. Without
        it, float noise pushes an exact 14.70 to 14.700000000000001 and the
        ceiling charges an extra cent - so 0.30 and 0.70 contracts, which the
        formula says must cost the same, would not.
        """
        mult = self.maker_multiplier if maker else self.taker_multiplier
        raw = round(mult * contracts * price * (1 - price), 9)
        return math.ceil(raw * 100) / 100

    def fee_per_contract(
        self, price: float, contracts: int = 1, maker: bool = False
    ) -> float:
        return self.order_fee(price, contracts, maker) / max(contracts, 1)

    def fee_as_pct_of_stake(
        self, price: float, contracts: int = 1, maker: bool = False
    ) -> float:
        """
        Fee as a percentage of capital actually risked.

        This is the number that matters and the one the formula hides. The
        fee is computed on contract value, so a 50c contract paying a 1.75c
        fee is really costing 3.5% of your stake.
        """
        stake = price * contracts
        if stake == 0:
            return 0.0
        return self.order_fee(price, contracts, maker) / stake * 100


# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------


def log_returns(candles: list[Candle]) -> list[float]:
    out = []
    for i in range(1, len(candles)):
        prev, cur = candles[i - 1].close, candles[i].close
        if prev > 0 and cur > 0:
            out.append(math.log(cur / prev))
    return out


def close_to_close_vol(candles: list[Candle]) -> float | None:
    """Standard deviation of log returns, per candle."""
    rets = log_returns(candles)
    if len(rets) < 2:
        return None
    return pstdev(rets)


def parkinson_vol(candles: list[Candle]) -> float | None:
    """
    High-low range estimator, per candle.

    Roughly five times more efficient than close-to-close because it uses
    the whole bar instead of two points. It ignores gaps, which for
    continuously-traded BTC barely matters.
    """
    vals = []
    for c in candles:
        if c.low > 0 and c.high > 0:
            vals.append(math.log(c.high / c.low) ** 2)
    if len(vals) < 2:
        return None
    return math.sqrt(mean(vals) / (4 * math.log(2)))


def ewma_vol(candles: list[Candle], lam: float = 0.94) -> float | None:
    """
    Exponentially weighted volatility, per candle.

    Recent bars count for more. This reacts to a regime change in minutes
    where a flat average takes an hour, which on a 15-minute contract is
    the difference between pricing the market you're in and the one you
    were in.
    """
    rets = log_returns(candles)
    if len(rets) < 5:
        return None
    var = rets[0] ** 2
    for r in rets[1:]:
        var = lam * var + (1 - lam) * r * r
    return math.sqrt(var)


@dataclass
class VolEstimate:
    """Three estimators and their spread. Disagreement is information."""

    close_to_close: float | None
    parkinson: float | None
    ewma: float | None
    per_candle_minutes: float

    @property
    def blended(self) -> float | None:
        """
        Weighted blend, leaning on EWMA for responsiveness and Parkinson
        for efficiency.
        """
        parts = [
            (self.ewma, 0.5),
            (self.parkinson, 0.3),
            (self.close_to_close, 0.2),
        ]
        live = [(v, w) for v, w in parts if v is not None]
        if not live:
            return None
        total_w = sum(w for _, w in live)
        return sum(v * w for v, w in live) / total_w

    @property
    def disagreement(self) -> float | None:
        """
        Spread between the highest and lowest estimator, as a fraction of
        the blend. High disagreement means the vol regime is unstable and
        every probability downstream is shakier than it looks.
        """
        vals = [
            v
            for v in (self.close_to_close, self.parkinson, self.ewma)
            if v is not None
        ]
        blend = self.blended
        if len(vals) < 2 or not blend:
            return None
        return (max(vals) - min(vals)) / blend

    def sigma_over(self, minutes: float) -> float | None:
        """
        Scale per-candle volatility to a horizon.

        Volatility grows with the square root of time, so a 15-minute
        horizon on 1-minute candles is sigma * sqrt(15).
        """
        blend = self.blended
        if blend is None or minutes <= 0:
            return None
        return blend * math.sqrt(minutes / self.per_candle_minutes)


def estimate_vol(candles: list[Candle], candle_minutes: float = 1.0) -> VolEstimate:
    return VolEstimate(
        close_to_close=close_to_close_vol(candles),
        parkinson=parkinson_vol(candles),
        ewma=ewma_vol(candles),
        per_candle_minutes=candle_minutes,
    )


# --------------------------------------------------------------------------
# Fair value
# --------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def prob_above(spot: float, strike: float, sigma: float) -> float:
    """
    P(price > strike at expiry), given volatility over the remaining window.

    Drift is omitted deliberately. Over 15 minutes, any realistic drift is
    swamped by noise, and assuming a drift you cannot measure is how a model
    starts telling you what you want to hear.

    At expiry (sigma = 0) this collapses to a step function, which is
    correct: the outcome is already determined.
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if sigma <= 0:
        return 1.0 if spot > strike else 0.0
    return norm_cdf(math.log(spot / strike) / sigma)


def sigmas_from_money(spot: float, strike: float, sigma: float) -> float | None:
    """How many standard deviations the strike sits from spot."""
    if sigma <= 0 or spot <= 0 or strike <= 0:
        return None
    return abs(math.log(spot / strike) / sigma)


def tail_warning(sigmas: float | None) -> str | None:
    """
    Flag strikes where the normal assumption is doing the most lying.

    Beyond about two standard deviations, real crypto returns diverge sharply
    from normal and the model's probability is too low.
    """
    if sigmas is None:
        return None
    if sigmas >= 3.0:
        return "far out of money: model badly understates this, do not trust it"
    if sigmas >= 2.0:
        return "out of money: fat tails make the true probability higher than modelled"
    return None


@dataclass
class Edge:
    """A fair-value read on one contract against its market price."""

    side: str               # "yes" or "no"
    fair_prob: float        # model probability the side wins
    market_price: float     # what you pay per contract
    fee_per_contract: float
    contracts: int
    sigmas_out: float | None
    warning: str | None

    @property
    def total_cost_per_contract(self) -> float:
        return self.market_price + self.fee_per_contract

    @property
    def ev_per_contract(self) -> float:
        """Expected dollars per contract. Payoff is $1 or nothing."""
        return self.fair_prob - self.total_cost_per_contract

    @property
    def ev_pct_of_stake(self) -> float:
        stake = self.market_price
        return self.ev_per_contract / stake * 100 if stake > 0 else 0.0

    @property
    def edge_before_fees(self) -> float:
        return self.fair_prob - self.market_price

    @property
    def worth_taking(self) -> bool:
        """
        Positive EV alone is not enough.

        The model's own error bar is wider than a one-cent edge, so anything
        under two cents is noise dressed up as an opportunity. A warning
        disqualifies the contract outright.
        """
        return self.ev_per_contract >= 0.02 and self.warning is None


def evaluate(
    spot: float,
    strike: float,
    minutes_left: float,
    vol: VolEstimate,
    yes_ask: float,
    no_ask: float | None = None,
    contracts: int = 1,
    fees: KalshiFees | None = None,
    maker: bool = False,
) -> list[Edge]:
    """
    Price both sides of a contract and return whichever have positive EV.

    `yes_ask` and `no_ask` are what the order book is charging, in dollars
    (62 cents is 0.62). On Kalshi the two sides rarely sum to exactly 1 -
    the gap is the spread, and it is a real cost.
    """
    fees = fees or KalshiFees()
    sigma = vol.sigma_over(minutes_left)

    if sigma is None:
        raise ValueError("not enough data to estimate volatility")

    p_yes = prob_above(spot, strike, sigma)
    sigmas = sigmas_from_money(spot, strike, sigma)
    warning = tail_warning(sigmas)

    out = []
    for side, fair, ask in (
        ("yes", p_yes, yes_ask),
        ("no", 1 - p_yes, no_ask if no_ask is not None else 1 - yes_ask),
    ):
        if ask is None or ask <= 0 or ask >= 1:
            continue
        out.append(
            Edge(
                side=side,
                fair_prob=fair,
                market_price=ask,
                fee_per_contract=fees.fee_per_contract(ask, contracts, maker),
                contracts=contracts,
                sigmas_out=sigmas,
                warning=warning,
            )
        )

    out.sort(key=lambda e: e.ev_per_contract, reverse=True)
    return out


# --------------------------------------------------------------------------
# Position sizing in dollars
# --------------------------------------------------------------------------


@dataclass
class Position:
    """
    What a dollar amount actually buys.

    Kalshi presents trading as "how much do you want to put in", not "how many
    contracts". Underneath it is still contracts, and the gap matters: a stake
    rarely divides evenly into the contract price, and the fee is charged on
    contract value rather than on what you staked. So the amount that leaves
    your balance is not the amount you typed.
    """

    stake_requested: float
    price: float
    contracts: int
    contract_cost: float
    fee: float
    fair_prob: float

    @property
    def total_cost(self) -> float:
        """What actually leaves your account, fee included."""
        return self.contract_cost + self.fee

    @property
    def payout_if_win(self) -> float:
        """Each contract settles at $1."""
        return float(self.contracts)

    @property
    def profit_if_win(self) -> float:
        return self.payout_if_win - self.total_cost

    @property
    def loss_if_lose(self) -> float:
        """A losing contract settles at zero, so the whole cost is gone."""
        return self.total_cost

    @property
    def unspent(self) -> float:
        """Stake left over because contracts come in whole units."""
        return max(self.stake_requested - self.total_cost, 0.0)

    @property
    def expected_value(self) -> float:
        return self.fair_prob * self.payout_if_win - self.total_cost

    @property
    def return_if_win_pct(self) -> float:
        if self.total_cost == 0:
            return 0.0
        return self.profit_if_win / self.total_cost * 100

    @property
    def fee_pct_of_stake(self) -> float:
        if self.total_cost == 0:
            return 0.0
        return self.fee / self.total_cost * 100

    @property
    def breakeven_prob(self) -> float:
        """
        How often this has to win just to break even.

        This is the number worth staring at. A contract at 35c needs to win
        more than 35% of the time before fees, and rather more after them.
        """
        if self.contracts == 0:
            return 1.0
        return self.total_cost / self.payout_if_win

    def to_dict(self) -> dict:
        return {
            "contracts": self.contracts,
            "price": self.price,
            "contract_cost": round(self.contract_cost, 2),
            "fee": round(self.fee, 2),
            "total_cost": round(self.total_cost, 2),
            "payout_if_win": round(self.payout_if_win, 2),
            "profit_if_win": round(self.profit_if_win, 2),
            "loss_if_lose": round(self.loss_if_lose, 2),
            "unspent": round(self.unspent, 2),
            "expected_value": round(self.expected_value, 2),
            "return_if_win_pct": round(self.return_if_win_pct, 1),
            "fee_pct_of_stake": round(self.fee_pct_of_stake, 2),
            "breakeven_prob": round(self.breakeven_prob, 4),
            "fair_prob": round(self.fair_prob, 4),
        }


def size_position(
    stake: float,
    price: float,
    fair_prob: float,
    fees: KalshiFees | None = None,
    maker: bool = False,
) -> Position:
    """
    Work out what `stake` dollars buys at `price` per contract.

    Contract count is floored, then reduced if the fee would push the total
    past the stake. Buying more than you meant to spend is never the right
    default, so the sizing errs downward.
    """
    fees = fees or KalshiFees()

    if price <= 0 or price >= 1 or stake <= 0:
        return Position(stake, price, 0, 0.0, 0.0, fair_prob)

    contracts = int(stake // price)
    while contracts > 0:
        cost = contracts * price
        fee = fees.order_fee(price, contracts, maker)
        if cost + fee <= stake + 1e-9:
            return Position(stake, price, contracts, cost, fee, fair_prob)
        contracts -= 1

    return Position(stake, price, 0, 0.0, 0.0, fair_prob)
