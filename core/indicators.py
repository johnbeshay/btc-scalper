"""
Indicator math and the fee model.

Everything here is pure Python on lists of Candle - no pandas, no numpy.
At 200 candles the cost is negligible and the dependency footprint stays at
zero, which makes the whole thing trivial to deploy.
"""

from __future__ import annotations

from statistics import mean, pstdev

from .feed import Candle


# --------------------------------------------------------------------------
# Fee model
# --------------------------------------------------------------------------


class FeeModel:
    """
    Round-trip cost of a trade, expressed in percent.

    This is the single most important number in a 15-minute strategy. Coinbase
    Advanced taker fees at low 30-day volume are around 0.6% per side; maker
    around 0.25%. A round trip therefore costs roughly 0.5% to 1.2% of
    position value before any profit exists.

    Set these to YOUR actual tier - check your fee schedule, don't guess.
    """

    def __init__(
        self,
        entry_fee_pct: float = 0.60,
        exit_fee_pct: float = 0.60,
        slippage_pct: float = 0.05,
    ) -> None:
        self.entry_fee_pct = entry_fee_pct
        self.exit_fee_pct = exit_fee_pct
        self.slippage_pct = slippage_pct

    @property
    def round_trip_pct(self) -> float:
        """Total percentage the price must move just to break even."""
        return self.entry_fee_pct + self.exit_fee_pct + (self.slippage_pct * 2)

    def breakeven_price(self, entry: float, side: str = "long") -> float:
        """
        Exit price at which the trade nets exactly zero.

        A short profits as price falls, so its breakeven is entry scaled DOWN
        by the cost. Dividing by (1 + cost) is the near-miss here - it is only
        a first-order approximation and leaves a real short fractionally in
        the red. Kept explicit so the two branches stay inverses of net_pct().
        """
        cost = self.round_trip_pct / 100
        return entry * (1 + cost) if side == "long" else entry * (1 - cost)

    def net_pct(self, entry: float, exit_: float, side: str = "long") -> float:
        """Realised percentage after fees. Negative means a loss."""
        gross = (exit_ - entry) / entry * 100
        if side == "short":
            gross = -gross
        return gross - self.round_trip_pct

    def clears_costs(self, expected_move_pct: float, margin: float = 1.5) -> bool:
        """
        True if an expected move is big enough to be worth taking.

        `margin` requires the move to beat costs by a multiple, not merely
        match them - a signal that only just breaks even is not a signal.
        """
        return abs(expected_move_pct) >= self.round_trip_pct * margin


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------


def sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return mean(values[-period:])


def sma_series(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(values)):
        window = values[max(0, i - period + 1) : i + 1]
        out.append(mean(window) if len(window) == period else None)
    return out


def true_range(current: Candle, previous: Candle | None) -> float:
    if previous is None:
        return current.range
    return max(
        current.high - current.low,
        abs(current.high - previous.close),
        abs(current.low - previous.close),
    )


def atr(candles: list[Candle], period: int = 14) -> float | None:
    """Average True Range - the standard volatility measure."""
    if len(candles) < period + 1:
        return None
    trs = [true_range(candles[i], candles[i - 1]) for i in range(1, len(candles))]
    return mean(trs[-period:])


def atr_pct(candles: list[Candle], period: int = 14) -> float | None:
    """ATR as a percentage of price - comparable across price levels."""
    value = atr(candles, period)
    if value is None or not candles:
        return None
    return value / candles[-1].close * 100


def rsi(candles: list[Candle], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    closes = [c.close for c in candles]
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))

    avg_gain = mean(gains[-period:])
    avg_loss = mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def zscore(value: float, population: list[float]) -> float | None:
    """How many standard deviations `value` sits from the population mean."""
    if len(population) < 2:
        return None
    sd = pstdev(population)
    if sd == 0:
        return None
    return (value - mean(population)) / sd


def swing_points(
    candles: list[Candle], lookback: int = 2
) -> tuple[list[float], list[float]]:
    """
    Fractal swing highs and lows.

    A swing high is a candle whose high exceeds `lookback` candles on each
    side. These are the raw material for support and resistance levels.
    """
    highs, lows = [], []
    for i in range(lookback, len(candles) - lookback):
        window = candles[i - lookback : i + lookback + 1]
        centre = candles[i]
        if centre.high == max(c.high for c in window):
            highs.append(centre.high)
        if centre.low == min(c.low for c in window):
            lows.append(centre.low)
    return highs, lows


def cluster_levels(prices: list[float], tolerance_pct: float = 0.15) -> list[dict]:
    """
    Group nearby prices into levels and count how often each was touched.

    A level touched four times matters more than one touched once, so the
    touch count is what downstream agents rank on.
    """
    if not prices:
        return []

    clusters: list[list[float]] = []
    for price in sorted(prices):
        placed = False
        for cluster in clusters:
            if abs(price - mean(cluster)) / mean(cluster) * 100 <= tolerance_pct:
                cluster.append(price)
                placed = True
                break
        if not placed:
            clusters.append([price])

    levels = [{"price": mean(c), "touches": len(c)} for c in clusters]
    levels.sort(key=lambda level: level["touches"], reverse=True)
    return levels
