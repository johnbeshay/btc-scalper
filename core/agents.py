"""
Tier 1 agents - the five that matter on a 15-minute horizon.

Each one reads the same candle list and returns exactly one Signal. They
never talk to each other; aggregation is the orchestrator's job.
"""

from __future__ import annotations

from .base import Agent, Confidence, Direction, Signal
from .feed import Candle
from .indicators import (
    atr_pct,
    cluster_levels,
    rsi,
    sma,
    swing_points,
    zscore,
)


class PriceActionAgent(Agent):
    """
    Sharp moves in the last few candles.

    On a 5-minute chart a 15-minute trade spans three candles, so that is the
    window this agent judges momentum over.
    """

    name = "price_action"
    warmup = 20

    def __init__(self, fees=None, window: int = 3, spike_z: float = 1.8):
        super().__init__(fees)
        self.window = window
        self.spike_z = spike_z

    def analyze(self, candles: list[Candle]) -> Signal:
        recent = candles[-self.window :]
        start, last = recent[0].open, recent[-1].close
        move_pct = (last - start) / start * 100

        history = [abs(c.pct_change) for c in candles[-40:-self.window]]
        z = zscore(abs(move_pct), history) or 0.0

        if abs(move_pct) < 0.05:
            return Signal(
                agent=self.name,
                direction=Direction.NEUTRAL,
                confidence=Confidence.LOW,
                headline="Flat",
                detail=f"{move_pct:+.2f}% over last {self.window} candles.",
                expected_move_pct=abs(move_pct),
                metrics={"move_pct": round(move_pct, 3)},
            )

        direction = Direction.BULLISH if move_pct > 0 else Direction.BEARISH

        if z >= self.spike_z:
            confidence = Confidence.HIGH
            headline = "Sharp move"
        elif z >= 1.0:
            confidence = Confidence.MEDIUM
            headline = "Above-average move"
        else:
            confidence = Confidence.LOW
            headline = "Ordinary drift"

        return Signal(
            agent=self.name,
            direction=direction,
            confidence=confidence,
            headline=headline,
            detail=(
                f"{move_pct:+.2f}% across {self.window} candles "
                f"({z:.1f} sd vs recent norm)."
            ),
            expected_move_pct=abs(move_pct),
            metrics={"move_pct": round(move_pct, 3), "zscore": round(z, 2)},
        )


class VolumeAgent(Agent):
    """
    Volume anomalies.

    Volume confirms price. A move on thin volume is noise; the same move on
    three times normal volume is participation.
    """

    name = "volume"
    warmup = 30

    def __init__(self, fees=None, spike_ratio: float = 1.8):
        super().__init__(fees)
        self.spike_ratio = spike_ratio

    def analyze(self, candles: list[Candle]) -> Signal:
        last = candles[-1]
        baseline = sma([c.volume for c in candles[:-1]], 20)
        if not baseline:
            return self._idle()

        ratio = last.volume / baseline if baseline else 1.0

        if ratio < self.spike_ratio:
            return Signal(
                agent=self.name,
                direction=Direction.NEUTRAL,
                confidence=Confidence.LOW,
                headline="Normal volume",
                detail=f"{ratio:.2f}x the 20-candle average.",
                metrics={"volume_ratio": round(ratio, 2)},
            )

        direction = Direction.BULLISH if last.is_green else Direction.BEARISH
        confidence = Confidence.HIGH if ratio >= 3.0 else Confidence.MEDIUM
        side = "buying" if last.is_green else "selling"

        return Signal(
            agent=self.name,
            direction=direction,
            confidence=confidence,
            headline=f"Volume spike ({ratio:.1f}x)",
            detail=f"{ratio:.1f}x average volume on a {side} candle.",
            expected_move_pct=abs(last.pct_change),
            metrics={
                "volume_ratio": round(ratio, 2),
                "candle_pct": round(last.pct_change, 3),
            },
        )


class LevelsAgent(Agent):
    """
    Support and resistance, and how close price is to them.

    Levels are built from fractal swing points clustered by proximity. The
    more times a level was touched, the more it matters.
    """

    name = "levels"
    warmup = 50

    def __init__(self, fees=None, near_pct: float = 0.30):
        super().__init__(fees)
        self.near_pct = near_pct

    def analyze(self, candles: list[Candle]) -> Signal:
        price = candles[-1].close
        highs, lows = swing_points(candles, lookback=2)

        resistances = [
            level
            for level in cluster_levels(highs)
            if level["price"] > price and level["touches"] >= 2
        ]
        supports = [
            level
            for level in cluster_levels(lows)
            if level["price"] < price and level["touches"] >= 2
        ]

        nearest_res = min(resistances, key=lambda x: x["price"], default=None)
        nearest_sup = max(supports, key=lambda x: x["price"], default=None)

        metrics = {
            "price": round(price, 2),
            "support": round(nearest_sup["price"], 2) if nearest_sup else None,
            "support_touches": nearest_sup["touches"] if nearest_sup else 0,
            "resistance": round(nearest_res["price"], 2) if nearest_res else None,
            "resistance_touches": nearest_res["touches"] if nearest_res else 0,
        }

        # Distance to each level, as a percentage.
        to_sup = (price - nearest_sup["price"]) / price * 100 if nearest_sup else None
        to_res = (nearest_res["price"] - price) / price * 100 if nearest_res else None
        metrics["pct_to_support"] = round(to_sup, 3) if to_sup is not None else None
        metrics["pct_to_resistance"] = round(to_res, 3) if to_res is not None else None

        # Risk/reward for a long taken here, stop below support, target at
        # resistance. Fees are charged against the reward, not the risk.
        if to_sup and to_res:
            net_reward = to_res - self.fees.round_trip_pct
            metrics["net_reward_pct"] = round(net_reward, 3)
            metrics["risk_reward"] = (
                round(net_reward / to_sup, 2) if to_sup > 0 else None
            )

        if to_sup is not None and to_sup <= self.near_pct:
            return Signal(
                agent=self.name,
                direction=Direction.BULLISH,
                confidence=(
                    Confidence.HIGH
                    if nearest_sup["touches"] >= 3
                    else Confidence.MEDIUM
                ),
                headline="At support",
                detail=(
                    f"Price {to_sup:.2f}% above support "
                    f"${nearest_sup['price']:,.0f} "
                    f"(touched {nearest_sup['touches']}x)."
                ),
                expected_move_pct=to_res,
                metrics=metrics,
            )

        if to_res is not None and to_res <= self.near_pct:
            return Signal(
                agent=self.name,
                direction=Direction.BEARISH,
                confidence=(
                    Confidence.HIGH
                    if nearest_res["touches"] >= 3
                    else Confidence.MEDIUM
                ),
                headline="At resistance",
                detail=(
                    f"Price {to_res:.2f}% below resistance "
                    f"${nearest_res['price']:,.0f} "
                    f"(touched {nearest_res['touches']}x)."
                ),
                expected_move_pct=to_sup,
                metrics=metrics,
            )

        if nearest_sup and nearest_res:
            detail = (
                f"Between support ${nearest_sup['price']:,.0f} "
                f"and resistance ${nearest_res['price']:,.0f}."
            )
        elif nearest_sup:
            detail = (
                f"Support ${nearest_sup['price']:,.0f} below, "
                "nothing overhead yet."
            )
        elif nearest_res:
            detail = (
                f"Resistance ${nearest_res['price']:,.0f} above, "
                "no support built below."
            )
        else:
            detail = "No levels with repeat touches in range."

        return Signal(
            agent=self.name,
            direction=Direction.NEUTRAL,
            confidence=Confidence.LOW,
            headline="Mid-range",
            detail=detail,
            expected_move_pct=to_res,
            metrics=metrics,
        )


class TrendAgent(Agent):
    """
    Moving-average structure plus RSI.

    Periods are short on purpose. A 200-period MA on 5-minute candles is
    almost 17 hours of history, which is macro context, not scalping context.
    """

    name = "trend"
    warmup = 55

    def analyze(self, candles: list[Candle]) -> Signal:
        closes = [c.close for c in candles]
        price = closes[-1]
        fast, mid, slow = sma(closes, 9), sma(closes, 21), sma(closes, 50)
        momentum = rsi(candles, 14)

        if None in (fast, mid, slow):
            return self._idle()

        metrics = {
            "sma9": round(fast, 2),
            "sma21": round(mid, 2),
            "sma50": round(slow, 2),
            "rsi": round(momentum, 1) if momentum else None,
        }

        stacked_up = price > fast > mid > slow
        stacked_down = price < fast < mid < slow
        spread_pct = abs(fast - slow) / price * 100
        metrics["ma_spread_pct"] = round(spread_pct, 3)

        if stacked_up:
            return Signal(
                agent=self.name,
                direction=Direction.BULLISH,
                confidence=(
                    Confidence.HIGH if spread_pct > 0.25 else Confidence.MEDIUM
                ),
                headline="Uptrend",
                detail=(
                    f"Price above 9/21/50 MAs, spread {spread_pct:.2f}%. "
                    f"RSI {momentum:.0f}."
                ),
                expected_move_pct=spread_pct,
                metrics=metrics,
            )

        if stacked_down:
            return Signal(
                agent=self.name,
                direction=Direction.BEARISH,
                confidence=(
                    Confidence.HIGH if spread_pct > 0.25 else Confidence.MEDIUM
                ),
                headline="Downtrend",
                detail=(
                    f"Price below 9/21/50 MAs, spread {spread_pct:.2f}%. "
                    f"RSI {momentum:.0f}."
                ),
                expected_move_pct=spread_pct,
                metrics=metrics,
            )

        return Signal(
            agent=self.name,
            direction=Direction.NEUTRAL,
            confidence=Confidence.LOW,
            headline="No clear trend",
            detail=f"MAs tangled, spread {spread_pct:.2f}%. Chop risk is high.",
            expected_move_pct=spread_pct,
            metrics=metrics,
        )


class VolatilityAgent(Agent):
    """
    ATR regime - the gatekeeper agent.

    This one rarely says "buy". Its job is to tell you whether the market is
    even moving enough to pay for a round trip. In a low-ATR regime, no other
    signal should be acted on.
    """

    name = "volatility"
    warmup = 40

    def analyze(self, candles: list[Candle]) -> Signal:
        current = atr_pct(candles, 14)
        if current is None:
            return self._idle()

        history = [
            atr_pct(candles[: i + 1], 14)
            for i in range(30, len(candles))
        ]
        history = [h for h in history if h is not None]
        percentile = (
            sum(1 for h in history if h < current) / len(history) * 100
            if history
            else 50.0
        )

        # An ATR reading is roughly the move available per candle. Three
        # candles is the 15-minute horizon.
        available = current * 3
        cost = self.fees.round_trip_pct

        metrics = {
            "atr_pct": round(current, 3),
            "atr_percentile": round(percentile, 0),
            "expected_15m_range_pct": round(available, 3),
            "round_trip_cost_pct": round(cost, 3),
            "cost_coverage": round(available / cost, 2) if cost else None,
        }

        if available < cost:
            return Signal(
                agent=self.name,
                direction=Direction.NEUTRAL,
                confidence=Confidence.HIGH,
                headline="Too quiet to trade",
                detail=(
                    f"Typical 15-min range is {available:.2f}%, "
                    f"round-trip cost is {cost:.2f}%. "
                    "The math does not work right now."
                ),
                expected_move_pct=available,
                metrics=metrics,
            )

        if percentile < 25:
            return Signal(
                agent=self.name,
                direction=Direction.NEUTRAL,
                confidence=Confidence.MEDIUM,
                headline="Compressed",
                detail=(
                    f"ATR in the {percentile:.0f}th percentile. "
                    "Quiet periods often precede expansion - wait for the break."
                ),
                expected_move_pct=available,
                metrics=metrics,
            )

        if percentile > 80:
            return Signal(
                agent=self.name,
                direction=Direction.NEUTRAL,
                confidence=Confidence.MEDIUM,
                headline="High volatility",
                detail=(
                    f"ATR in the {percentile:.0f}th percentile. "
                    "Moves are large - size down and widen stops."
                ),
                expected_move_pct=available,
                metrics=metrics,
            )

        return Signal(
            agent=self.name,
            direction=Direction.NEUTRAL,
            confidence=Confidence.LOW,
            headline="Normal volatility",
            detail=(
                f"Typical 15-min range {available:.2f}% "
                f"vs {cost:.2f}% costs ({available / cost:.1f}x coverage)."
            ),
            expected_move_pct=available,
            metrics=metrics,
        )


TIER_1 = [
    PriceActionAgent,
    VolumeAgent,
    LevelsAgent,
    TrendAgent,
    VolatilityAgent,
]
