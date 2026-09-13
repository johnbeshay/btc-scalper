"""
Adjusters: agents that refine the estimate instead of voting on direction.

The spot build had agents voting bullish or bearish. That framing is wrong
here. Over fifteen minutes any real drift is swamped by noise, which is why
`prob_above` omits drift entirely - and an agent that injects a directional
opinion would be smuggling back in exactly what the math excludes on purpose.

So these do something different. Each one looks at the market and returns an
adjustment to the *distribution*: widen it, narrow it, nudge it, or refuse to
trade at all. They compose, in a fixed order, into one final estimate.

Each carries an `evidence` rating, because they are not equally well founded:

    strong   - the effect is well established and measured from live data
    moderate - the effect is real but the magnitude here is estimated
    weak     - plausible, small, and the first thing to switch off
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, pstdev

from .feed import Candle
from .kalshi import VolEstimate, log_returns


@dataclass
class Context:
    """Everything an adjuster is allowed to look at."""

    spot: float
    candles: list[Candle]          # recent 1-minute bars
    vol: VolEstimate
    minutes_left: float
    now: datetime
    hourly: list[Candle] | None = None   # ~12 days of 1-hour bars, if available


@dataclass
class Adjustment:
    """
    One agent's contribution.

    vol_multiplier   scales the standard deviation (1.0 = no change)
    drift_pct        shifts the centre of the distribution, in percent
    suppress         if True, no trade should be taken at all
    """

    agent: str
    evidence: str
    vol_multiplier: float = 1.0
    drift_pct: float = 0.0
    suppress: bool = False
    headline: str = ""
    detail: str = ""
    metrics: dict = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return (
            self.suppress
            or abs(self.vol_multiplier - 1.0) > 0.02
            or abs(self.drift_pct) > 0.001
        )

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "evidence": self.evidence,
            "vol_multiplier": round(self.vol_multiplier, 4),
            "drift_pct": round(self.drift_pct, 5),
            "suppress": self.suppress,
            "headline": self.headline,
            "detail": self.detail,
            "active": self.is_active,
            "metrics": self.metrics,
        }


class Adjuster:
    name = "unnamed"
    evidence = "moderate"

    def run(self, ctx: Context) -> Adjustment:
        try:
            return self.analyze(ctx)
        except Exception as exc:  # one broken agent must not kill the page
            return Adjustment(
                self.name, self.evidence,
                headline="Unavailable",
                detail=f"could not evaluate: {exc}",
            )

    def analyze(self, ctx: Context) -> Adjustment:
        raise NotImplementedError

    def _idle(self, detail: str) -> Adjustment:
        return Adjustment(self.name, self.evidence, headline="No effect", detail=detail)


# --------------------------------------------------------------------------


class JumpDetector(Adjuster):
    """
    Catches the moment the normal-distribution assumption breaks.

    A single outsized candle does two things at once: it contaminates the
    volatility estimate (which now averages in one huge bar) and it signals
    that something happened - news, a liquidation cascade, a large order.
    Both mean the model's picture of the next fifteen minutes is unreliable.

    This is the most valuable agent here, because it prevents the model's
    worst failures rather than trying to improve its average.
    """

    name = "jump_detector"
    evidence = "strong"

    def __init__(self, threshold_sd: float = 3.5, cooloff_min: int = 5):
        self.threshold_sd = threshold_sd
        self.cooloff_min = cooloff_min

    def analyze(self, ctx: Context) -> Adjustment:
        rets = log_returns(ctx.candles)
        if len(rets) < 30:
            return self._idle("not enough history")

        # Baseline excludes the recent window so a jump cannot hide itself
        # inside its own baseline.
        baseline = rets[:-self.cooloff_min]
        recent = rets[-self.cooloff_min:]
        sd = pstdev(baseline)
        if sd == 0:
            return self._idle("no variation to measure against")

        worst, worst_age = 0.0, None
        for age, r in enumerate(reversed(recent), start=1):
            z = abs(r) / sd
            if z > worst:
                worst, worst_age = z, age

        metrics = {"max_sd": round(worst, 2), "minutes_ago": worst_age}

        if worst < self.threshold_sd:
            return Adjustment(
                self.name, self.evidence,
                headline="Calm",
                detail=f"No unusual candles. Largest recent move {worst:.1f}sd.",
                metrics=metrics,
            )

        direction = "up" if recent[-worst_age] > 0 else "down"
        return Adjustment(
            self.name, self.evidence,
            vol_multiplier=1.35,
            suppress=worst_age <= 2,
            headline="Sudden move detected",
            detail=(
                f"A {worst:.1f}sd candle {direction} hit {worst_age} minute(s) ago. "
                "Something happened. The model's assumptions do not hold right "
                "after a jump, so treat every number as unreliable until things settle."
            ),
            metrics=metrics,
        )


class TimeOfDayVol(Adjuster):
    """
    Bitcoin does not move the same amount at every hour.

    The base estimate assumes the last two hours predict the next fifteen
    minutes. That is wrong at session boundaries: two quiet hours before the
    US open systematically understate what is about to happen, and two busy
    hours at 4pm overstate the evening.

    This measures the actual hour-of-day profile from about twelve days of
    hourly candles and corrects for it. No hardcoded session times - if the
    pattern is not in the data, the multiplier stays at 1.
    """

    name = "time_of_day"
    evidence = "strong"

    SHRINK_PRIOR = 24    # samples needed before half the raw ratio is kept
    CLAMP_LO = 0.85
    CLAMP_HI = 1.20
    DEADBAND = 0.02      # below this the correction is not worth applying

    def analyze(self, ctx: Context) -> Adjustment:
        if not ctx.hourly or len(ctx.hourly) < 120:
            return self._idle("needs more history than is loaded")

        buckets: dict[int, list[float]] = {}
        prev = None
        for c in ctx.hourly:
            if prev is not None and prev.close > 0 and c.close > 0:
                r = abs(math.log(c.close / prev.close))
                buckets.setdefault(c.ts.hour, []).append(r)
            prev = c

        usable = {h: v for h, v in buckets.items() if len(v) >= 4}
        if len(usable) < 12:
            return self._idle("not enough samples per hour yet")

        overall = mean([r for v in usable.values() for r in v])
        if overall == 0:
            return self._idle("no measurable variation")

        hour = ctx.now.hour
        if hour not in usable:
            return self._idle("no samples for this hour")

        ratio = mean(usable[hour]) / overall

        # Shrink toward 1 by how little evidence the bucket holds.
        #
        # This is the fix for a real failure. With 300 hourly candles a bucket
        # holds about 12 samples, and the standard error on a mean absolute
        # return that size is roughly 30% of the mean - so a ratio can land at
        # 1.4 or 0.7 on noise alone. It did: 38% of logged windows sat exactly
        # on the old clamp, which is what an estimate pinned at its bounds by
        # noise looks like, not a seasonal pattern.
        #
        # Clamping alone cannot fix that; it only caps how wrong the number
        # gets. Shrinking scales the correction by the evidence behind it, so
        # a thin bucket moves the multiplier a little and a thick one moves it
        # more. n/(n+PRIOR) is the standard shrinkage weight: at n=12 it keeps
        # about a third of the raw signal, at n=60 about three quarters.
        n = len(usable[hour])
        weight = n / (n + self.SHRINK_PRIOR)
        shrunk = 1 + (ratio - 1) * weight

        # Tighter clamp than before. With shrinkage doing the real work the
        # clamp is a backstop against pathological input, not the main guard.
        mult = max(self.CLAMP_LO, min(self.CLAMP_HI, shrunk))

        quietest = min(usable, key=lambda h: mean(usable[h]))
        busiest = max(usable, key=lambda h: mean(usable[h]))

        metrics = {
            "hour_utc": hour,
            "raw_ratio": round(ratio, 3),
            "shrink_weight": round(weight, 3),
            "samples": n,
            "quietest_hour_utc": quietest,
            "busiest_hour_utc": busiest,
        }

        if abs(mult - 1) < self.DEADBAND:
            return Adjustment(
                self.name, self.evidence,
                headline="Typical hour",
                detail=f"{hour:02d}:00 UTC moves about as much as an average hour.",
                metrics=metrics,
            )

        if mult < 1:
            detail = (
                f"{hour:02d}:00 UTC is usually {(1 - mult) * 100:.0f}% quieter than "
                "average. The recent two hours overstate what is likely next."
            )
            headline = "Quiet hour"
        else:
            detail = (
                f"{hour:02d}:00 UTC is usually {(mult - 1) * 100:.0f}% busier than "
                "average. Expect more movement than the recent past suggests."
            )
            headline = "Busy hour"

        return Adjustment(
            self.name, self.evidence,
            vol_multiplier=mult,
            headline=headline,
            detail=detail,
            metrics=metrics,
        )


class VolUncertainty(Adjuster):
    """
    When you do not know the volatility, the distribution is wider.

    Three estimators that disagree mean the regime is unstable. Uncertainty
    about a parameter widens the predictive distribution around it - pretending
    to a precision you do not have is how a model produces confident wrong
    answers. This is the mathematically principled version of the old
    "estimators disagree" warning.
    """

    name = "vol_uncertainty"
    evidence = "strong"

    WIDEN = 0.0          # 0 disables widening; 0.5 was the old behaviour
    WIDEN_CAP = 1.5
    SUPPRESS_AT = 0.8    # unchanged - the safety flag stays on

    def analyze(self, ctx: Context) -> Adjustment:
        d = ctx.vol.disagreement
        if d is None:
            return self._idle("only one estimator available")

        metrics = {
            "disagreement_pct": round(d * 100, 1),
            "ewma": ctx.vol.ewma,
            "parkinson": ctx.vol.parkinson,
            "close_to_close": ctx.vol.close_to_close,
        }

        # Widening is off. The reasoning behind it is sound - uncertainty
        # about a parameter really does widen the distribution around it -
        # but it was the wrong correction for this model.
        #
        # Ablation on 1,278 logged calls: removing this agent's multiplier
        # improved Brier by 0.9%. The mechanism is visible in the scores.
        # Widening pushes probabilities toward 50%, and near the money this
        # model's favoured side already wins MORE often than it claims
        # (confidence bias +7.5%). It is underconfident there, so widening
        # made it worse. The fix is not to widen less but to stop widening,
        # which is exactly what the ablation measured.
        #
        # Set WIDEN above 0 to turn it back on if later data disagrees.
        mult = min(1 + d * self.WIDEN, self.WIDEN_CAP) if self.WIDEN else 1.0

        # The threshold is 0.4, not 0.25, because the three estimators measure
        # different things - Parkinson reads the bar range, close-to-close
        # reads only closes - so they disagree somewhat even in a perfectly
        # stable market. Treating that structural gap as instability would
        # widen every estimate by a tenth, permanently, for no reason.
        if d < 0.4:
            return Adjustment(
                self.name, self.evidence,
                headline="Estimators agree",
                detail=(
                    f"The three volatility measures are within {d * 100:.0f}%, "
                    "which is normal. Solid footing."
                ),
                metrics=metrics,
            )

        if d > self.SUPPRESS_AT:
            return Adjustment(
                self.name, self.evidence,
                vol_multiplier=mult, suppress=True,
                headline="Cannot pin down volatility",
                detail=(
                    f"The three measures disagree by {d * 100:.0f}%. The market is "
                    "changing character faster than it can be measured. Nothing here "
                    "is trustworthy right now."
                ),
                metrics=metrics,
            )

        return Adjustment(
            self.name, self.evidence,
            vol_multiplier=mult,
            headline="Volatility unclear",
            detail=(
                f"The measures disagree by {d * 100:.0f}%. Flagged, but the "
                "estimate is left alone - widening on this signal was measured "
                "to make predictions worse, not better."
            ),
            metrics=metrics,
        )


class MomentumSkew(Adjuster):
    """
    Measures short-horizon autocorrelation instead of assuming it.

    Very short return series sometimes show mild persistence or reversal. The
    honest approach is to compute it from the data rather than to assume
    momentum exists because it feels like it should.

    In practice this reads near zero most of the time and contributes nothing,
    which is the correct behaviour. The drift it can apply is capped at a
    fraction of one standard deviation - well below the noise floor - because
    the underlying effect is genuinely small.
    """

    name = "momentum"
    evidence = "weak"

    def analyze(self, ctx: Context) -> Adjustment:
        rets = log_returns(ctx.candles)
        if len(rets) < 40:
            return self._idle("not enough history")

        a, b = rets[:-1], rets[1:]
        ma, mb = mean(a), mean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        if den == 0:
            return self._idle("no variation to correlate")

        rho = num / den
        # Standard error of a correlation on n samples.
        se = 1 / math.sqrt(len(a))
        metrics = {"autocorrelation": round(rho, 4), "significant": abs(rho) > 2 * se}

        if abs(rho) <= 2 * se:
            return Adjustment(
                self.name, self.evidence,
                headline="No momentum",
                detail=(
                    f"Recent moves show no reliable follow-through "
                    f"(correlation {rho:+.2f}, inside the noise band). "
                    "Direction is a coin flip, as usual."
                ),
                metrics=metrics,
            )

        recent = mean(rets[-5:])
        sigma_1m = ctx.vol.blended or 0
        # Cap at a fifth of a one-minute move. Deliberately timid.
        drift = max(-0.2, min(0.2, rho)) * recent * ctx.minutes_left
        drift = max(-sigma_1m * 0.2, min(sigma_1m * 0.2, drift)) * 100

        way = "continuing" if rho > 0 else "reversing"
        return Adjustment(
            self.name, self.evidence,
            drift_pct=drift,
            headline=f"Mild {way} bias",
            detail=(
                f"Moves have been {way} slightly (correlation {rho:+.2f}). "
                "The effect is small and this is the weakest signal here - "
                "do not lean on it."
            ),
            metrics=metrics,
        )


class RoundNumberPull(Adjuster):
    """
    Price clusters near round numbers, which pins outcomes toward them.

    Order books thicken at round levels, so price tends to stall there. For a
    binary contract that means a strike sitting exactly on a round number is
    more likely to end up near it - which pushes the probability toward a coin
    flip rather than toward either side.

    This is the weakest-founded agent here. The clustering effect is well
    documented across markets; its magnitude for fifteen-minute Bitcoin is
    not. The adjustment is kept small on purpose.
    """

    name = "round_numbers"
    evidence = "weak"

    def analyze(self, ctx: Context) -> Adjustment:
        spot = ctx.spot
        sigma_pts = spot * (ctx.vol.sigma_over(ctx.minutes_left) or 0)
        if sigma_pts == 0:
            return self._idle("no volatility estimate yet")

        # Proximity must scale with volatility, not price. A fixed percentage
        # is roughly $150 at Bitcoin's level, which means something is always
        # "near" a round thousand and the signal means nothing. A level only
        # matters if price could plausibly sit on it at the close.
        reach = sigma_pts * 0.5

        # Only levels people actually watch. 250-increments were tried and
        # dropped for the same always-near reason.
        nearest, size = None, None
        for step in (10000, 5000, 1000):
            candidate = round(spot / step) * step
            if abs(spot - candidate) <= reach:
                nearest, size = candidate, step
                break

        if nearest is None:
            return self._idle("no round level within reach")

        distance = abs(spot - nearest)
        metrics = {
            "level": nearest,
            "step": size,
            "distance": round(distance, 2),
            "reach": round(reach, 2),
        }

        # Narrow slightly: price is likelier to stay near the level than the
        # unconstrained model expects.
        return Adjustment(
            self.name, self.evidence,
            vol_multiplier=0.94,
            headline=f"Near ${nearest:,.0f}",
            detail=(
                f"Bitcoin is ${distance:,.0f} from ${nearest:,.0f}. Round levels "
                "attract order flow and price tends to stall there, so outcomes "
                "cluster a little closer to the level than the plain model expects."
            ),
            metrics=metrics,
        )


ALL_ADJUSTERS = [
    JumpDetector,
    TimeOfDayVol,
    VolUncertainty,
    MomentumSkew,
    RoundNumberPull,
]


@dataclass
class Estimate:
    """The finished estimate after every adjuster has had its say."""

    base_sigma: float
    final_sigma: float
    drift_pct: float
    suppressed: bool
    adjustments: list[Adjustment]

    @property
    def total_vol_change_pct(self) -> float:
        if self.base_sigma == 0:
            return 0.0
        return (self.final_sigma / self.base_sigma - 1) * 100

    @property
    def suppressors(self) -> list[Adjustment]:
        return [a for a in self.adjustments if a.suppress]

    def to_dict(self) -> dict:
        return {
            "base_sigma": self.base_sigma,
            "final_sigma": self.final_sigma,
            "drift_pct": self.drift_pct,
            "suppressed": self.suppressed,
            "vol_change_pct": round(self.total_vol_change_pct, 1),
            "adjustments": [a.to_dict() for a in self.adjustments],
        }


def build_estimate(ctx: Context, adjusters: list[Adjuster] | None = None) -> Estimate:
    """
    Run every adjuster and compose their output into one estimate.

    Volatility multipliers compound, drifts add. Any single suppressor stops
    the whole thing - refusing to trade is the one call a single agent gets to
    make alone, because the cost of a bad trade exceeds the cost of a missed one.
    """
    adjusters = adjusters or [cls() for cls in ALL_ADJUSTERS]
    results = [a.run(ctx) for a in adjusters]

    base = ctx.vol.sigma_over(ctx.minutes_left) or 0.0
    mult = 1.0
    drift = 0.0
    for r in results:
        mult *= r.vol_multiplier
        drift += r.drift_pct

    return Estimate(
        base_sigma=base,
        final_sigma=base * mult,
        drift_pct=drift,
        suppressed=any(r.suppress for r in results),
        adjustments=results,
    )
