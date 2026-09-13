"""
Runs every agent and folds their signals into one verdict.

The aggregation rule matters more than any individual agent: the volatility
agent holds a veto. If the market is not moving enough to cover a round trip,
no amount of bullish agreement makes a trade worth taking.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .agents import TIER_1
from .base import Agent, Confidence, Direction, Signal
from .feed import Candle
from .indicators import FeeModel


class Verdict:
    """The aggregate read across all agents."""

    def __init__(
        self,
        direction: Direction,
        score: float,
        headline: str,
        signals: list[Signal],
        vetoed: bool = False,
        veto_reason: str = "",
    ) -> None:
        self.direction = direction
        self.score = score
        self.headline = headline
        self.signals = signals
        self.vetoed = vetoed
        self.veto_reason = veto_reason
        self.ts = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "direction": self.direction.value,
            "score": round(self.score, 1),
            "headline": self.headline,
            "vetoed": self.vetoed,
            "veto_reason": self.veto_reason,
            "ts": self.ts.isoformat(),
            "signals": [s.to_dict() for s in self.signals],
        }


class Orchestrator:
    def __init__(
        self,
        agents: list[Agent] | None = None,
        fees: FeeModel | None = None,
    ) -> None:
        self.fees = fees or FeeModel()
        self.agents = agents or [cls(fees=self.fees) for cls in TIER_1]

    def run(self, candles: list[Candle]) -> Verdict:
        signals = [agent.run(candles) for agent in self.agents]
        return self._aggregate(signals)

    def _aggregate(self, signals: list[Signal]) -> Verdict:
        bull = sum(
            s.confidence.weight for s in signals if s.direction is Direction.BULLISH
        )
        bear = sum(
            s.confidence.weight for s in signals if s.direction is Direction.BEARISH
        )

        total = bull + bear
        if total == 0:
            score = 0.0
            direction = Direction.NEUTRAL
        else:
            net = (bull - bear) / total
            score = abs(net) * 100
            if net > 0.2:
                direction = Direction.BULLISH
            elif net < -0.2:
                direction = Direction.BEARISH
            else:
                direction = Direction.NEUTRAL

        # Volatility veto.
        vol = next((s for s in signals if s.agent == "volatility"), None)
        if vol and vol.headline == "Too quiet to trade":
            return Verdict(
                direction=Direction.NEUTRAL,
                score=score,
                headline="Stand down",
                signals=signals,
                vetoed=True,
                veto_reason=vol.detail,
            )

        # Nothing agrees strongly enough.
        if direction is Direction.NEUTRAL:
            headline = "No edge right now"
        elif score >= 70:
            headline = f"Strong {direction.value} alignment"
        elif score >= 40:
            headline = f"Leaning {direction.value}"
        else:
            headline = f"Weak {direction.value} tilt"

        return Verdict(direction, score, headline, signals)

    # -- convenience -------------------------------------------------------

    def describe(self, verdict: Verdict) -> str:
        """Plain-text rendering, useful before the dashboard exists."""
        lines = [
            "=" * 62,
            f"  {verdict.headline.upper()}   (score {verdict.score:.0f}/100)",
        ]
        if verdict.vetoed:
            lines.append(f"  VETO: {verdict.veto_reason}")
        lines.append("=" * 62)

        for s in verdict.signals:
            mark = {"bullish": "+", "bearish": "-", "neutral": "."}[s.direction.value]
            flag = ""
            if s.tradeable is False:
                flag = "  [below cost]"
            lines.append(
                f"  {mark} {s.agent:<14} {s.confidence.value:<7} "
                f"{s.headline}{flag}"
            )
            lines.append(f"      {s.detail}")

        lines.append("=" * 62)
        return "\n".join(lines)
