"""
The Agent contract and the Signal every agent emits.

Adding a new agent means subclassing Agent and implementing analyze().
The orchestrator picks it up with no other changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum

from .feed import Candle
from .indicators import FeeModel


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def weight(self) -> float:
        return {"high": 1.0, "medium": 0.6, "low": 0.3}[self.value]


@dataclass
class Signal:
    """One agent's read on the current market."""

    agent: str
    direction: Direction
    confidence: Confidence
    headline: str
    detail: str
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Expected move in percent, if the agent can estimate one. This is what
    # gets checked against the fee model.
    expected_move_pct: float | None = None
    tradeable: bool | None = None

    # Freeform numbers for the dashboard to render.
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["direction"] = self.direction.value
        payload["confidence"] = self.confidence.value
        payload["ts"] = self.ts.isoformat()
        return payload


class Agent:
    """Base class. Subclasses set `name` and implement analyze()."""

    name: str = "unnamed"

    #: Minimum candles needed before this agent can say anything useful.
    warmup: int = 30

    def __init__(self, fees: FeeModel | None = None) -> None:
        self.fees = fees or FeeModel()

    def analyze(self, candles: list[Candle]) -> Signal:
        raise NotImplementedError

    # -- helpers available to every subclass -------------------------------

    def _idle(self, reason: str = "not enough data yet") -> Signal:
        return Signal(
            agent=self.name,
            direction=Direction.NEUTRAL,
            confidence=Confidence.LOW,
            headline="Standing by",
            detail=reason,
        )

    def _finish(self, signal: Signal) -> Signal:
        """
        Stamp the fee verdict onto a signal before it leaves the agent.

        Only directional signals get gated. A neutral signal is context, not
        a trade proposal, so "below cost" would be meaningless on it.
        """
        if signal.direction is not Direction.NEUTRAL and (
            signal.expected_move_pct is not None
        ):
            signal.tradeable = self.fees.clears_costs(signal.expected_move_pct)
        if signal.expected_move_pct is not None:
            signal.metrics["round_trip_cost_pct"] = round(
                self.fees.round_trip_pct, 3
            )
        return signal

    def run(self, candles: list[Candle]) -> Signal:
        if len(candles) < self.warmup:
            return self._idle(f"needs {self.warmup} candles, have {len(candles)}")
        return self._finish(self.analyze(candles))
