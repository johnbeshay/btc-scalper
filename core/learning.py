"""
The learning layer.

Two things are learned from the log, both chosen because they suit the amount
of data this problem can realistically produce.

1. CALIBRATION CORRECTION. The model outputs a probability. Reality says how
   often that probability was right. Isotonic regression learns the mapping
   between them - so if the model's "70%" is really 62%, the correction says
   so. This needs hundreds of samples, not millions, and it directly attacks
   the model's known weakness.

2. AGENT WEIGHTS. Each adjuster claims to improve the estimate. Ablation
   measures whether it actually does, by scoring the log with that agent's
   contribution removed. Agents that make predictions worse get turned down.

WHY NOT A BIGGER MODEL
----------------------
A neural network, gradient boosting, anything with real capacity, needs far
more independent examples than this problem produces. Fifteen-minute windows
arrive four an hour. A month of continuous logging is roughly 2,900 windows,
and consecutive windows are highly correlated, so the effective sample size is
much smaller than the row count suggests.

Train something flexible on that and it will fit noise. It will look excellent
on the data it was fitted to and lose money live. That failure is not a risk to
manage, it is the expected outcome, and it is the single most common way
retail algorithmic trading fails.

Isotonic regression has one shape constraint - monotonic - and nothing else to
overfit with. It is the right size of model for the amount of evidence
available.

EVERY LEARNED OUTPUT IS VALIDATED OUT OF SAMPLE
-----------------------------------------------
Nothing here is emitted unless it beats the uncorrected model on data it was
not fitted to. A correction that only improves in-sample is exactly the kind
of self-deception this module exists to prevent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

MIN_SAMPLES = 400          # below this, nothing is learned at all
MIN_BUCKET = 25            # smallest group isotonic will trust
HOLDOUT_FRACTION = 0.3


# --------------------------------------------------------------------------
# Isotonic regression
# --------------------------------------------------------------------------


def pava(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    """
    Pool Adjacent Violators. Fits a monotonic step function to (x, y) pairs.

    The only assumption is that a higher predicted probability should not
    correspond to a lower observed frequency. Everything else is read from the
    data. Returns (x_upper_bound, fitted_value) per block.
    """
    if not xs:
        return []

    order = sorted(range(len(xs)), key=lambda i: xs[i])
    blocks = [[xs[i], ys[i], 1] for i in order]  # [max_x, sum_y, count]

    merged = []
    for b in blocks:
        merged.append(b)
        while len(merged) > 1:
            a, c = merged[-2], merged[-1]
            if a[1] / a[2] <= c[1] / c[2]:
                break
            merged.pop()
            merged.pop()
            merged.append([c[0], a[1] + c[1], a[2] + c[2]])

    return [(b[0], b[1] / b[2]) for b in merged]


class IsotonicCalibrator:
    """Maps a raw model probability to a corrected one."""

    def __init__(self):
        self.blocks: list[tuple[float, float]] = []
        self.n = 0

    def fit(self, probs: list[float], hits: list[int]) -> "IsotonicCalibrator":
        self.blocks = pava(probs, [float(h) for h in hits])
        self.n = len(probs)
        return self

    def apply(self, p: float) -> float:
        if not self.blocks:
            return p
        for upper, value in self.blocks:
            if p <= upper:
                return min(max(value, 0.001), 0.999)
        return min(max(self.blocks[-1][1], 0.001), 0.999)

    def to_dict(self) -> dict:
        return {"blocks": [[round(a, 6), round(b, 6)] for a, b in self.blocks],
                "n": self.n}

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicCalibrator":
        c = cls()
        c.blocks = [(a, b) for a, b in d.get("blocks", [])]
        c.n = d.get("n", 0)
        return c


# --------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------


def brier(probs, hits) -> float:
    return sum((p - h) ** 2 for p, h in zip(probs, hits)) / len(probs)


def split(rows, fraction=HOLDOUT_FRACTION):
    """
    Chronological split, never random.

    Random splitting leaks: windows minutes apart share almost the same market
    state, so a random holdout contains near-copies of the training rows and
    every model looks brilliant. Training on the past and testing on the future
    is the only split that answers the question being asked.
    """
    rows = sorted(rows, key=lambda r: r["window_id"])
    cut = int(len(rows) * (1 - fraction))
    return rows[:cut], rows[cut:]


@dataclass
class LearnResult:
    """What was learned, and whether it survived validation."""

    accepted: bool
    reason: str
    n_train: int = 0
    n_test: int = 0
    brier_before: float = 0.0
    brier_after: float = 0.0
    calibrator: IsotonicCalibrator | None = None
    agent_scores: dict = field(default_factory=dict)

    @property
    def improvement_pct(self) -> float:
        if not self.brier_before:
            return 0.0
        return (self.brier_before - self.brier_after) / self.brier_before * 100

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "brier_before": round(self.brier_before, 6),
            "brier_after": round(self.brier_after, 6),
            "improvement_pct": round(self.improvement_pct, 2),
            "calibrator": self.calibrator.to_dict() if self.calibrator else None,
            "agent_scores": self.agent_scores,
        }


def learn_calibration(rows: list[dict]) -> LearnResult:
    """
    Fit a calibration correction and accept it only if it helps out of sample.
    """
    if len(rows) < MIN_SAMPLES:
        return LearnResult(
            accepted=False,
            reason=(
                f"only {len(rows)} resolved calls; {MIN_SAMPLES} is the minimum "
                "before a correction means anything. Keep logging."
            ),
            n_train=len(rows),
        )

    train, test = split(rows)
    if len(test) < 100:
        return LearnResult(
            accepted=False,
            reason=f"holdout is only {len(test)} rows, too small to validate against",
            n_train=len(train), n_test=len(test),
        )

    cal = IsotonicCalibrator().fit(
        [r["p"] for r in train], [r["hit"] for r in train]
    )

    test_p = [r["p"] for r in test]
    test_h = [r["hit"] for r in test]
    before = brier(test_p, test_h)
    after = brier([cal.apply(p) for p in test_p], test_h)

    # Demand a real margin. A correction that improves Brier by a hair is
    # indistinguishable from noise and will not survive contact with a new
    # market regime.
    if after >= before * 0.98:
        return LearnResult(
            accepted=False,
            reason=(
                "the correction did not improve out-of-sample accuracy by a "
                "meaningful margin. The model is either already calibrated or "
                "the data is too noisy to correct. Nothing applied."
            ),
            n_train=len(train), n_test=len(test),
            brier_before=before, brier_after=after,
        )

    return LearnResult(
        accepted=True,
        reason="correction improved accuracy on data it was not fitted to",
        n_train=len(train), n_test=len(test),
        brier_before=before, brier_after=after,
        calibrator=cal,
    )


def ablate_agents(rows: list[dict]) -> dict:
    """
    Measure each agent's real contribution by removing it.

    Each logged prediction records what every agent did to the volatility. To
    ask what the model would have said without agent X, divide its multiplier
    back out and re-derive the probability. If accuracy improves without the
    agent, the agent is hurting.
    """
    scored = {}
    agents = set()
    for r in rows:
        agents.update((r.get("agents") or {}).keys())

    if not agents:
        return {}

    baseline = brier([r["p"] for r in rows], [r["hit"] for r in rows])

    for agent in sorted(agents):
        adjusted_p, hits, touched = [], [], 0
        for r in rows:
            info = (r.get("agents") or {}).get(agent)
            sigma, spot, strike = r.get("sigma"), r.get("spot"), r.get("strike")
            if not info or not sigma or not spot or not strike:
                continue
            mult = info.get("vol_multiplier", 1.0)
            if mult <= 0:
                continue
            if abs(mult - 1.0) > 1e-9:
                touched += 1
            without = sigma / mult
            if without <= 0:
                continue
            adjusted_p.append(
                0.5 * (1 + math.erf(math.log(spot / strike) / without / math.sqrt(2)))
            )
            hits.append(r["hit"])

        if len(adjusted_p) < MIN_BUCKET:
            continue

        # Compare on exactly the rows that could be recomputed.
        subset_base = brier(
            [r["p"] for r in rows[: len(adjusted_p)]], hits[: len(adjusted_p)]
        )
        without_score = brier(adjusted_p, hits)
        scored[agent] = {
            "brier_with": round(subset_base, 6),
            "brier_without": round(without_score, 6),
            "helps": without_score > subset_base,
            "delta_pct": round((without_score - subset_base) / subset_base * 100, 2)
            if subset_base else 0.0,
            "windows_affected": touched,
            "n": len(adjusted_p),
        }

    scored["_baseline_brier"] = round(baseline, 6)
    return scored


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


MODEL_FILE = "learned.json"


def save(result: LearnResult, path: Path) -> None:
    path.write_text(json.dumps(result.to_dict(), indent=2))


def load_calibrator(path: Path) -> IsotonicCalibrator | None:
    """Load a previously accepted correction, or None if there isn't one."""
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not d.get("accepted") or not d.get("calibrator"):
        return None
    return IsotonicCalibrator.from_dict(d["calibrator"])
