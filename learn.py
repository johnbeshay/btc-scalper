"""
Learn a correction from the log.

    python learn.py                # fit, validate, report
    python learn.py --apply        # write learned.json if it validates
    python learn.py --agents       # which agents actually help

Nothing is written unless the correction beats the uncorrected model on data
it was never fitted to. If it does not, that is a result too - it means the
model is already calibrated, or the data is too noisy to correct.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import score
from core.learning import MIN_SAMPLES, ablate_agents, learn_calibration, save

HERE = Path(__file__).parent


def show(res):
    print()
    print("=" * 64)
    print("  Calibration learning")
    print("=" * 64)
    print()
    if not res.accepted:
        print(f"  Nothing applied.")
        print(f"  {res.reason}")
        if res.n_test:
            print()
            print(f"  fitted on {res.n_train:,}, tested on {res.n_test:,}")
            print(f"  Brier without correction  {res.brier_before:.5f}")
            print(f"  Brier with correction     {res.brier_after:.5f}")
        print()
        return

    print(f"  Accepted. {res.reason}")
    print()
    print(f"  fitted on {res.n_train:,} calls, tested on {res.n_test:,} it never saw")
    print(f"  Brier without correction  {res.brier_before:.5f}")
    print(f"  Brier with correction     {res.brier_after:.5f}")
    print(f"  improvement               {res.improvement_pct:+.1f}%")
    print()
    print("  What it learned")
    print("  " + "-" * 50)
    print(f"  {'model says':>14}  {'really means':>14}")
    seen = set()
    for upper, value in res.calibrator.blocks:
        key = (round(upper, 2), round(value, 2))
        if key in seen:
            continue
        seen.add(key)
        arrow = "  (no change)" if abs(upper - value) < 0.02 else ""
        print(f"  {upper * 100:>13.0f}%  {value * 100:>13.0f}%{arrow}")
    print()


def show_agents(scores):
    print()
    print("=" * 64)
    print("  Agent contribution")
    print("=" * 64)
    print()
    if not scores:
        print("  No agent data in the log yet.")
        print()
        return

    base = scores.pop("_baseline_brier", None)
    if base:
        print(f"  baseline Brier with everything on: {base:.5f}")
    print()
    print(f"  {'agent':>16}  {'with':>9}  {'without':>9}  {'verdict':>22}")
    print("  " + "-" * 62)
    for name, s in sorted(scores.items(), key=lambda kv: -kv[1]["delta_pct"]):
        if s["helps"]:
            verdict = f"helps ({s['delta_pct']:+.1f}% worse off)"
        else:
            verdict = f"HURTS ({s['delta_pct']:+.1f}%)"
        print(
            f"  {name:>16}  {s['brier_with']:>9.5f}  {s['brier_without']:>9.5f}  "
            f"{verdict:>22}"
        )
    print("  " + "-" * 62)
    print("  'without' is the score with that agent's effect divided back out.")
    print("  Higher without = the agent was helping. An agent marked HURTS is")
    print("  making predictions worse and should be switched off.")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description="Learn from the prediction log")
    p.add_argument("--log", default=str(HERE / "predictions.jsonl"))
    p.add_argument("--out", default=str(HERE / "learned.json"))
    p.add_argument("--apply", action="store_true", help="write the correction if valid")
    p.add_argument("--agents", action="store_true", help="run agent ablation")
    args = p.parse_args()

    rows, total, unresolved = score.load(Path(args.log))
    if not rows:
        print()
        print(f"  No resolved data in {args.log}.")
        print("  Run:  python logger.py")
        print()
        return 0

    print(f"\n  {len(rows):,} resolved calls available "
          f"({MIN_SAMPLES} needed to learn anything)")

    res = learn_calibration(rows)
    show(res)

    if args.agents:
        show_agents(ablate_agents(rows))

    if args.apply:
        if res.accepted:
            save(res, Path(args.out))
            print(f"  Written to {args.out}. The dashboard will pick it up.")
        else:
            print("  Not written - the correction did not validate.")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
