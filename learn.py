"""
Learn a correction from the log.

  python learn.py                # fit, validate, report
  python learn.py --apply        # write learned.json if it validates
  python learn.py --agents       # which agents actually help
  python learn.py --agents --schema 2     # an older era, for comparison
  python learn.py --agents --no-holdout   # full-sample only, no split

SCHEMA, AND WHY IT MATTERS HERE MOST OF ALL
-------------------------------------------
Defaults to the newest record version in the log, because an adjuster that
was changed is a different adjuster and pooling the eras measures neither.

This bit harder here than anywhere else. Two examples from the log:

  vol_uncertainty   WIDEN was set to 0.0, so on current records its
                    multiplier is exactly 1.0 and removing it can change
                    nothing. Pooled across eras it showed "confirmed HURTS
                    -0.4%" - entirely from old records logged while widening
                    was still on.

  time_of_day       shrinkage, a tighter clamp and a deadband were added on
                    2026-09-13. Most pre-schema-3 records are the old
                    version, which was pinned at its clamp 38% of the time.
                    A pooled ablation reports on a version that no longer
                    exists.

An ablation is a claim about what the code does now. Scoring it on records
produced by different code is not a weaker version of that claim; it is a
claim about something else.

Nothing is written unless the correction beats the uncorrected model on data
it was never fitted to. If it does not, that is a result too - it means the
model is already calibrated, or the data is too noisy to correct.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import score
from core.learning import (
    HOLDOUT_FRACTION,
    MIN_SAMPLES,
    ablate_agents,
    learn_calibration,
    save,
)

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
    holdout = scores.pop("_holdout", None)

    if base:
        print(f"  baseline Brier with everything on:  {base:.5f}")
        print()

    print(f"  {'agent':>16} {'with':>9} {'without':>9} {'verdict':>22}")
    print("  " + "-" * 62)
    for name, s in sorted(scores.items(), key=lambda kv: -kv[1]["delta_pct"]):
        # An agent that never moved the multiplier cannot be ablated: taking
        # nothing away changes nothing. Reporting a rounding artefact as
        # "HURTS" for such an agent is how vol_uncertainty - which is
        # hardcoded to 1.0 - ended up flagged as harmful.
        if not s.get("windows_affected"):
            verdict = "inactive - never fired"
        elif s["helps"]:
            verdict = f"helps ({s['delta_pct']:+.1f}% worse off)"
        elif abs(s["delta_pct"]) < 0.05:
            verdict = "no measurable effect"
        else:
            verdict = f"HURTS ({s['delta_pct']:+.1f}%)"
        print(
            f"  {name:>16} {s['brier_with']:>9.5f} {s['brier_without']:>9.5f} "
            f"{verdict:>22}"
        )
    print("  " + "-" * 62)
    print("  'without' is the score with that agent's effect divided back out.")
    print("  Higher without = the agent was helping. An agent marked HURTS is")
    print("  making predictions worse and should be switched off.")
    print()

    if not holdout:
        print("  Full sample only - no chronological split was run.")
        print("  These numbers include the data the adjusters were tuned on.")
        print()
        return

    print("=" * 64)
    print("  Does it hold up out of sample?")
    print("=" * 64)
    print()
    print(f"  early {holdout['n_train']:,} rows fitted-era, "
          f"late {holdout['n_test']:,} rows held out "
          f"({holdout['fraction']:.0%})")
    print()
    print(f"  {'agent':>16} {'early':>10} {'late':>10} {'verdict':>18}")
    print("  " + "-" * 60)

    for name, s in sorted(scores.items(), key=lambda kv: -kv[1]["delta_pct"]):
        tr, te = s.get("train"), s.get("test")
        early = f"{tr['delta_pct']:+.1f}%" if tr else "  n/a"
        late = f"{te['delta_pct']:+.1f}%" if te else "  n/a"

        v = s.get("verdict", "no holdout")
        if not s.get("windows_affected"):
            label = "never fired"
        elif v == "helps":
            label = "confirmed helps"
        elif v == "hurts":
            label = "confirmed HURTS"
        elif v == "unproven":
            label = "UNPROVEN - flips"
        else:
            label = "too few rows"

        print(f"  {name:>16} {early:>10} {late:>10} {label:>18}")

    print("  " + "-" * 60)
    print("  A sign that flips between the two periods is noise, not skill.")
    print("  Only 'confirmed' rows are evidence of anything. Treat an")
    print("  UNPROVEN adjuster as untested, however good the full-sample")
    print("  number looks.")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description="Learn from the prediction log")
    p.add_argument("--log", default=str(HERE / "predictions.jsonl"))
    p.add_argument("--out", default=str(HERE / "learned.json"))
    p.add_argument("--apply", action="store_true", help="write the correction if valid")
    p.add_argument("--agents", action="store_true", help="run agent ablation")
    p.add_argument("--no-holdout", action="store_true",
                   help="agent ablation on the full sample only")
    p.add_argument("--holdout", type=float, default=HOLDOUT_FRACTION,
                   help=f"holdout fraction for ablation (default {HOLDOUT_FRACTION})")
    p.add_argument("--schema", default="latest",
                   help="record version to learn from: a number, 'all', or "
                        "'latest' (default). Adjusters change; eras are not "
                        "pooled silently.")
    args = p.parse_args()

    rows, total, unresolved = score.load(Path(args.log))
    if not rows:
        print()
        print(f"  No resolved data in {args.log}.")
        print("  Run: python logger.py")
        print()
        return 0

    present = score.schema_summary(rows)
    label = str(max(present)) if args.schema == "latest" else args.schema
    if label.lower() != "all":
        want = int(label)
        kept = [r for r in rows if r["schema"] == want]
        if not kept:
            print(f"\n  No schema {want} records. Present: "
                  + ", ".join(f"{v} ({w} windows)" for v, w in present.items())
                  + "\n")
            return 0
        if len(kept) < len(rows):
            other = {v: w for v, w in present.items() if v != want}
            print(f"\n  schema {want} only; excluded "
                  + ", ".join(f"schema {v} ({w} windows)"
                              for v, w in other.items()))
        rows = kept
    elif len(present) > 1:
        print("\n  WARNING: pooling schema versions "
              + ", ".join(str(v) for v in present) + ".")
        print("  Adjusters were changed between these eras. An ablation here")
        print("  describes a model that is no longer running.")

    print(f"\n  {len(rows):,} resolved calls available "
          f"({MIN_SAMPLES} needed to learn anything)")
    print(f"  across {score.n_windows(rows):,} independent windows")

    res = learn_calibration(rows)
    show(res)

    if args.agents:
        holdout = None if args.no_holdout else args.holdout
        scores = ablate_agents(rows, holdout=holdout)
        show_agents(scores)
        idle = [n for n, v in scores.items()
                if isinstance(v, dict) and not v.get("windows_affected")]
        if idle:
            print(f"  {', '.join(sorted(idle))} never moved the multiplier on")
            print("  these records, so there is nothing to ablate. That is not")
            print("  a finding about the agent - it is the agent being off.")
            print()

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
