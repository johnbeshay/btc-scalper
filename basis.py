"""
Measure the gap between the price the model sees and the one Kalshi settles on.

    python basis.py              # fit, validate out of sample, report
    python basis.py --apply      # also write basis.json for the logger to use
    python basis.py --schema all

THE PROBLEM THIS ADDRESSES
--------------------------
The model prices a window as "will Coinbase spot finish above the strike".
Kalshi settles it as "did the 60-second average of CF Benchmarks' BRTI finish
at or above the reference" - a different index, averaged rather than sampled.

The two disagree on one window in five. Against the true settlement the
model's calibration falls apart at the extremes: strikes it calls 3-4% likely
settle YES around 17-20% of the time, and its far-strike Brier is
catastrophic. Not fat tails in Bitcoin. The model is confidently answering a
question about the wrong number.

THE MODEL
---------
Treat the settlement index as the Coinbase close times a noise term:

    settlement = close * exp(e),   e ~ N(0, beta^2)

so P(settles above strike | close) = Phi( ln(close/strike) / beta ).

`beta` is one number and it absorbs everything between the two references:
index basis, the averaging window, the timing gap between the last candle
and the actual close. It is fitted by maximum likelihood on windows where
both the Coinbase close and Kalshi's real result are known.

Then the model prices with

    sigma_eff = sqrt(sigma^2 + beta^2)

because the two sources of uncertainty are independent and variances add.
Near the money this changes little. Far out it pulls probabilities toward
0.5, which is exactly what the calibration table is asking for.

VALIDATED BEFORE IT SHIPS
-------------------------
beta is fitted on the earlier 70% of settled windows and evaluated on the
later 30% it never saw. It is only written to basis.json if skill vs market
on the held-out windows improves. A correction that only helps in-sample is
the kind of self-deception this project has already paid for once.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import score

LOG = Path(__file__).parent / "predictions.jsonl"
BASIS_FILE = Path(__file__).parent / "basis.json"


def phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def one_per_window(rows) -> list[dict]:
    """
    One observation per ticker: (ln(close/strike), hit).

    Every reading of a window shares the same close, strike and result, so
    using all three would count each settlement three times. Only rows
    graded from Kalshi's own result are used - the whole point is to measure
    the gap to the true settlement, so the Coinbase-proxy rows are excluded
    rather than allowed to contaminate the fit.
    """
    seen = {}
    for r in rows:
        if r.get("hit_source") != "kalshi":
            continue
        t = r.get("ticker")
        if not t or t in seen:
            continue
        if not r.get("close") or not r.get("strike"):
            continue
        z = math.log(r["close"] / r["strike"])
        seen[t] = {"z": z, "hit": r["hit"], "window_id": r["window_id"],
                   "ticker": t}
    return sorted(seen.values(), key=lambda o: o["window_id"])


def log_likelihood(obs, beta: float) -> float:
    ll = 0.0
    for o in obs:
        p = phi(o["z"] / beta)
        p = min(max(p, 1e-6), 1 - 1e-6)
        ll += math.log(p) if o["hit"] else math.log(1 - p)
    return ll


def fit_beta(obs, lo: float = 1e-5, hi: float = 2e-2, steps: int = 400) -> float:
    """
    Maximum-likelihood beta by grid search on a log scale, then refined.

    A grid is plenty: one parameter, a smooth likelihood, and the answer
    only needs to be right to a few percent to do its job.
    """
    if len(obs) < 20:
        raise ValueError(f"need at least 20 settled windows, have {len(obs)}")
    best_b, best_ll = None, -math.inf
    for i in range(steps + 1):
        b = lo * (hi / lo) ** (i / steps)
        ll = log_likelihood(obs, b)
        if ll > best_ll:
            best_b, best_ll = b, ll
    # refine around the best grid point
    for _ in range(3):
        span = best_b * 0.3
        for i in range(41):
            b = best_b - span + 2 * span * i / 40
            if b <= 0:
                continue
            ll = log_likelihood(obs, b)
            if ll > best_ll:
                best_b, best_ll = b, ll
    return best_b


def reprice(row: dict, beta: float) -> float:
    """The model's probability with basis noise folded into sigma."""
    sigma = row.get("sigma") or 0.0
    spot, strike = row.get("spot"), row.get("strike")
    if not spot or not strike or sigma <= 0:
        return row["p"]
    sigma_eff = math.sqrt(sigma * sigma + beta * beta)
    drifted = spot * (1 + (row.get("drift_pct") or 0.0) / 100)
    return phi(math.log(drifted / strike) / sigma_eff)


def evaluate(rows, beta: float) -> dict:
    """Skill vs market, before and after, on the given rows."""
    priced = [r for r in rows if r["mkt_p"] is not None]
    if not priced:
        return {}
    before = [dict(r) for r in priced]
    after = [dict(r, p=reprice(r, beta)) for r in priced]
    mk = score.brier(priced, "mkt_p")
    return {
        "windows": score.n_windows(priced),
        "brier_market": mk,
        "brier_before": score.brier(before),
        "brier_after": score.brier(after),
        "skill_before": score.skill_pct(score.brier(before), mk),
        "skill_after": score.skill_pct(score.brier(after), mk),
        "ci_before": score.bootstrap_skill(before),
        "ci_after": score.bootstrap_skill(after),
    }


def bootstrap_gain(rows, beta: float, iters: int = 2000, seed: int = 0):
    """
    95% interval on (skill_after - skill_before), paired by window.

    Comparing two separate point estimates on a small holdout is unreliable:
    on 90 synthetic windows it declared a correct correction harmful. The
    difference has to be resampled as a difference, window by window, so the
    noise that hits both versions the same way cancels instead of showing up
    as a spurious gap.
    """
    import random
    from collections import defaultdict
    priced = [r for r in rows if r["mkt_p"] is not None]
    by_w = defaultdict(list)
    for r in priced:
        by_w[r["window_id"]].append(r)
    wins = list(by_w)
    if len(wins) < 20:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(iters):
        sample = []
        for _ in wins:
            sample.extend(by_w[rng.choice(wins)])
        mk = score.brier(sample, "mkt_p")
        if not mk:
            continue
        before = score.skill_pct(score.brier(sample), mk)
        after = score.skill_pct(
            score.brier([dict(r, p=reprice(r, beta)) for r in sample]), mk)
        draws.append(after - before)
    if not draws:
        return None
    draws.sort()
    return (draws[int(0.025 * len(draws))],
            draws[int(0.975 * len(draws)) - 1])


def fmt_ci(ci) -> str:
    if not ci:
        return "(too few windows)"
    _, lo, hi = ci
    return f"[{lo:+.1f}% to {hi:+.1f}%]"


def main() -> int:
    p = argparse.ArgumentParser(description="Fit the settlement basis")
    p.add_argument("--log", default=str(LOG))
    p.add_argument("--schema", default="all",
                   help="which records to fit on (default all - basis is a "
                        "property of the exchange, not of the spot feed)")
    p.add_argument("--holdout", type=float, default=0.3)
    p.add_argument("--apply", action="store_true",
                   help="write basis.json if the holdout validates")
    args = p.parse_args()

    rows, _, _ = score.load(Path(args.log))
    if args.schema.lower() != "all":
        rows = [r for r in rows if r["schema"] == int(args.schema)]
    rows = [r for r in rows if not r["suppressed"]]

    obs = one_per_window(rows)
    print()
    print("=" * 64)
    print(f"  {len(obs):,} settled windows with a Kalshi result")
    print("=" * 64)
    if len(obs) < 40:
        print("  Not enough to fit and validate. Run backfill.py, or wait.")
        print()
        return 0

    # ---- chronological split ---------------------------------------------
    cut = int(len(obs) * (1 - args.holdout))
    train_obs, test_obs = obs[:cut], obs[cut:]
    train_ids = {o["ticker"] for o in train_obs}
    test_ids = {o["ticker"] for o in test_obs}
    train_rows = [r for r in rows if r.get("ticker") in train_ids]
    test_rows = [r for r in rows if r.get("ticker") in test_ids]

    beta_train = fit_beta(train_obs)
    beta_full = fit_beta(obs)

    typical_sigma = sorted(r["sigma"] for r in rows if r.get("sigma"))
    typical_sigma = typical_sigma[len(typical_sigma) // 2] if typical_sigma else 0

    print()
    print(f"  beta (fit on early {len(train_obs)})   {beta_train:.6f}  "
          f"= {beta_train * 100:.3f}% of price")
    print(f"  beta (fit on all {len(obs)})           {beta_full:.6f}  "
          f"= {beta_full * 100:.3f}% of price")
    if typical_sigma:
        print(f"  typical model sigma            {typical_sigma:.6f}  "
              f"= {typical_sigma * 100:.3f}%")
        ratio = beta_full / typical_sigma
        print(f"  basis / sigma                  {ratio:.2f}")
        print()
        if ratio > 0.7:
            print("  The settlement basis is comparable to the model's own")
            print("  forecast uncertainty. A strike the model calls 'certain'")
            print("  is not certain at all once you ask which index settles it.")
        elif ratio > 0.3:
            print("  A meaningful fraction of the model's uncertainty budget is")
            print("  basis, not forecast. Far strikes are where it bites.")
        else:
            print("  Basis is small relative to sigma. Whatever is wrong at the")
            print("  extremes, this is not most of it.")

    # ---- does it help out of sample? -------------------------------------
    print()
    print(f"  Held-out evaluation: beta fitted on the first {1 - args.holdout:.0%},")
    print(f"  scored on the last {args.holdout:.0%} it never saw")
    print("  " + "-" * 62)
    ev = evaluate(test_rows, beta_train)
    if not ev:
        print("  no priced rows in the holdout")
        print()
        return 0

    print(f"  {'':>22} {'brier':>8} {'vs market':>10}   95% CI")
    print(f"  {'market':>22} {ev['brier_market']:>8.4f}")
    print(f"  {'model, as logged':>22} {ev['brier_before']:>8.4f} "
          f"{ev['skill_before']:>+9.1f}%   {fmt_ci(ev['ci_before'])}")
    print(f"  {'model + basis':>22} {ev['brier_after']:>8.4f} "
          f"{ev['skill_after']:>+9.1f}%   {fmt_ci(ev['ci_after'])}")
    print("  " + "-" * 62)

    gain = ev["skill_after"] - ev["skill_before"]
    gci = bootstrap_gain(test_rows, beta_train)
    if gci:
        glo, ghi = gci
        print(f"  change on held-out windows: {gain:+.1f} points "
              f"[95% CI {glo:+.1f} to {ghi:+.1f}]  ({ev['windows']} windows)")
    else:
        glo = ghi = None
        print(f"  change on held-out windows: {gain:+.1f} points "
              f"({ev['windows']} windows, too few for an interval)")

    if gci and glo > 0:
        validated = True
        print("  The whole interval is above zero: helps out of sample.")
    elif gci and ghi < 0:
        validated = False
        print("  The whole interval is below zero: HURTS out of sample.")
        print("  Do not apply.")
    else:
        validated = False
        print("  The interval spans zero. Cannot tell yet whether this helps.")
        print("  Not shipping on a coin flip; revisit with more windows.")

    # ---- in-sample too, for the record -----------------------------------
    ev_all = evaluate(rows, beta_full)
    if ev_all:
        print()
        print(f"  (in-sample, all {ev_all['windows']} windows: "
              f"{ev_all['skill_before']:+.1f}% -> {ev_all['skill_after']:+.1f}%. "
              "Not evidence; shown so the two can be compared.)")

    # ---- write ----------------------------------------------------------
    print()
    if args.apply:
        if not validated:
            print("  --apply refused: the holdout did not validate.")
            print()
            return 1
        BASIS_FILE.write_text(json.dumps({
            "beta": round(beta_full, 8),
            "beta_pct": round(beta_full * 100, 4),
            "fitted_on_windows": len(obs),
            "holdout_gain_points": round(gain, 2),
            "holdout_windows": ev["windows"],
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "note": "sigma_eff = sqrt(sigma^2 + beta^2); see basis.py",
        }, indent=2))
        print(f"  wrote {BASIS_FILE.name}  (beta {beta_full:.6f})")
        print("  Restart the logger to price with it. Records will carry")
        print("  schema 4 so the eras stay separable.")
    else:
        print("  Nothing written. Re-run with --apply to write basis.json.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
