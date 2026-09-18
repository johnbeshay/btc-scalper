# BTC SCALPER — HANDOFF (Sept 21, 2026)

## WHERE WE ARE

**Phase A (replay simulator): done, and it finally works correctly.**
**Phase B (demo executor): built and verified end to end on demo money.**
**Phase C (live): still gated. Not close, but closer than it was.**

The headline changed this session. On taker fills the model loses; on maker
fills it does not. That is the whole live question now.

---

## THE NUMBERS THAT MATTER

All on **schema 3** (299 windows), graded against Kalshi's own settlement.

| measure | result |
|---|---|
| Skill vs market | **+0.2%**  [95% CI −2.4% to +2.9%] |
| Near the money (0–0.5 sd) | +1.0% |
| Replay, `--fill ask` (taker) | **−$7.07**, 10 trades |
| Replay, `--fill bid` (maker) | **+$13.06**, 26 trades, CI −$28 to +$56 |
| Threshold agreement, ask | −0.19 (fitting noise) |
| Threshold agreement, bid | +0.39 (weak but positive) |

Skill vs market by sample size: +1.4% (149 win) → +0.8% (225) → +0.2% (299).
It is converging on zero. The model matches the book; it does not beat it.

**But gross P&L was positive all along.** The taker fee, ~1.75¢ per contract
at 50¢, is several times larger than the edge. That is why the sign flips
between fill modes.

---

## THE FEE FINDING (most important thing this session)

Kalshi's published fee schedule charges trading fees **only on orders that
match immediately**. Resting orders pay nothing unless the series is in the
"Maker Fees" list — GDP, payrolls, CPI, NBA, NHL, golf, tennis, NASCAR.

**KXBTC15M is not on that list. Maker fills are free.**

Schedule is dated July 1, 2025. **Verify against a live order ticket before
relying on it.**

---

## WHAT WORKED

- **Grading against Kalshi settlement instead of a Coinbase close.** The
  proxy was wrong on 20.6% of windows. This was the single biggest fix.
- **Live spot from `/ticker`.** The candle feed runs ~5 min behind and was
  $88 off against a ~$60 sigma. Fixing it moved skill vs market from −15.5%
  to roughly zero and fixed the far-strike collapse.
- **Schema versioning.** Every tool now refuses to pool eras silently.
- **Refusing to guess.** The settlement parser reporting UNPARSEABLE is what
  surfaced `yes_total_cost_dollars`. Code that fails loudly beat code that
  returned a plausible zero, repeatedly.
- **Synthetic-data testing of the tools themselves.** Planting a known edge
  and checking whether the tool finds it caught two bugs in the replay
  verdict logic that would otherwise have shipped.

## WHAT DIDN'T

- **The five adjusters.** On schema 3 they do essentially nothing:
  `momentum` and `vol_uncertainty` never fire, `round_numbers` flips,
  `time_of_day` −0.1%, `jump_detector` −0.2%. The two "confirmed HURTS" are
  real but worth ~0.3% Brier combined. Disabling them is free and correct
  and will not create an edge.
- **Isotonic calibration.** Rejected every time; the model is already well
  calibrated (most buckets within 2 points on schema 3).
- **Drift.** Confirmed dead: −20.1% vs −20.2% with it zeroed.
- **The settlement basis.** Real and measured (0.054% of price, ~half of
  sigma) but folding it into sigma helped only +1.1 points on held-out data,
  CI −0.7 to +2.7. `basis.py --apply` correctly refused.
- **The threshold sweep.** Selecting a threshold on one period and reporting
  on another produced contradictory answers for weeks. Tested against
  synthetic data with a known edge, the sweep detected it 16% of the time; a
  fixed pre-committed threshold detected it 68%. **The sweep costs ~4x the
  statistical power.** Not yet changed — see open items.
- **More data as a fix.** It did not rescue the number; it revealed the
  number. That is the tool working.

---

## BUGS FOUND AND FIXED THIS SESSION

1. `volume` read as an integer field that Kalshi now sends as `volume_fp`
   string → reported 0 forever. "Volume 0 every time checked" was false;
   these markets trade over a million contracts.
2. Spot taken from a cached candle feed ~5 min stale.
3. Outcomes graded from a Coinbase close instead of Kalshi's BRTI
   settlement. Wrong 20.6% of the time, concentrated near the money.
4. `replay.py` had no schema filter — pooled 185 stale-spot windows into a
   schema 3 run and reported a profit for a model that never existed.
5. Replay selected its threshold on total net, which favours "trade
   everything" and loses to fees. Now selects on return on stake.
6. The period-agreement check compared the *slope* of each column; both
   columns normally peak in the middle, so two identical shapes read as
   disagreement. Now correlates the columns against each other.
7. `learn.py --agents` had no schema filter. `vol_uncertainty` showed
   "confirmed HURTS −0.4%" purely from records logged before `WIDEN = 0.0`.
8. Ablation baseline was misaligned (`rows[:len(adjusted_p)]`), inflating
   agent credit ~9x. The old "+2.98% Brier" figure came from this.

---

## TOOLS

| file | what it does |
|---|---|
| `logger.py` | records predictions + book at T-12/8/4, then Kalshi settlement. Schema 3. |
| `score.py` | Brier, skill vs 50% and vs market, calibration, distance bands. `--schema` defaults to latest. Bootstrap CI clustered on windows. |
| `replay.py` | P&L in dollars. `--fill ask|mid|bid`. Threshold sweep, period-agreement check, bootstrap CI, 30-trade minimum before any verdict. |
| `learn.py` | isotonic calibration + `--agents` ablation, both with chronological holdout and `--schema`. |
| `backfill.py` | fetches Kalshi settlement for windows already logged. Append-only, idempotent. |
| `basis.py` | fits the Coinbase↔BRTI basis, validates out of sample, refuses to apply unless it helps. |
| `diagnose.py` | two sanity checks: does disagreement track the missed move; does Coinbase agree with settlement. |
| `disagree.py` | anatomy of high-disagreement windows — what differs about them. |
| `executor.py` | manual demo orders. Six rails, kill switch, `sync` derives P&L from settlements. |
| `runner.py` | automated demo loop. Real book in, demo orders out, own log. |
| `shardcheck.py` | per-shard balance and a market's `exchange_index`. |

~309 tests, stdlib only except `cryptography` (isolated to `core/kalshi_auth.py`).

---

## ENVIRONMENT GOTCHAS

- **PowerShell** (`PS C:\...>`) uses `$env:USERPROFILE`. Command Prompt uses
  `%USERPROFILE%`. Mixing them silently copies nothing.
- Kalshi tickers are **ET**; `window_id` is **UTC**. Four hours apart.
- Crypto markets live on **exchange shard 2**. Collateral is per-shard and
  does not follow the order; transfer with `executor.py transfer --to 2`.
- Demo prices do **not** track production. Measured on the same contract at
  the same instant: demo 6.5–9.5¢ vs production 1.7¢, spread 3 points vs
  0.1, volume 939 vs 1,238,784. Demo tests plumbing, never strategy.
- Order API is **v2**: `/portfolio/events/orders`, side is `bid`/`ask` on
  the YES leg, count and price are fixed-point **strings**.
- Settlement index differs per asset: BTC = BRTI, ETH = ETHUSD_RTI.
- GitHub repo-root fetches were stuck on a stale cache all session. Paste
  blob links or upload files directly.

---

## OPEN ITEMS, IN ORDER

1. **Test whether resting orders actually fill.** This is now the central
   question, not the model. Point `runner.py` at the bid instead of the ask
   on demo and measure: what fraction fill, how long they wait, and at what
   prices. `--fill bid` in replay is a *ceiling* — it assumes every resting
   order fills at the quoted bid, and ignores adverse selection (a resting
   buy fills when someone sells into it, and they sell into it when price is
   about to drop). Real maker P&L will be lower. Possibly by all of it.

2. **Keep logging.** The maker result rests on 26 held-out trades, below the
   30-trade minimum. Needs more windows before it means anything.

3. **Switch replay to a fixed pre-committed threshold.** Measured 4x more
   power than the sweep. Keep the sweep visible as a diagnostic; stop letting
   it choose. Pick the threshold once, write it down, do not change it after
   seeing results.

4. **Disable `time_of_day` and `jump_detector`.** Both confirmed HURTS on
   schema 3. Worth ~0.3% Brier. Free, correct, and will not create an edge.

5. **Verify the maker fee on a live order ticket** before building anything
   on it. The schedule is from July 2025.

6. **Consider more markets** — `KX{ASSET}15M` exists for BTC, ETH, SOL, XRP,
   DOGE, BNB, HYPE. 3x the windows per hour. Simulated: on markets with zero
   edge, held-out P&L comes back positive 40% of the time, so testing 5
   markets gives a false winner 92% of the time. Current guards caught 0/60
   false positives, so it is safe — but only while the guards stay strict.
   Do this **after** the maker question is settled, not alongside it, or you
   will not know which change caused what.

---

## THE GATE

Unchanged: Phase A positive out of sample **and** Phase B running clean for
days. Phase C is a one-line edit to `BASE` in `core/kalshi_exec.py` plus a
credential swap — deliberately requiring a source edit, not a flag.

What would make it reasonable to proceed: `--fill bid` positive with the
whole interval above zero, on 30+ held-out trades, **and** demonstrated fill
rates on resting orders. Not before.

What would make it reasonable to stop: resting orders do not fill, or fill
only when adversely selected. That would mean the edge exists on paper and
cannot be captured.

---

## A NOTE FOR NEXT TIME

Three days of work moved the number from "−20%, clearly losing" to "roughly
zero on taker, plausibly positive on maker." Almost none of that came from
improving the model. All of it came from fixing measurement bugs.

The model was never the problem. The instruments were.
