# F1 Race-Winner Model — Failure Analysis

Analysis of the walk-forward backtest of the LightGBM lambdarank + grid-blend model
(`backend/ml/train.py`, BLEND_W=0.55, 30-race warm-up, 71 evaluated races, seasons 2023–2026).

**Scripts (re-runnable, from `backend/` with `./.venv/bin/python <script>`):**
- `backtest_repro.py` — reproduces the walk-forward backtest, writes `backtest_races.csv`
- `categorize.py` — joins race metadata, tags causes, writes `backtest_failures.csv`
- `blend_sweep.py` — BLEND_W sweep + override analysis (caches raw scores in `raw_scores_cache.pkl`)
- `strategy_feats_ablation.py` — adds the 4 unused strategy features and re-runs the backtest

---

## 1. Reproduction

Exact match with `backend/data/models/metrics.json`, to the last digit:

| metric | reproduced | metrics.json |
|---|---|---|
| races | 71 | 71 |
| winner_hit_rate | **0.6479** | 0.6479 |
| grid-baseline hit rate | 0.5775 | 0.5775 |
| podium_overlap_avg | 2.0563 | 2.0563 |
| position_mae | 3.2810 | 3.2810 |
| spearman_avg | 0.6774 | 0.6774 |

25 of 71 races missed the winner (base miss rate **0.352**). In 15/25 misses the actual
winner was the model's rank-2 pick; only **5 severe misses** had the winner outside the
model's top 3 (Spa '23, Spielberg '24, Silverstone '24, Monza '24, São Paulo '24 —
4 of the 5 were wet races).

## 2. Failure taxonomy (primary cause, n=25)

| primary cause | n | example races |
|---|---|---|
| Strategy variance (dry winner from grid ≥3) | 7 | Austin '23 (VER from P6), Mexico '23, Spa '24, Monza '24, Austin '24, Miami '24, Lusail '25 |
| Driver variance (predicted winner DNF / collapsed) | 6 | Melbourne '24 (VER DNF), Silverstone '24 (RUS DNF), Montréal '26 (RUS DNF), Barcelona '26 (ANT DNF), Las Vegas '25 (NOR P19), Silverstone '26 (ANT P15) |
| Weather (wet race, order shuffled) | 5 | Spa '23, Spielberg '24, São Paulo '24, Miami '25, Spa '25 |
| Feature weighting (model wrongly overrode a winning pole-sitter) | 5 | Zandvoort '24, Las Vegas '24, Suzuka '25, Montréal '25, Monza '25 |
| Safety car (9 SC laps reshuffled the lead) | 1 | Imola '25 |
| Unexplained (clean dry race, P3 starter won) | 1 | Budapest '24 (Piastri from P2 beat pred VER) |

Counting secondary causes too (a miss can have several):
track_characteristics 10, safety_car 9, driver_variance 8, weather 7,
strategy_variance 7, feature_weighting 6.

Full per-race tags in `backtest_failures.csv` (all 71 races, `winner_hit`,
`primary_cause`, `secondary_causes`, `cause_notes`, `recoverability`).

## 3. Conditional miss rates (base = 0.352)

| condition | miss rate | n | vs complement |
|---|---|---|---|
| **Actual winner started grid ≥3** | **0.846** | 13 | 0.241 (n=58) |
| Winner started P2 | 0.529 | 17 | — |
| Winner started on pole | **0.122** | 41 | 0.667 (n=30) |
| Wet race | 0.438 | 16 | 0.327 dry (n=55) |
| SC laps > median (2) | 0.265 | 34 | 0.432 (n=37) |
| SC laps ≥ 5 | 0.263 | 19 | 0.385 (n=52) |
| Low-overtake track (below-median track_overtakes_avg) | 0.286 | 35 | 0.417 (n=36) |
| High attrition (≥4 DNFs) | 0.261 | 23 | 0.396 (n=48) |

Winner-hit by winner's grid slot: P1 → 0.878 (36/41), P2 → 0.471 (8/17), P3 → 0.333 (2/6),
P4+ → **0.000** (0/7). The model has essentially no mechanism to pick a winner from
outside the front row — every P4+ winner was missed.

Two counter-intuitive results worth noting:
- **Safety cars are NOT a net cause of misses.** SC-heavy races have a *lower* miss rate
  (0.263–0.265 vs 0.385–0.432). SC appears as a secondary factor in 9 misses, but
  conditionally the model does better in SC races — SC often neutralizes gaps without
  changing the winner. The "safety car disruption" hypothesis is refuted at race level.
- **Low-overtaking tracks help, not hurt** (0.286 vs 0.417): grid dominance is exactly
  what a 0.45-grid-weight blend predicts well. Track-characteristics is a real *secondary*
  amplifier of the pole-override failures (Suzuka, Monza, Zandvoort, Montréal are all
  low-overtake tracks where the model flipped away from a winning pole-sitter), but the
  bucket is not a standalone cause.

## 4. Season / regulation clustering

| season | miss rate | n |
|---|---|---|
| 2023 | 0.214 | 14 |
| 2024 | **0.458** | 24 |
| 2025 | 0.333 | 24 |
| 2026 | 0.333 | 9 |

By round-third of season: early 0.263 (n=19), **mid 0.440 (n=25)**, late 0.333 (n=27).

2024 is the standout: 11 of 25 total misses. **5 of the 11 2024 misses had Verstappen as
the predicted winner** (rounds 3, 6, 11, 13, plus grid-following cases) while the actual
winners were SAI/NOR/RUS/HAM/PIA/LEC — the Red Bull decline / McLaren-Mercedes surge.
Season-cumulative features (`season_points`, `season_rank`, `team_season_points`) and
5-race form windows lag a mid-season competitive shift by several races. The 2026
regulation-change season itself is *not* elevated (0.333, and its 3 misses are all
predicted-winner DNFs, not form-model errors) — the problem is *mid-season upgrade
swings*, not season boundaries.

## 5. The blend: where the 0.648 comes from and where it breaks

BLEND_W sweep (raw ranker scores cached from the same walk-forward, `blend_sweep.py`):

| w | hit | podium | MAE | wet hit (n=16) | dry hit (n=55) |
|---|---|---|---|---|---|
| 0.0 (grid) | 0.577 | 2.00 | 3.34 | 0.438 | 0.618 |
| 0.50 | 0.577 | 2.04 | 3.26 | 0.500 | 0.600 |
| **0.55 (prod)** | **0.648** | 2.06 | 3.28 | 0.562 | **0.673** |
| 0.60 | 0.648 | 2.06 | 3.29 | 0.562 | 0.673 |
| 0.90 | 0.662 | 2.00 | 3.48 | 0.688 | 0.655 |
| 1.0 (model only) | 0.662 | 2.00 | 3.49 | **0.688** | 0.655 |

Findings:
- The hit rate jumps 0.577 → 0.648 exactly between w=0.50 and w=0.55 — the production
  operating point sits on a knife edge where the model rank first out-votes the grid rank
  at the top of the order. Winner-hit is flat at the grid baseline for the whole 0–0.5 range.
- **The raw model alone (w=1.0) beats the blend on winner-hit (0.662 vs 0.648)** but is
  clearly worse on MAE (3.49 vs 3.28) and podium (2.00 vs 2.06). The blend is a
  whole-order compromise that costs ~1 winner.
- **Wet races monotonically prefer the model** (0.438 grid-only → 0.688 model-only);
  dry races peak at w≈0.55–0.60. A per-condition blend (w=1.0 wet, 0.55 dry) scores
  **0.676 hit / 2.07 podium / 3.31 MAE** — +2 winners over production. (Upper bound:
  uses the realized wet flag; a real system needs a pre-race rain forecast.)

Override behavior at w=0.55: the blend picked someone other than the grid leader in
19/71 races and hit 10 of them (0.526); when it followed the grid leader it hit 36/52
(0.692). In those 19 overridden races the grid leader only won 5 — overriding is
**net +5 wins vs never overriding**. So the override mechanism works on average; the
5 feature-weighting misses are its false positives.

The 5 wrong overrides share one exact signature: pole-sitter wins, model ranks them
**3rd**, blend flips the pick to the P2 qualifier (Zandvoort '24 NOR/VER, Las Vegas '24
RUS/SAI, Suzuka '25 VER/NOR, Montréal '25 RUS/PIA, Monza '25 VER/NOR — four of five on
low-overtake tracks; four involve the model preferring the championship-form favourite
over an on-form pole-sitter).

**Negative result:** gating overrides on how deep the model demotes the pole-sitter
cannot fix this. In the 10 *correct* overrides the model ranked the beaten pole-sitter
2nd; in the 5 *wrong* ones it ranked the winning pole-sitter 3rd. Reverting overrides
whenever model-rank(pole) ≤ K gives 0.577 for every K∈{2,3,4,5} — it deletes all 10 good
overrides before it saves the 5 bad ones. Rank-space demotion depth carries no signal;
any fix must use raw-score margins or new features, not rank gates.

## 6. Recoverability (25 misses)

| class | n | composition |
|---|---|---|
| Irreducible | 12 | 4 predicted-winner DNFs, 5 wet-chaos races, 3 heavy-SC/attrition races |
| Recoverable | 11 | 5 wrong pole-overrides + 6 dry strategy/track misses where winner was model rank 2–5 |
| Borderline | 2 | Budapest '24, Las Vegas '25 |

Practical ceiling: irreducible misses cap the hit rate near (71−12)/71 ≈ **0.83**.
Realistic near-term targets backed by measured prototypes: per-condition blend +2 races
(0.676); recovering even 3 of the 5 wrong pole-overrides without losing correct
overrides would reach ~0.69.

## 7. Ranked causes by impact

1. **Front-row myopia / strategy variance** — 7 primary misses; winner from grid ≥3 is
   missed 84.6% of the time and P4+ winners 100%. Partly irreducible (comebacks are rare
   events) but the model's zero hit rate on 7 P4+ winners and 4/6 missed P3 winners shows
   it never prices in race-pace-vs-grid mismatches.
2. **Driver variance** — 6 primary misses (4 predicted-winner DNFs). Mostly irreducible
   for a pre-race ranker; `driver_dnf_rate`/`team_dnf_rate` importances are near the
   bottom (51/147) and cannot foresee specific failures.
3. **Weather** — 5 primary (7 total) misses; wet miss rate 0.438 vs 0.327 dry. But the
   *model* is not the weak link — the grid prior is: wet grid-only hit is 0.438 vs
   model-only 0.688. The blend under-trusts the model exactly where the model is best.
   Largely recoverable via conditional blending (measured +2 races).
4. **Feature weighting (wrong pole overrides)** — 5 primary misses, all one signature;
   fully recoverable in principle (grid alone got all 5 right) but rank-gating provably
   fails; needs margin-aware blending or head-to-head features.
5. **Constructor/upgrade lag** — overlaps buckets 1 and 4: 2024 miss rate 0.458, mid-season
   thirds 0.440, 5 of 11 2024 misses predicted a declining Verstappen. Season-cumulative
   features lag competitive shifts.
6. **Safety car & track characteristics** — refuted as primary causes (SC races miss
   *less*: 0.265 vs 0.432; low-overtake tracks miss less: 0.286 vs 0.417). Keep only as
   secondary context.

## 8. Experiments

**E1. Condition-dependent blend weight (weather).**
- Hypothesis: the grid prior is the weak component in wet races; w should rise toward 1.0 when rain is likely.
- Method: extend `blend_scores` to `w = 0.55 + (1-0.55)*p_wet`, with `p_wet` from a pre-race forecast proxy (start: the realized `wet_race` flag as oracle; then degrade it with 10–30% label noise to simulate forecast error). Re-run `blend_sweep.py` machinery (raw scores already cached — seconds per variant).
- Metric: winner_hit_rate overall and split wet/dry; podium, MAE as guardrails.
- Confirm: overall hit ≥0.66 with noisy flag and no dry-race regression. Refute: gains vanish below ~85% forecast accuracy. (Oracle result already measured: 0.676.)

**E2. Margin-aware override gating.**
- Hypothesis: wrong pole-overrides (5 misses) have smaller raw-score margins between the model's #1/#2 and the pole-sitter than correct overrides do; rank-space gating fails (proven) but score-space may not.
- Method: on `raw_scores_cache.pkl`, compute margin = raw(model pick) − raw(pole) normalized per race (e.g., by race std of raw scores); revert override when margin < τ; sweep τ. Zero retraining needed.
- Metric: winner_hit_rate; count of good-overrides lost vs bad-overrides saved.
- Confirm: some τ saves ≥3 of the 5 wrong overrides while losing ≤1 of the 10 correct ones (net ≥ +2, → ~0.68). Refute: margin distributions overlap like the ranks did.

**E3. Form-momentum features for upgrade swings.**
- Hypothesis: 2024's 0.458 miss rate comes from season-cumulative features lagging the mid-season Red Bull→McLaren shift.
- Method: add to `features.py`: `team_form_finish_3` (3-race window), `pace_trend` (OLS slope of last 5 `pace_delta_ms`), `points_momentum` (mean of last 3 races' points − season mean). Re-run walk-forward.
- Metric: winner_hit_rate overall and per season; specifically the 5 predicted-VER 2024 misses.
- Confirm: 2024 hit improves ≥2 races, overall ≥0.66, no 2023 regression. Refute: flat overall (form windows may already saturate the signal).

**E4. Activate the dormant strategy features — measured, neutral-to-mild-positive.**
- `pit_gain_5`, `pit_stop_delta_5`, `undercut_rate_10`, `stint1_pace_5` are computed by `features.py:driver_features` and present in dataset.parquet but **absent from FEATURE_COLS**. Ran the full walk-forward with them added (`strategy_feats_ablation.py`): hit 0.648 → 0.648 (3 races gained: Mexico '23, Zandvoort '24, Monza '25; 3 lost: Yas Island '24, R2 '25, R12 '25), podium 2.06 → **2.11**, MAE 3.28 → 3.27. Two of the three gains are exactly wrong-pole-override misses. Worth adopting for the podium gain; combine with E2/E3 to keep the flipped-back races.

**E5. DNF-conditioned evaluation (measurement, not model change).**
- Hypothesis: ~4–6 misses are pure attrition noise; the model should be tracked on a "winner-hit among non-DNF predicted winners" basis so real regressions aren't masked by variance.
- Method: add to `train.py` metrics a secondary hit rate excluding races where the predicted winner DNF'd (measured here: 46/(71−4) = 0.687), and report wet/dry splits.
- Confirm/refute: n/a — this is instrumentation, but it defines the correct baseline (~0.69, not 0.65) for judging E1–E4.

Priority order: **E2** (cheapest, targets 5 fully-recoverable misses), **E1** (prototype already shows +2 races), **E3** (targets the largest season cluster).
