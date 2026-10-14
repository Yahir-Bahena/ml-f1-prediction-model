# Ranking-quality research review (2026-07-19)

Senior-review of the F1 prediction system's evaluation strategy. No implementation;
this document is the reference for the metric panel, diagnostics design, and
research roadmap. Grounded in the production system: 33-feature LambdaRank +
grid blend (w=0.55, weather-gated), 48/71 winner, MAE 3.262, top3_hit 0.9155,
spearman 0.679, Platt-calibrated win probabilities.

## Part 1 — Metric panel

The central deficiency of the current evaluation is RESOLUTION, not coverage.
A 71-race winner metric has a ±2-race noise floor; nearly every recent
experiment landed inside it. Pairwise metrics evaluate ~190 pairs per race
(~13,500 total) — two orders of magnitude more statistical resolution from the
same 71 races. That, not metric variety, is the argument for the redesign.

| metric | measures | why it matters | weakness | production? |
|---|---|---|---|---|
| Winner hit rate | top-1 binary | headline; what users judge | knife-edge; ±1 race = 1.4pp noise; blind to rest of order | KEEP, never as sole gate |
| Podium overlap (have) | top-3 set intersection | podium display quality | order-blind within top 3 | KEEP |
| Top-3 hit (winner in pred top-3) | top-of-order safety | already 0.9155 → near ceiling | low discrimination | report only |
| Top-5 set overlap | broader front group | points-paying sharp end | even nearer ceiling | report only |
| MAE (have) | mean |pred − actual| | whole-order magnitude | dominated by midfield churn + DNF outliers | KEEP |
| Median + P90 abs error | robust / tail error | separates "typically close" from "occasionally awful" | none serious | ADD |
| Spearman (have) | monotone whole-field correlation | continuity with history | weights P15 ordering same as P1 | KEEP for continuity |
| Kendall tau | pairwise concordance | — | linear transform of pairwise accuracy → duplicate | SKIP (use pairwise acc) |
| **Pairwise accuracy** | % of driver pairs ordered correctly | closest to what LambdaRank optimizes; decomposable by pair type; high resolution | needs a DNF convention | **ADD — the key upgrade** |
| NDCG@5 (label 25−pos) | top-weighted order quality | matches training objective | opaque; gain function arbitrary | research metric only |
| Win-prob Brier / reliability (have) | probability honesty | product trust; already validated OOS | 71 winners → coarse bins | KEEP; extend to podium/points probs |

Complementarity: winner (top-1) · podium overlap (top-3 set) · pairwise accuracy
(whole order, granular) · MAE+median (magnitude) · NDCG@5 (top-weighted) ·
Brier (probabilities). Spearman is retained only so history stays comparable.
Rule: an experiment "improves ranking quality" only if pairwise accuracy or
MAE moves outside a race-clustered bootstrap band — winner_hit alone never decides again.

## Part 2 — Residual analysis pipeline

Unit: signed residual r = predicted_rank − actual_finish per driver-race
(negative = over-predicted / flattered). Pipeline: one pass over the walk-forward
producing a residual table, then grouped reports with (mean signed bias, MAE, n,
bootstrap CI), rendered as markdown + heatmap visuals.

Dimensions (all joinable from existing data):
- Driver / team bias (shrink small-n cells; flag rookies separately — their
  features are cold-started and they are the most likely genuine bias).
- Track taxonomy: street set {Monaco, Baku, Marina Bay, Jeddah, Vegas, Miami};
  overtaking terciles (track_overtakes_avg); degradation terciles (deg data);
  qualifying importance (per-track grid–finish correlation, already computed
  for the tracks page).
- Weather: dry / wet (race_meta); "mixed" requires in-race rainfall — approximate
  with wet_race + rainfall minutes if available, else defer.
- Grid bands: pole, P2, P3–P5, P6–P10, P11–P15, P16+ (bias here directly
  audits the blend's grid anchor).
- Race type: sprint vs conventional (flag exists), night races (static list),
  altitude (static: Mexico City, São Paulo, Spielberg partial).

Statistical discipline (mandatory): ~40 cells × n≤71 races means ~2 spurious
"significant biases" at p<0.05 are EXPECTED. A bias is actionable only if it
(a) persists in both halves of the backtest, and (b) survives a
multiple-comparison-aware threshold. The pipeline must print this caveat on
every report — the residual atlas is a hypothesis generator, not a verdict machine.
Note also: rank residuals are zero-sum within a race; driver biases must be
read jointly (someone flattered means someone else slighted).

## Part 3 — Pairwise ranking analysis

For each race, all C(n,2)≈190 ordered pairs. Convention: pairs where either
driver DNF'd are scored separately ("reliability-contaminated") — skill ordering
and attrition luck must not be mixed.

Buckets to report accuracy on:
- Teammates (~10 pairs/race; the cleanest car-controlled skill test)
- Actual-margin buckets: adjacent (1–2 positions apart), mid (3–5), distant (6+)
  → the accuracy-vs-margin curve locates the model's resolution limit; adjacent
  pairs near 55–60% are largely irreducible, distant pairs below ~90% are signal gaps
- Field tiers by actual finish: top-4 internal, midfield (5–14) internal,
  backmarker internal, cross-tier
- Most-reversed recurring driver pairs (min 8 meetings) — likely to surface
  specific biases (e.g., a driver whose quali flatters race results)

Expected structure worth verifying, not assuming: LambdaRank's NDCG-discounted
gradients should make top pairs better than midfield pairs; if midfield internal
accuracy is near coin-flip, most of the MAE is irreducible churn and MAE-targeted
feature work is wasted — that single finding would redirect the roadmap.

This is the highest-value item in the whole review: it converts the ±1-race
undecidability that killed five experiments into ~13.5k-sample decisions, and
it gives every future feature a named target bucket.

## Part 4 — Position-delta model (finish − grid): honest evaluation

Advantages in principle: directly models racecraft/recovery (the 2/13 comeback
bucket); decouples from the grid anchor; regression target with more variance.

Disadvantages, grounded in this project's record:
1. Delta error double-counts grid information at recomposition (grid + delta →
   order), and delta is chaos-dominated (SC timing, others' DNFs) — a noisier
   target than position.
2. Three prior experiments already probed this territory and failed: grid_penalty
   (neutral), grid–pace mismatch ranks (refuted), model-heavy blends (whole-order
   cost). The residual signal a delta model would chase has repeatedly measured ~0.
3. 2k rows across a second model + a new blend weight tuned on 71 races —
   context-blend showed exactly this kind of tuning does not validate OOS.
4. Delta is asymmetrically bounded by grid slot (pole can only lose) — modeling
   it properly conditional on grid reinvents the ranker.

Verdict: LOW priority. Do not prototype now. Revisit only if Part 3 shows pair
errors concentrated on high-|delta| drivers — that would be the first evidence
the ranker specifically fails at movement rather than order.

## Part 5 — Teammate-relative feature family

Grounding: quali_gap_to_teammate is ALREADY in production and historically
top-3 by importance — this family has a proven member. The theoretical basis is
the strongest available: the car is the dominant confounder in every finishing
statistic, and teammate deltas are the only construct that cancels it exactly.

- Theoretically strongest: rolling race-pace delta vs teammate (pace_delta_ms
  difference within team, ~5-race window) — a car-free driver-speed signal that
  no current feature isolates; predicted effect: teammate-pair accuracy (measurable
  in Part 3). Second: rolling head-to-head finish record.
- Likely redundant: average finish vs teammate (implicitly encoded by
  form_finish_5 minus team_form_finish_5, both in the model); quali gap variants
  beyond the existing feature.
- Cautious: teammate practice-pace delta — cancels fuel/program bias (same car,
  same convention), which fixes the exact failure mode that sank the practice
  pipeline, but inherits its coverage holes; only worth revisiting on that thesis.
- Weak: "improvement vs teammate" trajectories — rookie-thin data.

Risks: half the sample of driver-level features (2 cars), broken by mid-season
swaps, and marginal-information questions vs the existing quali gap. Worth
investigating? Yes — exactly ONE candidate (rolling race-pace-vs-teammate delta),
and only after the pairwise framework exists to judge it on teammate-pair
accuracy rather than the winner knife-edge.

## Part 6 — Automatic diagnostics report

Formalize what has been done by hand ~10 times. After every walk-forward
evaluation, auto-generate (markdown + JSON, next to the experiment artifacts):
1. Metric panel deltas vs frozen baseline, each with its noise band
   (race-clustered bootstrap; winner_hit also gets McNemar on discordant races).
2. Flip list: races gained/lost with winner, grid slot, weather/SC context, and
   the miss-taxonomy tag of each flipped race.
3. Bucket migration: which taxonomy buckets shrank/grew (did we fix the bucket
   we aimed at, or win randomly elsewhere? — the quali_surprise lesson).
4. Slice deltas: per-season, per-track-type, per-driver (min-n gated), pairwise
   buckets from Part 3.
5. Churn ratio: total flips / |net change| (e.g., 6 flips for net +1 = 6.0 —
   high churn, weak signal; the practice-pace experiments would have scored
   terribly here and been rejected in one glance).
6. Calibration drift: Brier + reliability bins vs baseline.
Verdict line: SIGNAL / CHURN / REGRESSION, from pre-registered thresholds.

## Part 7 — Prioritized roadmap (by expected placement-quality value)

1. **Pairwise framework + metric panel** (Parts 1+3). Upside: converts
   undecidable experiments into decidable ones; locates the model's real
   resolution limit. Downside: none (measurement). Complexity: low. Overfit
   risk: none. Prototype: immediately. Outperforms any feature experiment
   because it changes the value of every subsequent one.
2. **Auto-diagnostics report** (Part 6). Same sprint; prevents shipping churn.
3. **Residual bias atlas** (Part 2). Upside: likely 1–2 real, persistent biases
   (rookie cold-start and wet are the priors). Risk: false positives — handled
   by the persistence rule. Complexity: low-medium.
4. **Podium/points probability calibration.** The win-prob recipe validated OOS
   in both splits; extending it is known-good method with direct product value.
   Low risk, low complexity. Prototype: yes.
5. **Teammate rolling race-pace delta** (Part 5's single survivor). The only
   feature idea admitted, because it targets a named measurable bucket
   (teammate-pair accuracy) and cancels the car confounder. Moderate overfit
   risk; judge only through the new panel. Prototype: after 1–2.
6. **Positional sigma structure** (front vs midfield noise bands fit on backtest
   residuals — NOT per-driver, which is refuted). Targets MAE realism and
   probability sharpness; judge on Brier + pairwise, never winner. Medium risk.
7. **Position-delta model** (Part 4): parked pending pairwise evidence.
8. **Not recommended:** neural rankers (2k rows; LambdaRank is the right
   inductive bias at this scale and a transformer would need two orders of
   magnitude more data), any further momentum/gating/context-weight variants
   (all refuted with validation), features without a named failure mechanism.

The unifying judgment: this project's constraint is measurement resolution, not
model capacity or feature imagination. Build the instruments first; the next
genuine improvement will be found through them or not at all.
