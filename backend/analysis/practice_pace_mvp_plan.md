# Practice Long-Run Pace — critical review + MVP experimental plan (2026-07-18)

Companion to practice_pace_design.md. No implementation yet. Grounded in the
2026-07-18 miss taxonomy: 4 "race pace not captured" + 2 "quali overperformance"
misses are the addressable bucket (of 23 total; 13+ are pre-race-unknowable).

## Part 1 — Hypothesis critique
FOR: race sims are the only public pre-race measurement of race-trim pace; the
capturable misses share exactly the "faster on race day than history predicted"
signature; matches the proven weekend-deviation thesis (quali_surprise adopted,
all historical-average features refuted); relative comparisons cancel shared
unknowns (track, temp, evolution).
AGAINST, by factor:
- Fuel (unobservable, ~0.03 s/kg/lap): 10 kg spread ≈ 0.3 s ≈ the size of the
  signal sought. Kills absolute times. Relative deltas survive only if fuel
  conventions are stable; STABLE team bias cancels in the surprise feature IF
  expectations are computed in practice-space (see Part 3 F2). VARIABLE
  week-to-week fuel choices are irreducible noise.
- Engine modes: post-2020 mode restrictions shrink this; residual deployment
  differences fold into the same stable-bias argument.
- Sandbagging / hiding pace: stable sandbagging cancels vs own practice norm;
  the dangerous case is upgrade weekends where a team changes behavior — exactly
  when the signal matters most. Accept as risk.
- Setup changes: parc ferme starts at quali; Friday sims run on non-final setup.
  Signal decays Friday->Sunday.
- Weather: wet practice = no sims (NaN); Friday-hot/Sunday-cool shifts tyre
  behavior. Wet-race misses will NOT be helped (they're in the weather bucket).
- Track evolution: 1-2 s grip ramp is shared -> cancels in relative deltas.
- Small n: one 6-8 lap stint, lap σ ~0.2-0.3 s -> median SE ~0.1 s. Team pooling
  doubles samples.
- Sprint weekends: single FP, race sims rare -> NaN (~6/season).
- Missing data: expect 10-20% driver-weekends NaN, non-randomly (chaotic
  weekends) -> encode missingness, never impute.
FAILURE CONDITION: if week-to-week program/fuel variation exceeds true pace
deviation, the feature is noise. Empirical question -> hence the MVP.

## Part 2 — Race-sim detection (the core heuristic)
Score each candidate stint on 4 criteria, accept score >= 3 of 4:
1. Length: >= 5 valid laps (green, IsAccurate, not deleted, not in/out).
   NOT sufficient alone — long stints also include tyre-management tests, aero
   rakes, systems checks.
2. Pace band: stint median in [102.5%, 108%] of driver's session best.
   Below -> quali/short-run trim; above -> cruising/rake/tyre-save running.
3. Consistency: detrend the stint (fit linear deg slope), require residual
   robust-std <= ~1.0% of median. Detrending matters: legitimate degradation
   must not read as inconsistency; traffic laps survive trimming as residual
   spikes -> MAD-trim then re-check.
4. Session timing: second half of FP2 (conventional weekends) = +1 evidence.
   Tiebreak only, never a gate (programs vary; sprint weekends differ).
Guard against "detects drivers who ran long": the FEATURE is pace-relative, not
participation-relative — no sim -> NaN + confidence 0 (never a penalty); deltas
computed only among drivers who ran classified sims; compound/age normalization
so a 12-lap hard run compares fairly to a 7-lap medium run.
Calibration: validate the classifier on 2-3 hand-inspected 2025 sessions against
published race-pace analyses BEFORE trusting it at scale.

## Part 3 — MVP feature set (exactly three)
F1 `practice_longrun_delta_ms`: driver's normalized race-sim median minus the
field's race-sim median that weekend. Compound handling in MVP: compute deltas
within compound vs same-compound field median, average across a driver's sims
weighted by laps; static fallback offsets only when cross-compound comparison is
unavoidable. Relative-not-absolute: cancels track/temp/evolution/shared-fuel
terms; medians for outlier robustness.
F2 `practice_pace_surprise`: F1 minus the TEAM's rolling mean of F1 over the
last 5 weekends-with-sims (shift(1), expanding — leakage-free). KEY DESIGN
POINT: expectation lives in practice-measurement-space (practice vs own past
practice), NOT practice vs race pace — stable fuel/sandbagging bias cancels by
construction. Team (not driver) expectation: car dominates, 2 cars = 2x samples,
new-driver cold start covered.
F3 `practice_sample_confidence` (0-3): 0 none; 1 one short sim (5-6 laps);
2 one solid sim (>=7) or two short; 3 multiple sims/both cars. Why feed
confidence instead of dropping: missingness is informative and non-random;
LightGBM can learn the interaction "trust F1/F2 only when confidence high";
dropping would discard weak-but-real evidence and bias the sample toward calm
weekends.

## Part 4 — Deliberately excluded from MVP
- Deg slope: 6-point slopes are noise-dominated; targets the strategy bucket we
  classified as mostly unpredictable; correlated with F1. Phase 3 at best.
- Separate team-level feature: team info already enters as F2's expectation
  baseline; a standalone team_delta would double-count with F1 + existing
  team_form features. Revisit only if driver-level NaN rate is a problem.

## Part 5 — Build order
Phase 1 (one session of work): 2025-2026 conventional weekends only (~33
events), FP2 only (+ FP3 fallback when FP2 unusable), no telemetry, fixed
compound fallbacks, hand-validate classifier on 2-3 sessions. Sprint weekends
NaN. Backfill via existing sqlite cache, rate-limit friendly.
Phase 2: standard framework, walk-forward on the 2025-2026 backtest subset
(33 eval races; current model: 21/33 there). Mechanism check FIRST (as with
quali_surprise): do the target-race winners actually show positive surprise
before any model is fit?
Phase 3 (only if Phase 2 passes): backfill 2022-2024 (adds targets Qatar '24,
Spa '23, Austin '23), full-backtest eval vs frozen 48/71, per-event compound
fixed-effects model, deg slope, telemetry traffic filter.

## Part 6 — Success criteria (pre-registered)
Phase-2 window (33 races, current 21/33):
- Mechanism: >= 2 of the 3 in-window target races (Suzuka '25, Montreal '25,
  Barcelona '26) show the winner with clearly positive F2 (top-3 of field).
  FAIL here -> abandon before Phase 3 (fuel masks the signal).
- Model: >= +2 winners on the window (>= 23/33) at w=0.55; >= 1 target race
  flips; MAE regression <= 0.05; top3 drop <= 0.01; no season in the window
  regresses by more than 1 race; NaN rate <= 25%; nonzero importance.
Full adoption (after Phase 3): >= 50/71 overall with the standard gates
(podium >= 2.05, MAE <= 3.33), improvements concentrated in the race-pace and
quali-overperformance buckets, not random churn.

## PHASE 0 RESULTS (2026-07-18, phase0_practice_validation.py — VERDICT: PASS with caveats)
Three FP2 sessions hand-validated via live FastF1 fetch (cached in data/cache):
- Bahrain '25 (dominant weekend): 20 accepted sims / 18 drivers. Winner PIA
  ranked 2nd on practice pace; Spearman(practice rank, race finish) = 0.44.
  Classifier accepts/rejects all sensible on inspection. FALSE-POSITIVE FOUND:
  ALO topped sims (-1.6 s!) and finished P15 — classic low-fuel Friday glory run;
  confirms F2 must use own-practice-norm expectations (stable glory-running
  cancels), and confidence weighting matters.
- Jeddah '25: default thresholds too strict (5 clean laps) for a night/street
  program — retuned to MIN_LAPS=4, trim 3.5% → 3 sims: PIA -975 ms vs VER,
  SAI +419. Race: PIA beat VER. **Practice pace correctly called an actual
  model MISS** (Jeddah '25 was classified "unpredictable/penalty" — partially
  capturable after all). Coverage caveat: only 3 of 20 drivers.
- Suzuka '25 (target race): UNRESCUABLE — FP2 red-flag-disrupted, zero race
  sims at any threshold. The feature is NaN exactly on a target race; the
  missing-programs failure mode is real and hits where it hurts.
Learnings for Phase 1: min clean laps = 4; circuit-adaptive trim; expect
partial (3-10 driver) coverage on many weekends and full NaN on some; the
Phase-2 mechanism gate must be judged on sims-available target races only.

## MISS-OVERLAP RESULTS (2026-07-18, miss_practice_overlap.py + .csv)
Practice signal computed for all 23 current misses (winner's practice rank of covered drivers):
- PRACTICE COULD FIX (winner top-of-field in sims, quali_surprise did NOT flag): Miami '24
  (NOR 1/15, qsurp -0.2 — the upgrade weekend), Austin '24 (LEC 1/16, qsurp 0.4),
  Jeddah '25 (PIA fastest sim by ~1 s, qsurp -0.2; from Phase 0 — overlap script's
  FP3-fallback clobbered FP2, fix in Phase 1), Silverstone '26 (LEC 1/19, -2.4 s!),
  partial: Miami '25 (PIA 4/15), Vegas '25 (VER 3/8, but DSQ-decided).
- QUALI ALREADY CAPTURES (practice duplicates): Montreal '25 (qsurp 4.8, practice 4/14 mild),
  Barcelona '26 (qsurp 2.6, practice 4/13 mild).
- PRACTICE ACTIVELY MISLEADS — VER sprint-FP1 false negatives: Austin '23 (15/18),
  Qatar '24 (16/16), Sao Paulo '24 (14/16), Qatar '25 (15/17). All four are Verstappen
  on sprint weekends: Red Bull skips FP1 long runs. TEAM-PROGRAM BIAS IS REAL and
  concentrated; the F2-vs-own-practice-norm design (stable program bias cancels) plus
  low confidence on sprint FP1 is mandatory, else the feature would have WORSENED these.
- No usable data: Spa '23, Suzuka '25 (+ winner-not-covered: Melbourne '24, Imola '25, Spa '25).
- Remaining misses show neutral practice signal (rank 5-7 mid-field).
NET: practice adds genuinely NEW information (orthogonal to quali_surprise) on ~4-5
misses — several in buckets taxonomy called strategy/unpredictable, so the addressable
set is BIGGER than the 6 estimated. Balanced against 4 would-be false negatives that
the design must neutralize. Phase 1 proceeds with: session-fallback keep-better fix,
sprint-FP1 confidence penalty, F2 strictly vs own-practice norms.

## PHASE 1-2 RESULTS (2026-07-18) — VERDICT: NOT ADOPTED (mechanism confirmed, net negative)
Extraction: all 101 events, 72.2% driver coverage (practice_features_extract.py ->
practice_features.csv). Evaluation (practice_pace_experiment.py, standard framework,
vs frozen 48/71): practice_delta_only (delta+confidence) 47/71; practice_full
(+surprise) 46/71. Blend sweep: no w reaches 48.
- MECHANISM CONFIRMED: both in-window targets flip to hits exactly as designed —
  Jeddah '25 (PIA -975 ms sim) and Vegas '25 (VER -464 ms). 2025 season 15/24 -> 17/24.
  Top3 +0.028, podium +0.014 on delta_only.
- COSTS: Mexico '23 (knife-edge churn — winner had good sims, model preferred LEC's),
  Hungary '24 (winner PIA had NO sim, conf 0 — missingness penalty materialized),
  Vegas '24 (practice preferred SAI's Friday pace, un-fixing quali_surprise's win).
  2023 -1, 2024 -2.
- Pre-registered gates: window gate (2025-26): +2 (21->23/33) PASS, targets PASS,
  guardrails PASS, NaN 27.8% > 25% marginal FAIL. FULL adoption gate (>=50/71):
  clear FAIL. Per plan discipline: not adopted.
- Interpretation: the signal is real but the exchange rate is ~1:1 with older-season
  knife-edge losses. The feature is most informative in 2025-26 (the era that looks
  like the future); pipeline is kept and cheap to re-run — REVISIT after more 2026
  races accumulate, or with improved extraction (2022-sprint Saturday FP2, better
  missing-winner handling). Surprise variant (F2) HURT vs delta-only (69% coverage
  compounding NaN) — drop it in any revisit.

## INTEGRATION VARIANTS (2026-07-18) — all three REFUTED
1. Confidence-gated (delta NaN below conf 2): 46/71. Protects Hungary '24 but loses
   Vegas '24 anyway; top3 0.930 (+). Net -2.
2. Tie-breaker (post-processing, flip top-2 within score gap tau by better sim):
   ONLY loses at every tau. tau=0.25 -> no qualifying ties (48, no-op); 0.5 -> -1
   (Mexico '24); 0.75-1.0 -> -3; 1.5 -> -7 with Jeddah finally gained. The gains
   the walk-forward variant found are NOT reachable by post-hoc tie-breaking —
   the baseline's top-2 gaps at the target races are too wide.
3. Practice/quali disagreement (percentile-space rank feature): 43/71 (-5). Worst.
   Rank-space interaction features fail again (cf. grid-pace mismatch).
CONCLUSION: five practice integrations tested (2 feature variants + 3 schemes);
none beats 48/71. The signal is real (mechanism confirmed on targets) but the
marginal economics are ~1:1 against the current backtest composition. STOP here;
revisit mid-2026 when the era where the signal works (2025+: 15->17/24) weighs more.

## Final recommendation
1. Worth building: YES, staged — it is the only remaining lever aimed at a
   measured, capturable bucket; everything cheaper is refuted on record.
2. Realistic upside: +2 to +4 winners full-backtest (0.676 -> ~0.70-0.73);
   information ceiling ~0.77-0.82 stands regardless.
3. Biggest technical risk: NOT fuel (bounded by the relative+practice-space
   design) but race-sim MISCLASSIFICATION — one quali-prep or tyre-save stint
   misread as a race sim corrupts the weekend's delta without bound. Hence
   hand-validation before scale.
4. Smallest validating experiment: before any pipeline — pull TWO known 2025
   sessions (one where a target-race winner impressed, e.g. Suzuka '25 VER;
   one boring weekend), run the classifier by hand, and check the computed
   deltas against published race-pace tables. Half a day; kills or confirms
   the classifier design immediately.
5. Abandon if: the mechanism check fails on the target races (winners show no
   practice edge — signal is fuel-masked), or Phase 2 lands < +1 on the window,
   or classifier hand-validation can't reach agreement with published analyses
   on 2 of 3 sessions.
