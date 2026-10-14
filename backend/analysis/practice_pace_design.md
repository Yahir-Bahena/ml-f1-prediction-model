# Design: practice-session long-run pace ingestion (no implementation yet)

Target: the 4 "race pace not captured" + 2 "quali overperformance" misses in the
2026-07-18 taxonomy — the only remaining pre-race-capturable bucket. Design honors
the project constraint (2022+ only) and the deviation-from-expectation thesis.

## 1. What FastF1 provides (verified against installed 3.8.3)
`get_session(year, round, 'FP1'|'FP2'|'FP3')` → `session.load(laps=True,
telemetry=False, weather=True)` gives per-lap:
- `Stint` (pre-numbered per tyre run), `Compound`, `TyreLife`, `FreshTyre`
- `PitOutTime` / `PitInTime` — non-null exactly on out-laps / in-laps
- `TrackStatus` (yellow/SC/VSC codes), `IsAccurate`, `Deleted`+`DeletedReason`
- `LapTime`, sector times, speed traps; `session.weather_data` (air/track temp, rainfall)
Coverage: live-timing archive from 2018 (we need 2022+). NOT available: fuel load,
engine modes, run-plan labels. Sprint weekends have a single FP session.

## 2. Long-run stint detection
Group by (driver, Stint). Valid lap = not out/in lap, not Deleted, IsAccurate,
TrackStatus == '1' (green). Long run = stint with >= 5 (tunable 5-7) valid laps on
one compound. Traffic/cooldown rejection inside a stint: drop laps > stint
median + ~2.5% (MAD-based trim); require >= 5 survivors.

## 3. Race sim vs quali sim (no labels exist — heuristic classifier)
A stint is a RACE SIM if: length >= 5 valid laps AND median lap >= ~102.5% of the
driver's session-best lap AND lap-to-lap std below a consistency ceiling. Quali
sims are 1-3 flying laps near session best on softs. Secondary signal: race sims
cluster in the second half of FP2 (traditional program); use as tiebreak, not gate.
Tunables calibrated once on a handful of manually-inspected sessions.

## 4. Tyre compound normalization
Per event, fit a fixed-effects model on all race-sim laps:
  laptime ~ stint_intercept(driver,stint) + compound_offset + tyre_age_slope(compound)
Normalize every stint to reference compound (MEDIUM) at fixed tyre age (e.g. lap 3).
Thin-sample fallback: prior events' compound offsets at that track, else Pirelli
nominal deltas. All comparisons after normalization.

## 5. Fuel-load uncertainty (the irreducible one)
Fuel is unobservable (~0.03 s/kg/lap; low-fuel vs race-fuel ≈ 2.5-3.5 s). Mitigations:
- Compare race sims ONLY against race sims (teams broadly run comparable fuel there).
- Features are RELATIVE: driver/team long-run median minus the field's race-sim
  median that weekend — common fuel conventions cancel; residual ~0.3-0.5 s is noise.
- Robust statistics (median, not mean); never mix quali-sim laps in.
Accept: fuel/engine-mode games put a floor on precision. Features must be ranks or
deltas, never absolute seconds.

## 6. Traffic and in/out-lap rejection
Handled by 2 (PitIn/PitOut nulls, TrackStatus, IsAccurate, MAD trim). Telemetry
gap-to-car-ahead is available but adds a telemetry download per session — skip in
v1; MAD trim empirically suffices for medians.

## 7. Derived features (per driver-weekend, all pre-race)
- `practice_longrun_delta_ms` — normalized race-sim median vs field median (core)
- `practice_pace_surprise` — the above minus the team/driver rolling pace norm
  (pace_delta_5-style expectation) → the weekend-deviation feature
- `team_practice_longrun_delta_ms` — 2-car pooled version (more robust, fewer NaN)
- `practice_deg_slope` — normalized within-stint degradation (strategy proxy)
- `practice_longrun_laps` — sample size, lets the model discount thin evidence
NaN when no race sim detected (LightGBM-native). Serving: FP sessions finish
before quali; ingest after FP2 (or sole FP on sprint weekends) → available pre-race.

## 8. Unavoidable failure modes
- Sandbagging / engine-mode games (systematic, team-specific, unobservable)
- Missing programs: wet practice, red flags, breakdowns, rookie FP1 drivers —
  expect ~10-20% of driver-weekends with no usable race sim (NaN)
- Sprint weekends: one FP session, often no race sim at all
- Post-practice setup changes (parc ferme only starts at quali) and track/weather
  evolution between Friday and Sunday
- Small n: one 6-8 lap stint is a noisy estimate; feature must carry sample size

## 9. Implementation complexity (largest remaining build)
- Ingest: ~101 events x 2-3 sessions, rate-limited API + sqlite cache; backfill is
  hours of wall-clock, one-time
- Stint pipeline + per-event fixed-effects normalization: the real engineering
- Join into dataset build + walk-forward eval via the standard framework
Estimate: 3-4 focused work sessions. Tuning risk: heuristics (102.5%, trim
thresholds) need validation against a few hand-inspected sessions per era.

## 10. Recommendation: YES, staged
This is the only lever left with real ceiling: measured addressable bucket is 4-6
races; realistic gain +2 to +4 winner hits (0.676 → ~0.70-0.73) against a hard
information limit near 0.77-0.82. Everything cheaper has been tried and is on
record as refuted. BUT de-risk it: **prototype on 2025-2026 only** (~33 events,
small backfill), evaluate with the standard framework against the 48/71 baseline,
and only backfill 2022-2024 if the prototype moves the metric. If the prototype
is flat, the remaining misses were fuel-masked and the full build isn't justified.
