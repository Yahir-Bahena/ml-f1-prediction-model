# Grid-Pace Mismatch Experiment - Results

## Leakage note
Both inputs to every mismatch feature are pre-race-safe:
- `pace_delta_5` is a rolling mean of PRIOR races' pace_delta_ms (strictly past races).
- `quali_pos` / `stint1_pace_5` are known before the race (quali happens pre-race;
  stint1_pace_5 is also a rolling PRIOR-race mean).
Ranking these WITHIN a race only reorders values already known pre-race, so it
introduces no future information -> leakage-free.

### Input NaN coverage
- pace_delta_5: NaN rate 0.0162
- stint1_pace_5: NaN rate 0.0182
- pace_rank_in_race: NaN rate 0.0162
- pace_grid_mismatch: NaN rate 0.0182
- stint1_rank_in_race: NaN rate 0.0182
- stint1_grid_mismatch: NaN rate 0.0201

(stint1_pace_5 NaN 1.8% << 50% threshold -> stint variant kept.)

## Variant comparison (walk-forward, BLEND_W=0.55)

| variant | hit_rate | hits | pod | mae | spear | grid>=3 | pole | dnf-adj |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.6479 | 46/71 | 2.06 | 3.28 | 0.677 | 2/13 | 36/41 | 46/67=0.687 |
| a_mismatch | 0.6338 | 45/71 | 2.06 | 3.25 | 0.681 | 1/13 | 36/41 | 45/68=0.662 |
| b_mismatch_rank | 0.6620 | 47/71 | 2.04 | 3.26 | 0.680 | 2/13 | 38/41 | 47/68=0.691 |
| c_stint | 0.6056 | 43/71 | 2.01 | 3.25 | 0.680 | 2/13 | 34/41 | 43/67=0.642 |
| d_both | 0.6197 | 44/71 | 2.01 | 3.29 | 0.679 | 1/13 | 36/41 | 44/68=0.647 |

Baseline reproduction: 0.6479 (target 0.6479) -> MATCH

## Per-season hit rates (BLEND_W=0.55)

| season | baseline | a_mismatch | b_mismatch_rank | c_stint | d_both |
|---|---|---|---|---|---|
| 2023 | 11/14 | 12/14 | 12/14 | 12/14 | 12/14 |
| 2024 | 13/24 | 13/24 | 13/24 | 13/24 | 12/24 |
| 2025 | 16/24 | 14/24 | 16/24 | 12/24 | 14/24 |
| 2026 | 6/9 | 6/9 | 6/9 | 6/9 | 6/9 |

## Best variant by (hit_rate, grid>=3, pole) at w=0.55: **b_mismatch_rank**

## BLEND_W sweep

| w | baseline hit | b_mismatch_rank hit | b_mismatch_rank grid>=3 | b_mismatch_rank pole | b_mismatch_rank pod | b_mismatch_rank mae |
|---|---|---|---|---|---|---|
| 0.45 | 0.5775 | 0.5915 | 0/13 | 40/41 | 2.03 | 3.26 |
| 0.50 | 0.5775 | 0.5915 | 0/13 | 40/41 | 2.03 | 3.26 |
| 0.55 | 0.6479 | 0.6620 | 2/13 | 38/41 | 2.04 | 3.26 |
| 0.60 | 0.6479 | 0.6620 | 2/13 | 38/41 | 2.04 | 3.27 |
| 0.65 | 0.6338 | 0.6479 | 2/13 | 37/41 | 2.04 | 3.31 |
| 0.70 | 0.6479 | 0.6479 | 4/13 | 35/41 | 2.03 | 3.36 |
| 0.80 | 0.6479 | 0.6479 | 4/13 | 35/41 | 2.00 | 3.43 |
| 0.90 | 0.6620 | 0.6620 | 6/13 | 34/41 | 2.00 | 3.48 |
| 1.00 | 0.6620 | 0.6620 | 6/13 | 34/41 | 2.00 | 3.50 |

Best w for b_mismatch_rank: **0.9** -> hit 0.6620, grid>=3 6/13, pole 34/41

## Flipped races (b_mismatch_rank @ w=0.9 vs baseline @ w=0.55)

GAINED (5):
  + 2023 R12 Belgian Grand Prix (winner grid 6, now predicts VER)
  + 2023 R18 United States Grand Prix (winner grid 6, now predicts VER)
  + 2023 R19 Mexico City Grand Prix (winner grid 3, now predicts VER)
  + 2024 R15 Dutch Grand Prix (winner grid 1, now predicts NOR)
  + 2025 R6 Miami Grand Prix (winner grid 4, now predicts PIA)
LOST (4):
  - 2024 R8 Monaco Grand Prix (winner grid 1)
  - 2024 R20 Mexico City Grand Prix (winner grid 1)
  - 2024 R23 Qatar Grand Prix (winner grid 2)
  - 2025 R2 Chinese Grand Prix (winner grid 1)

### Feature-only flips (b_mismatch_rank vs baseline, both w=0.55): gained 3, lost 2
  + 2023 R19 Mexico City Grand Prix (winner grid 3)
  + 2024 R15 Dutch Grand Prix (winner grid 1)
  + 2025 R16 Italian Grand Prix (winner grid 1)
  - 2024 R23 Qatar Grand Prix (winner grid 2)
  - 2025 R12 British Grand Prix (winner grid 3)

## Feature importance of new features (final full-history fit, gain-normalized split count)

**a_mismatch**:
  - pace_grid_mismatch: importance 147 (2.6% of total, rank 22/29)
**b_mismatch_rank**:
  - pace_grid_mismatch: importance 141 (2.5% of total, rank 24/30)
  - pace_rank_in_race: importance 102 (1.8% of total, rank 26/30)
**c_stint**:
  - stint1_grid_mismatch: importance 157 (2.8% of total, rank 20/30)
  - stint1_rank_in_race: importance 82 (1.5% of total, rank 26/30)
**d_both**:
  - pace_grid_mismatch: importance 134 (2.4% of total, rank 23/31)
  - pace_rank_in_race: importance 94 (1.7% of total, rank 27/31)
  - stint1_grid_mismatch: importance 110 (2.0% of total, rank 26/31)

## Verdict against success criteria

- grid>=3 winners recovered vs baseline: 2 -> 6 (delta +4); need >=3 recovered [PASS]
- pole-winners lost vs baseline: 36 -> 34 (lost 2); need <=1 lost [FAIL]
- overall hit >= 0.648 at some w: best b_mismatch_rank hits 0.6620 at w=0.9 [PASS]

### REFUTE: success criteria not met (see per-criterion flags above).
