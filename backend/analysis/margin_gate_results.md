# Experiment E2: Margin-Aware Override Gating - Results

Post-processing on cached walk-forward raw scores (variant `baseline`, w=0.55).
No model fitting. Gate implemented as a **top-pick swap** (exchange pick and
pole ranks; all other ranks unchanged).

## 1. Baseline reproduction

- winner_hit: **46/71** = 0.6479  (target 46/71 = 0.6479) -> MATCH
- overrides (blend pick != pole): **19**  | correct(hit) 10 | wrong(pole won) 5 | other(neither won) 4
- grid leader won in 5/19 overrides; overriding net 10-5 = +5

### Override races (hit / pole_won / category)

| season | round | event | cat | hit | pole_won |
|---|---|---|---|---|---|
| 2023 | 11 | Hungarian Grand Prix | correct | 1 | 0 |
| 2023 | 12 | Belgian Grand Prix | other | 0 | 0 |
| 2023 | 14 | Italian Grand Prix | correct | 1 | 0 |
| 2023 | 21 | Las Vegas Grand Prix | correct | 1 | 0 |
| 2024 | 9 | Canadian Grand Prix | correct | 1 | 0 |
| 2024 | 10 | Spanish Grand Prix | correct | 1 | 0 |
| 2024 | 13 | Hungarian Grand Prix | other | 0 | 0 |
| 2024 | 14 | Belgian Grand Prix | other | 0 | 0 |
| 2024 | 15 | Dutch Grand Prix | wrong | 0 | 1 |
| 2024 | 17 | Azerbaijan Grand Prix | correct | 1 | 0 |
| 2024 | 22 | Las Vegas Grand Prix | wrong | 0 | 1 |
| 2024 | 23 | Qatar Grand Prix | correct | 1 | 0 |
| 2025 | 3 | Japanese Grand Prix | wrong | 0 | 1 |
| 2025 | 5 | Saudi Arabian Grand Prix | correct | 1 | 0 |
| 2025 | 10 | Canadian Grand Prix | wrong | 0 | 1 |
| 2025 | 12 | British Grand Prix | correct | 1 | 0 |
| 2025 | 14 | Hungarian Grand Prix | correct | 1 | 0 |
| 2025 | 16 | Italian Grand Prix | wrong | 0 | 1 |
| 2026 | 7 | Barcelona Grand Prix | other | 0 | 0 |

## 2-3. Margin values and separation

### m1: `(raw[pick]-raw[pole]) / std(raw)`

- **correct** (n=10): 1.025 1.254 1.167 0.725 0.167 0.081 0.102 0.167 0.093 0.105
- **wrong** (n=5): 0.150 0.027 0.644 0.701 0.162
- **other** (n=4): 0.917 0.307 0.867 0.041
- correct[min/med/max] = 0.081/0.167/1.254 | wrong = 0.027/0.162/0.701
- **AUC P(correct>wrong) = 0.600**  (need > ~0.70 for usable separation)

### m2: `(raw[pick]-raw[pole]) / (max(raw)-min(raw))`

- **correct** (n=10): 0.277 0.347 0.294 0.232 0.051 0.027 0.033 0.052 0.034 0.034
- **wrong** (n=5): 0.051 0.009 0.194 0.230 0.055
- **other** (n=4): 0.232 0.101 0.274 0.014
- correct[min/med/max] = 0.027/0.052/0.347 | wrong = 0.009/0.055/0.230
- **AUC P(correct>wrong) = 0.560**  (need > ~0.70 for usable separation)

### m3_T1: `softmax(raw,T=1)[pick]-softmax(raw,T=1)[pole]`

- **correct** (n=10): 0.565 0.720 0.787 0.346 0.178 0.074 0.060 0.136 0.051 0.066
- **wrong** (n=5): 0.101 0.010 0.373 0.325 0.087
- **other** (n=4): 0.078 0.204 0.117 0.025
- correct[min/med/max] = 0.051/0.157/0.787 | wrong = 0.010/0.101/0.373
- **AUC P(correct>wrong) = 0.580**  (need > ~0.70 for usable separation)

### m3_Tstd: `softmax(raw,T=std)[pick]-softmax(raw,T=std)[pole]`

- **correct** (n=10): 0.198 0.248 0.297 0.115 0.040 0.017 0.017 0.037 0.014 0.020
- **wrong** (n=5): 0.028 0.004 0.120 0.112 0.028
- **other** (n=4): 0.086 0.060 0.076 0.007
- correct[min/med/max] = 0.014/0.039/0.297 | wrong = 0.004/0.028/0.120
- **AUC P(correct>wrong) = 0.580**  (need > ~0.70 for usable separation)

### m4: `(raw[model#1]-raw[model#2]) / std(raw)  (general top-pick conviction)`

- **correct** (n=10): 1.016 0.923 1.167 0.041 0.167 0.081 0.102 0.167 0.021 0.056
- **wrong** (n=5): 0.116 0.056 0.264 0.070 0.017
- **other** (n=4): 1.128 0.225 0.643 0.033
- correct[min/med/max] = 0.021/0.134/1.167 | wrong = 0.017/0.070/0.264
- **AUC P(correct>wrong) = 0.660**  (need > ~0.70 for usable separation)

Best-separating margin by AUC: **m4** (AUC 0.660).
Note on the 4 'other' overrides: reverting them changes nothing (both the
blend pick and the pole-sitter lost), so they are neutral for hit-rate and
excluded from the correct-vs-wrong AUC.

## 4. tau sweep (hit-rate view; override population only, others neutral)

`override_hits` = hits among the 19 override races (baseline = 10 correct).
Full-backtest winner_hit = 46 - 10 + override_hits = 36 + override_hits.

### m1  (best net +1 -> full-backtest 47/71 = 0.6620)

| tau | saved_wrong | lost_correct | net | override_hits |
|---|---|---|---|---|
| 0.0267 | 0 | 0 | +0 | 10 |
| 0.0338 | 1 | 0 | +1 | 11  <== best |
| 0.0609 | 1 | 0 | +1 | 11  <== best |
| 0.0871 | 1 | 1 | +0 | 10 |
| 0.0974 | 1 | 2 | -1 | 9 |
| 0.1034 | 1 | 3 | -2 | 8 |
| 0.1277 | 1 | 4 | -3 | 7 |
| 0.1560 | 2 | 4 | -2 | 8 |
| 0.1643 | 3 | 4 | -1 | 9 |
| 0.1670 | 3 | 5 | -2 | 8 |
| 0.2370 | 3 | 6 | -3 | 7 |
| 0.4754 | 3 | 6 | -3 | 7 |
| 0.6724 | 4 | 6 | -2 | 8 |
| 0.7129 | 5 | 6 | -1 | 9 |
| 0.7959 | 5 | 7 | -2 | 8 |
| 0.8920 | 5 | 7 | -2 | 8 |
| 0.9708 | 5 | 7 | -2 | 8 |
| 1.0961 | 5 | 8 | -3 | 7 |
| 1.2106 | 5 | 9 | -4 | 6 |
| 1.2537 | 5 | 10 | -5 | 5 |

Plateau at best net: tau in [0.0338, 0.0609] (width 0.0272, 2 candidate points).

### m2  (best net +1 -> full-backtest 47/71 = 0.6620)

| tau | saved_wrong | lost_correct | net | override_hits |
|---|---|---|---|---|
| 0.0087 | 0 | 0 | +0 | 10 |
| 0.0112 | 1 | 0 | +1 | 11  <== best |
| 0.0204 | 1 | 0 | +1 | 11  <== best |
| 0.0299 | 1 | 1 | +0 | 10 |
| 0.0333 | 1 | 2 | -1 | 9 |
| 0.0340 | 1 | 3 | -2 | 8 |
| 0.0424 | 1 | 4 | -3 | 7 |
| 0.0511 | 2 | 4 | -2 | 8 |
| 0.0517 | 2 | 5 | -3 | 7 |
| 0.0539 | 2 | 6 | -4 | 6 |
| 0.0782 | 3 | 6 | -3 | 7 |
| 0.1473 | 3 | 6 | -3 | 7 |
| 0.2118 | 4 | 6 | -2 | 8 |
| 0.2310 | 5 | 6 | -1 | 9 |
| 0.2321 | 5 | 7 | -2 | 8 |
| 0.2531 | 5 | 7 | -2 | 8 |
| 0.2756 | 5 | 7 | -2 | 8 |
| 0.2858 | 5 | 8 | -3 | 7 |
| 0.3207 | 5 | 9 | -4 | 6 |
| 0.3470 | 5 | 10 | -5 | 5 |

Plateau at best net: tau in [0.0112, 0.0204] (width 0.0091, 2 candidate points).

### m3_T1  (best net +1 -> full-backtest 47/71 = 0.6620)

| tau | saved_wrong | lost_correct | net | override_hits |
|---|---|---|---|---|
| 0.0101 | 0 | 0 | +0 | 10 |
| 0.0174 | 1 | 0 | +1 | 11  <== best |
| 0.0378 | 1 | 0 | +1 | 11  <== best |
| 0.0556 | 1 | 1 | +0 | 10 |
| 0.0633 | 1 | 2 | -1 | 9 |
| 0.0702 | 1 | 3 | -2 | 8 |
| 0.0761 | 1 | 4 | -3 | 7 |
| 0.0826 | 1 | 4 | -3 | 7 |
| 0.0941 | 2 | 4 | -2 | 8 |
| 0.1089 | 3 | 4 | -1 | 9 |
| 0.1264 | 3 | 4 | -1 | 9 |
| 0.1571 | 3 | 5 | -2 | 8 |
| 0.1910 | 3 | 6 | -3 | 7 |
| 0.2645 | 3 | 6 | -3 | 7 |
| 0.3356 | 4 | 6 | -2 | 8 |
| 0.3597 | 4 | 7 | -3 | 7 |
| 0.4691 | 5 | 7 | -2 | 8 |
| 0.6423 | 5 | 8 | -3 | 7 |
| 0.7535 | 5 | 9 | -4 | 6 |
| 0.7873 | 5 | 10 | -5 | 5 |

Plateau at best net: tau in [0.0174, 0.0378] (width 0.0204, 2 candidate points).

### m3_Tstd  (best net +1 -> full-backtest 47/71 = 0.6620)

| tau | saved_wrong | lost_correct | net | override_hits |
|---|---|---|---|---|
| 0.0038 | 0 | 0 | +0 | 10 |
| 0.0055 | 1 | 0 | +1 | 11  <== best |
| 0.0105 | 1 | 0 | +1 | 11  <== best |
| 0.0154 | 1 | 1 | +0 | 10 |
| 0.0171 | 1 | 2 | -1 | 9 |
| 0.0189 | 1 | 3 | -2 | 8 |
| 0.0242 | 1 | 4 | -3 | 7 |
| 0.0282 | 2 | 4 | -2 | 8 |
| 0.0326 | 3 | 4 | -1 | 9 |
| 0.0387 | 3 | 5 | -2 | 8 |
| 0.0502 | 3 | 6 | -3 | 7 |
| 0.0682 | 3 | 6 | -3 | 7 |
| 0.0812 | 3 | 6 | -3 | 7 |
| 0.0988 | 3 | 6 | -3 | 7 |
| 0.1135 | 4 | 6 | -2 | 8 |
| 0.1178 | 4 | 7 | -3 | 7 |
| 0.1593 | 5 | 7 | -2 | 8 |
| 0.2232 | 5 | 8 | -3 | 7 |
| 0.2728 | 5 | 9 | -4 | 6 |
| 0.2975 | 5 | 10 | -5 | 5 |

Plateau at best net: tau in [0.0055, 0.0105] (width 0.0050, 2 candidate points).

### m4  (best net +1 -> full-backtest 47/71 = 0.6620)

| tau | saved_wrong | lost_correct | net | override_hits |
|---|---|---|---|---|
| 0.0166 | 0 | 0 | +0 | 10 |
| 0.0191 | 1 | 0 | +1 | 11  <== best |
| 0.0274 | 1 | 1 | +0 | 10 |
| 0.0373 | 1 | 1 | +0 | 10 |
| 0.0485 | 1 | 2 | -1 | 9 |
| 0.0560 | 2 | 2 | +0 | 10 |
| 0.0633 | 2 | 3 | -1 | 9 |
| 0.0756 | 3 | 3 | +0 | 10 |
| 0.0913 | 3 | 4 | -1 | 9 |
| 0.1089 | 3 | 5 | -2 | 8 |
| 0.1415 | 4 | 5 | -1 | 9 |
| 0.1670 | 4 | 6 | -2 | 8 |
| 0.1961 | 4 | 7 | -3 | 7 |
| 0.2444 | 4 | 7 | -3 | 7 |
| 0.4534 | 5 | 7 | -2 | 8 |
| 0.7830 | 5 | 7 | -2 | 8 |
| 0.9696 | 5 | 8 | -3 | 7 |
| 1.0723 | 5 | 9 | -4 | 6 |
| 1.1479 | 5 | 9 | -4 | 6 |
| 1.1675 | 5 | 10 | -5 | 5 |

Plateau at best net: tau in [0.0191, 0.0191] (width 0.0000, 1 candidate points).

## Best gate overall: **m1**, best net **+1**, plateau tau in [0.0338, 0.0609].

## 5. Overfitting controls (n=19 overrides, 5 wrong)

### 7. Metric impact at m1, tau=0.0474 (top-pick swap)

| metric | baseline | gated | delta |
|---|---|---|---|
| winner_hit | 46/71 (0.6479) | 47/71 (0.6620) | +0.0141 |
| podium_overlap | 2.056 | 2.056 | +0.000 |
| mae | 3.281 | 3.278 | -0.003 |
- saved_wrong 1, lost_correct 0

### Plateau interpretation

- m1: best net +1, plateau width 0.0272 -> narrow/low-net (overfit-prone)
- m2: best net +1, plateau width 0.0091 -> narrow/low-net (overfit-prone)
- m3_T1: best net +1, plateau width 0.0204 -> narrow/low-net (overfit-prone)
- m3_Tstd: best net +1, plateau width 0.0050 -> narrow/low-net (overfit-prone)
- m4: best net +1, plateau width 0.0000 -> narrow/low-net (overfit-prone)

### Leave-one-race-out (LOO) over the 19 override races

Baseline override-hits = 10/19. Gate must beat this out-of-sample.

| margin | LOO override_hits | net vs baseline |
|---|---|---|
| m1 | 11/19 | +1 |
| m2 | 11/19 | +1 |
| m3_T1 | 11/19 | +1 |
| m3_Tstd | 11/19 | +1 |
| m4 | 8/19 | -2 |

LOO detail for best margin **m1** (event | cat | tau_on_other18 | hit):

- Hungarian Grand Prix       | correct | tau=0.0474 | hit=1
- Belgian Grand Prix         | other   | tau=0.0474 | hit=0
- Italian Grand Prix         | correct | tau=0.0474 | hit=1
- Las Vegas Grand Prix       | correct | tau=0.0474 | hit=1
- Canadian Grand Prix        | correct | tau=0.0474 | hit=1
- Spanish Grand Prix         | correct | tau=0.0474 | hit=1
- Hungarian Grand Prix       | other   | tau=0.0474 | hit=0
- Belgian Grand Prix         | other   | tau=0.0474 | hit=0
- Dutch Grand Prix           | wrong   | tau=0.0474 | hit=0
- Azerbaijan Grand Prix      | correct | tau=0.0504 | hit=1
- Las Vegas Grand Prix       | wrong   | tau=0.0509 | hit=1
- Qatar Grand Prix           | correct | tau=0.0474 | hit=1
- Japanese Grand Prix        | wrong   | tau=0.0474 | hit=0
- Saudi Arabian Grand Prix   | correct | tau=0.0474 | hit=1
- Canadian Grand Prix        | wrong   | tau=0.0474 | hit=0
- British Grand Prix         | correct | tau=0.0474 | hit=1
- Hungarian Grand Prix       | correct | tau=0.0474 | hit=1
- Italian Grand Prix         | wrong   | tau=0.0474 | hit=0
- Barcelona Grand Prix       | other   | tau=0.0539 | hit=0

### Time-split validation (best margin)

- FORWARD  (tau* on race_idx<35=10 ov, apply to 9 ov): tau*=0.1586, OOS override-hits 4 -> 2 (delta -2)
- REVERSE  (tau* on race_idx>=35=9 ov, apply to 10 ov): tau*=0.0504, OOS override-hits 6 -> 6 (delta +0)

## Verdict against success criteria

- Separation: best AUC 0.660 (m4); need >~0.70 -> FAIL (distributions overlap)
- Strong gate (save>=3 wrong, lose<=1 correct anywhere): NONE
- Best in-sample net: +1 (full-backtest 47/71); need >=+2 for CONFIRM
- LOO override-hits (m1): 11/19 (net +1); need >= +1

### REFUTE: margin distributions overlap (best AUC 0.660 < 0.70); no tau saves >=3 wrong while losing <=1 correct (the wrong overrides Suzuka/Montreal are HIGH-margin, high-conviction); best in-sample net only +1 (< +2).
