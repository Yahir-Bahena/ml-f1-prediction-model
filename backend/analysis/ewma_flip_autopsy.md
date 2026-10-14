# EWMA / momentum experiment — per-race winner-flip autopsy

- baseline: 32 production FEATURE_COLS
- variant : baseline + 7 EWMA/trend features (ewma_finish_3, ewma_pace_3, team_ewma_finish_3, team_ewma_pace_3, pace_trend_5, finish_trend_5, team_pace_trend_5)
- blend w = 0.55 (production); predicted winner = argmin blend score

Baseline walk-forward: 71 races (cache).
Variant  walk-forward: 71 races (fresh refit).

Overall winner_hit — baseline 46/71, variant 44/71.

## Definitive flip list — 9 races where the predicted winner changed

| season | round | event | actual winner | baseline pick | variant pick | outcome |
|---|---|---|---|---|---|---|
| 2023 | 18 | United States Grand Prix | VER | NOR | LEC | neutral (both wrong) |
| 2024 | 13 | Hungarian Grand Prix | PIA | VER | NOR | neutral (both wrong) |
| 2024 | 14 | Belgian Grand Prix | HAM | PIA | NOR | neutral (both wrong) |
| 2024 | 20 | Mexico City Grand Prix | SAI | SAI | NOR | LOST (baseline right) |
| 2024 | 23 | Qatar Grand Prix | VER | VER | RUS | LOST (baseline right) |
| 2025 | 12 | British Grand Prix | NOR | PIA | NOR | GAINED (variant right) |
| 2025 | 14 | Hungarian Grand Prix | NOR | NOR | PIA | LOST (baseline right) |
| 2025 | 16 | Italian Grand Prix | VER | VER | NOR | LOST (baseline right) |
| 2025 | 23 | Qatar Grand Prix | VER | PIA | VER | GAINED (variant right) |

---

## 2023 R18 — United States Grand Prix

- Actual winner: **VER** (grid 6, finish 1)
- Baseline predicted winner: **NOR**  | variant predicted winner: **LEC**
- Actual winner blend rank — baseline 3, variant 3

### Contender scores (baseline model -> variant model)

| driver | model_rank B->V | raw B->V | blend_rank B->V | blend_score B->V |
|---|---|---|---|---|
| NOR | 2 -> 3 | +1.010 -> +0.858 | 1 -> 2 | 2.000 -> 2.550 |
| LEC | 3 -> 2 | +0.843 -> +0.973 | 2 -> 1 | 2.100 -> 1.550 |
| VER | 1 -> 1 | +1.958 -> +1.916 | 3 -> 3 | 3.250 -> 3.250 |

### NOR — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 4.939 | +0.0770 |
| ewma_pace_3 | -710.892 | +0.1266 |
| team_ewma_finish_3 | 6.273 | -0.1471 |
| team_ewma_pace_3 | -521.128 | +0.0452 |
| pace_trend_5 | -376.250 | +0.1567 |
| finish_trend_5 | -1.400 | +0.0731 |
| team_pace_trend_5 | -312.700 | -0.0229 |

Sum of 7 new-feature contribs = +0.3085. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| quali_pos | +1.1292 | +0.8972 | -0.2320 |
| form_finish_3 | +0.1031 | +0.0029 | -0.1003 |
| wet_delta | +0.3388 | +0.2583 | -0.0805 |

### LEC — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 6.093 | +0.0362 |
| ewma_pace_3 | -454.741 | -0.0405 |
| team_ewma_finish_3 | 7.540 | -0.1718 |
| team_ewma_pace_3 | -390.184 | +0.0557 |
| pace_trend_5 | -408.850 | +0.1276 |
| finish_trend_5 | -2.800 | +0.2540 |
| team_pace_trend_5 | -193.350 | -0.0422 |

Sum of 7 new-feature contribs = +0.2191. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| quali_pos | +1.5350 | +1.3444 | -0.1906 |
| form_finish_3 | +0.1418 | -0.0017 | -0.1435 |
| form_quali_5 | +0.1474 | +0.2489 | +0.1015 |

### VER — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 1.564 | +0.1512 |
| ewma_pace_3 | -933.986 | +0.3856 |
| team_ewma_finish_3 | 4.807 | +0.2918 |
| team_ewma_pace_3 | -535.599 | +0.0315 |
| pace_trend_5 | 117.900 | -0.0188 |
| finish_trend_5 | 0.000 | -0.0468 |
| team_pace_trend_5 | 267.500 | +0.0227 |

Sum of 7 new-feature contribs = +0.8173. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| form_points_5 | +0.3353 | +0.1313 | -0.2040 |
| pace_delta_5 | +0.2469 | +0.0539 | -0.1931 |
| team_season_points | +0.3355 | +0.1727 | -0.1628 |

### Context feature values

| feature | NOR | LEC | VER |
|---|---|---|---|
| ewma_finish_3 | 4.939 | 6.093 | 1.564 |
| form_finish_5 | 4.400 | 7.200 | 1.800 |
| form_finish_3 | 2.333 | 4.333 | 2.333 |
| ewma_pace_3 | -710.892 | -454.741 | -933.986 |
| pace_delta_5 | -744.650 | -271.550 | -880.950 |
| finish_trend_5 | -1.400 | -2.800 | 0.000 |
| pace_trend_5 | -376.250 | -408.850 | 117.900 |
| team_ewma_finish_3 | 6.273 | 7.540 | 4.807 |
| team_form_finish_5 | 5.500 | 7.100 | 5.200 |
| quali_pos | 2.000 | 1.000 | 6.000 |
| grid_pos | 2.000 | 1.000 | 6.000 |
| season_rank | 7.000 | 6.000 | 1.000 |

### Recent race history before this race (last 5, from race_results)

**LEC** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2023 | 13 | Dutch Grand Prix | 9 | DNF | 1220 | True | True |
| 2023 | 14 | Italian Grand Prix | 3 | 4 | -532 | False | False |
| 2023 | 15 | Singapore Grand Prix | 3 | 4 | -660 | False | False |
| 2023 | 16 | Japanese Grand Prix | 4 | 4 | -591 | False | False |
| 2023 | 17 | Qatar Grand Prix | 5 | 5 | -795 | False | False |

**NOR** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2023 | 13 | Dutch Grand Prix | 2 | 7 | -240 | False | True |
| 2023 | 14 | Italian Grand Prix | 9 | 8 | 0 | False | False |
| 2023 | 15 | Singapore Grand Prix | 4 | 2 | -855 | False | False |
| 2023 | 16 | Japanese Grand Prix | 3 | 2 | -1016 | False | False |
| 2023 | 17 | Qatar Grand Prix | 10 | 3 | -1613 | False | False |

**VERDICT — neutral (both wrong; immaterial to winner_hit).** VER won from grid 6; both models rank him 3rd and neither picks him. The variant only reshuffled two wrong front-runners (NOR->LEC) via small EWMA-driven raw shifts. Not a real EWMA success or failure — a low-magnitude churn among already-wrong picks.

---

## 2024 R13 — Hungarian Grand Prix

- Actual winner: **PIA** (grid 2, finish 1)
- Baseline predicted winner: **VER**  | variant predicted winner: **NOR**
- Actual winner blend rank — baseline 2, variant 2

### Contender scores (baseline model -> variant model)

| driver | model_rank B->V | raw B->V | blend_rank B->V | blend_score B->V |
|---|---|---|---|---|
| VER | 1 -> 3 | +1.627 -> +1.384 | 1 -> 3 | 1.900 -> 3.000 |
| NOR | 3 -> 1 | +1.186 -> +1.683 | 3 -> 1 | 2.100 -> 1.000 |
| PIA | 2 -> 2 | +1.602 -> +1.527 | 2 -> 2 | 2.000 -> 2.000 |

### VER — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 2.797 | +0.0949 |
| ewma_pace_3 | -1079.091 | +0.3440 |
| team_ewma_finish_3 | 6.680 | -0.0644 |
| team_ewma_pace_3 | -685.647 | +0.2729 |
| pace_trend_5 | 119.525 | -0.0316 |
| finish_trend_5 | -0.400 | -0.0307 |
| team_pace_trend_5 | 166.550 | -0.0074 |

Sum of 7 new-feature contribs = +0.5776. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| pace_delta_5 | +0.5385 | +0.2624 | -0.2761 |
| season_rank | +0.4858 | +0.3012 | -0.1845 |
| quali_gap_to_teammate | +0.0073 | +0.1390 | +0.1317 |

### NOR — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 5.877 | +0.0696 |
| ewma_pace_3 | -1120.665 | +0.3234 |
| team_ewma_finish_3 | 5.517 | +0.1504 |
| team_ewma_pace_3 | -995.402 | +0.2403 |
| pace_trend_5 | 220.775 | +0.0091 |
| finish_trend_5 | 1.600 | -0.0767 |
| team_pace_trend_5 | 223.050 | -0.0073 |

Sum of 7 new-feature contribs = +0.7088. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| pace_delta_5 | +0.5581 | +0.2938 | -0.2644 |
| team_form_finish_5 | +0.1378 | +0.0469 | -0.0909 |
| quali_gap_to_teammate | +0.0185 | +0.0996 | +0.0811 |

### PIA — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 5.154 | +0.0968 |
| ewma_pace_3 | -869.266 | +0.1814 |
| team_ewma_finish_3 | 5.517 | +0.1716 |
| team_ewma_pace_3 | -995.402 | +0.2808 |
| pace_trend_5 | 225.325 | +0.0546 |
| finish_trend_5 | 0.100 | -0.0666 |
| team_pace_trend_5 | 223.050 | -0.0099 |

Sum of 7 new-feature contribs = +0.7087. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| pace_delta_5 | +0.6275 | +0.3061 | -0.3213 |
| team_form_finish_5 | +0.2316 | +0.0923 | -0.1394 |
| team_dnf_rate | +0.2443 | +0.1357 | -0.1087 |

### Context feature values

| feature | VER | NOR | PIA |
|---|---|---|---|
| ewma_finish_3 | 2.797 | 5.877 | 5.154 |
| form_finish_5 | 3.000 | 6.200 | 4.000 |
| form_finish_3 | 2.667 | 8.333 | 4.333 |
| ewma_pace_3 | -1079.091 | -1120.665 | -869.266 |
| pace_delta_5 | -1023.950 | -1223.450 | -1094.650 |
| finish_trend_5 | -0.400 | 1.600 | 0.100 |
| pace_trend_5 | 119.525 | 220.775 | 225.325 |
| team_ewma_finish_3 | 6.680 | 5.517 | 5.517 |
| team_form_finish_5 | 8.300 | 5.100 | 5.100 |
| quali_pos | 3.000 | 1.000 | 2.000 |
| grid_pos | 3.000 | 1.000 | 2.000 |
| season_rank | 1.000 | 2.000 | 5.000 |

### Recent race history before this race (last 5, from race_results)

**VER** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2024 | 8 | Monaco Grand Prix | 6 | 6 | -1436 | False | False |
| 2024 | 9 | Canadian Grand Prix | 2 | 1 | -883 | False | True |
| 2024 | 10 | Spanish Grand Prix | 2 | 1 | -1025 | False | True |
| 2024 | 11 | Austrian Grand Prix | 1 | 5 | -993 | False | True |
| 2024 | 12 | British Grand Prix | 4 | 2 | -783 | False | True |

**NOR** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2024 | 8 | Monaco Grand Prix | 4 | 4 | -1586 | False | False |
| 2024 | 9 | Canadian Grand Prix | 3 | 2 | -1658 | False | True |
| 2024 | 10 | Spanish Grand Prix | 1 | 2 | -1135 | False | True |
| 2024 | 11 | Austrian Grand Prix | 2 | 20 | -852 | False | True |
| 2024 | 12 | British Grand Prix | 3 | 3 | -886 | False | True |

**VERDICT — neutral (both wrong; immaterial).** PIA won from grid 2; both models rank him 2nd. Baseline VER -> variant NOR is a churn between wrong picks and does not move winner_hit.

---

## 2024 R14 — Belgian Grand Prix

- Actual winner: **HAM** (grid 3, finish 1)
- Baseline predicted winner: **PIA**  | variant predicted winner: **NOR**
- Actual winner blend rank — baseline 3, variant 3

### Contender scores (baseline model -> variant model)

| driver | model_rank B->V | raw B->V | blend_rank B->V | blend_score B->V |
|---|---|---|---|---|
| PIA | 2 -> 3 | +0.706 -> +0.668 | 1 -> 4 | 3.350 -> 3.900 |
| NOR | 3 -> 2 | +0.482 -> +0.696 | 2 -> 1 | 3.450 -> 2.900 |
| HAM | 4 -> 4 | -0.459 -> -0.274 | 3 -> 3 | 3.550 -> 3.550 |

### PIA — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 4.296 | +0.0929 |
| ewma_pace_3 | -919.568 | +0.3190 |
| team_ewma_finish_3 | 4.688 | +0.2880 |
| team_ewma_pace_3 | -1007.594 | +0.2579 |
| pace_trend_5 | 100.050 | -0.0490 |
| finish_trend_5 | -1.100 | +0.0621 |
| team_pace_trend_5 | 128.700 | +0.0561 |

Sum of 7 new-feature contribs = +1.0271. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| pace_delta_5 | +0.5930 | +0.2761 | -0.3169 |
| team_form_finish_5 | +0.3005 | +0.0824 | -0.2181 |
| form_finish_3 | +0.1661 | -0.0026 | -0.1687 |

### NOR — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 5.077 | +0.0863 |
| ewma_pace_3 | -1094.947 | +0.3129 |
| team_ewma_finish_3 | 4.688 | +0.2156 |
| team_ewma_pace_3 | -1007.594 | +0.2639 |
| pace_trend_5 | 157.350 | -0.0365 |
| finish_trend_5 | 0.100 | +0.0276 |
| team_pace_trend_5 | 128.700 | +0.0640 |

Sum of 7 new-feature contribs = +0.9339. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| pace_delta_5 | +0.5754 | +0.2739 | -0.3015 |
| team_form_finish_5 | +0.2165 | +0.0768 | -0.1398 |
| quali_gap_to_pole_ms | +0.0069 | -0.1033 | -0.1102 |

### HAM — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 4.486 | +0.0900 |
| ewma_pace_3 | -886.087 | +0.1906 |
| team_ewma_finish_3 | 6.152 | +0.1414 |
| team_ewma_pace_3 | -877.046 | +0.2018 |
| pace_trend_5 | -37.650 | -0.0377 |
| finish_trend_5 | -0.400 | -0.0700 |
| team_pace_trend_5 | -59.200 | -0.0140 |

Sum of 7 new-feature contribs = +0.5021. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| team_form_finish_5 | +0.2712 | +0.0818 | -0.1895 |
| pace_delta_5 | +0.3049 | +0.1254 | -0.1795 |
| form_finish_3 | +0.1176 | -0.0206 | -0.1382 |

### Context feature values

| feature | PIA | NOR | HAM |
|---|---|---|---|
| ewma_finish_3 | 4.296 | 5.077 | 4.486 |
| form_finish_5 | 3.800 | 5.800 | 3.000 |
| form_finish_3 | 2.333 | 8.333 | 2.667 |
| ewma_pace_3 | -919.568 | -1094.947 | -886.087 |
| pace_delta_5 | -1017.250 | -1105.350 | -951.350 |
| finish_trend_5 | -1.100 | 0.100 | -0.400 |
| pace_trend_5 | 100.050 | 157.350 | -37.650 |
| team_ewma_finish_3 | 4.688 | 4.688 | 6.152 |
| team_form_finish_5 | 4.800 | 4.800 | 5.000 |
| quali_pos | 6.000 | 5.000 | 4.000 |
| grid_pos | 5.000 | 4.000 | 3.000 |
| season_rank | 5.000 | 2.000 | 6.000 |

### Recent race history before this race (last 5, from race_results)

**PIA** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2024 | 9 | Canadian Grand Prix | 4 | 5 | -1541 | False | True |
| 2024 | 10 | Spanish Grand Prix | 9 | 7 | -964 | False | True |
| 2024 | 11 | Austrian Grand Prix | 7 | 2 | -648 | False | True |
| 2024 | 12 | British Grand Prix | 5 | 4 | -820 | False | True |
| 2024 | 13 | Hungarian Grand Prix | 2 | 1 | -1113 | False | False |

**NOR** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2024 | 9 | Canadian Grand Prix | 3 | 2 | -1658 | False | True |
| 2024 | 10 | Spanish Grand Prix | 1 | 2 | -1135 | False | True |
| 2024 | 11 | Austrian Grand Prix | 2 | 20 | -852 | False | True |
| 2024 | 12 | British Grand Prix | 3 | 3 | -886 | False | True |
| 2024 | 13 | Hungarian Grand Prix | 1 | 2 | -996 | False | False |

**VERDICT — neutral (both wrong; immaterial).** HAM won from grid 3 only after Russell's post-race disqualification (underweight car) — an outcome no form/pace model can predict. Baseline PIA -> variant NOR is irrelevant churn.

---

## 2024 R20 — Mexico City Grand Prix

- Actual winner: **SAI** (grid 1, finish 1)
- Baseline predicted winner: **SAI**  | variant predicted winner: **NOR**
- Actual winner blend rank — baseline 1, variant 2

### Contender scores (baseline model -> variant model)

| driver | model_rank B->V | raw B->V | blend_rank B->V | blend_score B->V |
|---|---|---|---|---|
| SAI | 3 -> 3 | +0.839 -> +1.038 | 1 -> 2 | 2.100 -> 2.100 |
| NOR | 2 -> 1 | +1.309 -> +1.630 | 3 -> 1 | 2.450 -> 1.900 |

### SAI — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 6.585 | +0.0450 |
| ewma_pace_3 | -905.766 | +0.1800 |
| team_ewma_finish_3 | 5.167 | +0.2895 |
| team_ewma_pace_3 | -914.531 | +0.1638 |
| pace_trend_5 | 5.525 | -0.1038 |
| finish_trend_5 | -0.300 | -0.0383 |
| team_pace_trend_5 | 11.225 | -0.0117 |

Sum of 7 new-feature contribs = +0.5245. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| pace_delta_5 | +0.3506 | +0.1762 | -0.1744 |
| team_form_finish_5 | +0.2960 | +0.1713 | -0.1248 |
| quali_pos | +1.4913 | +1.3694 | -0.1219 |

### NOR — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 3.494 | +0.1035 |
| ewma_pace_3 | -1155.919 | +0.1761 |
| team_ewma_finish_3 | 3.443 | +0.3701 |
| team_ewma_pace_3 | -1079.367 | +0.1428 |
| pace_trend_5 | -0.075 | -0.1200 |
| finish_trend_5 | 0.400 | +0.0606 |
| team_pace_trend_5 | 35.075 | +0.0098 |

Sum of 7 new-feature contribs = +0.7428. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| team_form_finish_5 | +0.4921 | +0.2356 | -0.2565 |
| pace_delta_5 | +0.3787 | +0.1784 | -0.2004 |
| season_rank | +0.2572 | +0.1102 | -0.1470 |

### Context feature values

| feature | SAI | NOR |
|---|---|---|
| ewma_finish_3 | 6.585 | 3.494 |
| form_finish_5 | 7.200 | 2.600 |
| form_finish_3 | 9.000 | 3.000 |
| ewma_pace_3 | -905.766 | -1155.919 |
| pace_delta_5 | -1009.900 | -1177.600 |
| finish_trend_5 | -0.300 | 0.400 |
| pace_trend_5 | 5.525 | -0.075 |
| team_ewma_finish_3 | 5.167 | 3.443 |
| team_form_finish_5 | 4.800 | 2.800 |
| quali_pos | 1.000 | 3.000 |
| grid_pos | 1.000 | 3.000 |
| season_rank | 5.000 | 2.000 |

### Recent race history before this race (last 5, from race_results)

**SAI** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2024 | 15 | Dutch Grand Prix | 10 | 5 | -1158 | False | False |
| 2024 | 16 | Italian Grand Prix | 5 | 4 | -1037 | False | False |
| 2024 | 17 | Azerbaijan Grand Prix | 3 | DNF | -1066 | True | False |
| 2024 | 18 | Singapore Grand Prix | 10 | 7 | -280 | False | False |
| 2024 | 19 | United States Grand Prix | 3 | 2 | -1508 | False | False |

**NOR** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2024 | 15 | Dutch Grand Prix | 1 | 1 | -1408 | False | False |
| 2024 | 16 | Italian Grand Prix | 1 | 3 | -1372 | False | False |
| 2024 | 17 | Azerbaijan Grand Prix | 15 | 4 | -76 | False | False |
| 2024 | 18 | Singapore Grand Prix | 1 | 1 | -1872 | False | False |
| 2024 | 19 | United States Grand Prix | 1 | 4 | -1159 | False | False |

**VERDICT — LOST; baseline was right.** SAI took pole (grid 1) and won. The variant promoted NOR from model-rank 2 to 1 (SAI held rank 3); the 0.45*grid + 0.55*model blend then flipped to NOR (1.90 vs SAI 2.10). NOR's momentum was genuinely hot (last 5: 1,3,4,1,4; ewma_finish_3=3.49) and the new features handed him a bigger boost (sum of 7 = +0.74 vs SAI +0.52; team_ewma_finish_3 +0.37, ewma_finish_3 +0.10). But SAI's momentum was DEFLATED by noise: his ewma_finish_3=6.59 / form_finish_5=7.20 are dragged up by a DNF at R17 Azerbaijan (running P-competitive, pace -1066) and a strategy-hit P7 at Singapore — not representative of his true pace. Strip the DNF and SAI's recent form is front-running. He was on pole at Mexico and won; the EWMA over-reacted to one retirement and a hot NOR streak.

---

## 2024 R23 — Qatar Grand Prix

- Actual winner: **VER** (grid 2, finish 1)
- Baseline predicted winner: **VER**  | variant predicted winner: **RUS**
- Actual winner blend rank — baseline 1, variant 3

### Contender scores (baseline model -> variant model)

| driver | model_rank B->V | raw B->V | blend_rank B->V | blend_score B->V |
|---|---|---|---|---|
| VER | 1 -> 3 | +1.475 -> +1.281 | 1 -> 3 | 1.450 -> 2.550 |
| RUS | 2 -> 1 | +1.313 -> +1.496 | 2 -> 1 | 1.550 -> 1.000 |

### VER — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 3.701 | +0.0852 |
| ewma_pace_3 | -965.450 | +0.2283 |
| team_ewma_finish_3 | 7.267 | -0.1277 |
| team_ewma_pace_3 | -620.904 | +0.2720 |
| pace_trend_5 | 245.475 | +0.1158 |
| finish_trend_5 | 0.400 | +0.0797 |
| team_pace_trend_5 | 157.975 | -0.0485 |

Sum of 7 new-feature contribs = +0.6049. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| wet_delta | +0.3885 | +0.2580 | -0.1306 |
| pace_delta_5 | +0.3265 | +0.1985 | -0.1281 |
| team_season_points | +0.2969 | +0.1698 | -0.1271 |

### RUS — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 4.927 | +0.1166 |
| ewma_pace_3 | -709.694 | +0.2293 |
| team_ewma_finish_3 | 5.907 | +0.1780 |
| team_ewma_pace_3 | -743.306 | +0.3311 |
| pace_trend_5 | -31.675 | -0.0431 |
| finish_trend_5 | -0.800 | +0.0093 |
| team_pace_trend_5 | -68.325 | +0.0163 |

Sum of 7 new-feature contribs = +0.8375. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| team_season_points | +0.3744 | +0.1730 | -0.2014 |
| team_form_finish_5 | +0.2821 | +0.1267 | -0.1554 |
| form_finish_3 | +0.2787 | +0.1552 | -0.1234 |

### Context feature values

| feature | VER | RUS |
|---|---|---|
| ewma_finish_3 | 3.701 | 4.927 |
| form_finish_5 | 3.400 | 4.000 |
| form_finish_3 | 4.000 | 3.333 |
| ewma_pace_3 | -965.450 | -709.694 |
| pace_delta_5 | -1056.300 | -638.100 |
| finish_trend_5 | 0.400 | -0.800 |
| pace_trend_5 | 245.475 | -31.675 |
| team_ewma_finish_3 | 7.267 | 5.907 |
| team_form_finish_5 | 7.200 | 6.200 |
| quali_pos | 1.000 | 2.000 |
| grid_pos | 2.000 | 1.000 |
| season_rank | 1.000 | 6.000 |

### Recent race history before this race (last 5, from race_results)

**VER** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2024 | 18 | Singapore Grand Prix | 2 | 2 | -1604 | False | False |
| 2024 | 19 | United States Grand Prix | 2 | 3 | -1407 | False | False |
| 2024 | 20 | Mexico City Grand Prix | 2 | 6 | -917 | False | False |
| 2024 | 21 | São Paulo Grand Prix | 17 | 1 | -546 | False | True |
| 2024 | 22 | Las Vegas Grand Prix | 5 | 5 | -807 | False | False |

**RUS** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2024 | 18 | Singapore Grand Prix | 4 | 4 | -531 | False | False |
| 2024 | 19 | United States Grand Prix | 20 | 6 | -604 | False | False |
| 2024 | 20 | Mexico City Grand Prix | 5 | 5 | -833 | False | False |
| 2024 | 21 | São Paulo Grand Prix | 2 | 4 | -460 | False | True |
| 2024 | 22 | Las Vegas Grand Prix | 1 | 1 | -762 | False | False |

**VERDICT — LOST; baseline was right.** VER won (grid 2, quali pole). The variant demoted VER two model ranks (1->3) and promoted RUS (2->1); blend then favored RUS 1.00 vs VER 2.55. The trigger is RUS's recency: he WON the immediately-prior race (Las Vegas, grid 1 -> P1), and the half-life-3 EWMA weights that single result heavily (new-feature sum RUS +0.84 vs VER +0.60). But VER's underlying pace was far superior all along (pace_delta_5 -1056 vs RUS -638; ewma_pace_3 -965 vs -709) and his ewma_finish_3 was actually better (3.70 vs 4.93). VER's finish_trend looked flat partly because R21 Sao Paulo was a WET win from grid 17 — an outlier that scrambles the trend. The EWMA over-weighted one RUS win over VER's deeper, faster race pace. VER won.

---

## 2025 R12 — British Grand Prix

- Actual winner: **NOR** (grid 3, finish 1)
- Baseline predicted winner: **PIA**  | variant predicted winner: **NOR**
- Actual winner blend rank — baseline 3, variant 1

### Contender scores (baseline model -> variant model)

| driver | model_rank B->V | raw B->V | blend_rank B->V | blend_score B->V |
|---|---|---|---|---|
| PIA | 1 -> 2 | +1.588 -> +1.259 | 1 -> 2 | 1.450 -> 2.000 |
| NOR | 2 -> 1 | +1.263 -> +1.312 | 3 -> 1 | 2.450 -> 1.900 |

### PIA — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 2.691 | -0.0208 |
| ewma_pace_3 | -1361.775 | +0.2655 |
| team_ewma_finish_3 | 3.661 | +0.2178 |
| team_ewma_pace_3 | -1362.514 | +0.3517 |
| pace_trend_5 | 41.450 | -0.0548 |
| finish_trend_5 | -0.100 | -0.0503 |
| team_pace_trend_5 | 25.050 | -0.0045 |

Sum of 7 new-feature contribs = +0.7046. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| pace_delta_5 | +0.3675 | +0.0593 | -0.3082 |
| team_form_finish_5 | +0.2281 | +0.0871 | -0.1410 |
| season_rank | +0.4526 | +0.3387 | -0.1140 |

### NOR — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 4.631 | +0.2004 |
| ewma_pace_3 | -1363.249 | +0.2708 |
| team_ewma_finish_3 | 3.661 | +0.2618 |
| team_ewma_pace_3 | -1362.514 | +0.3265 |
| pace_trend_5 | 8.650 | -0.0437 |
| finish_trend_5 | 1.500 | -0.0535 |
| team_pace_trend_5 | 25.050 | -0.0018 |

Sum of 7 new-feature contribs = +0.9606. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| pace_delta_5 | +0.3889 | +0.0699 | -0.3190 |
| season_rank | +0.2670 | +0.1341 | -0.1329 |
| team_form_finish_5 | +0.2400 | +0.1272 | -0.1128 |

### Context feature values

| feature | PIA | NOR |
|---|---|---|
| ewma_finish_3 | 2.691 | 4.631 |
| form_finish_5 | 2.600 | 4.800 |
| form_finish_3 | 2.333 | 7.000 |
| ewma_pace_3 | -1361.775 | -1363.249 |
| pace_delta_5 | -1543.000 | -1530.800 |
| finish_trend_5 | -0.100 | 1.500 |
| pace_trend_5 | 41.450 | 8.650 |
| team_ewma_finish_3 | 3.661 | 3.661 |
| team_form_finish_5 | 3.700 | 3.700 |
| quali_pos | 2.000 | 3.000 |
| grid_pos | 2.000 | 3.000 |
| season_rank | 1.000 | 2.000 |

### Recent race history before this race (last 5, from race_results)

**PIA** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2025 | 7 | Emilia Romagna Grand Prix | 1 | 3 | -1077 | False | False |
| 2025 | 8 | Monaco Grand Prix | 3 | 3 | -2347 | False | False |
| 2025 | 9 | Spanish Grand Prix | 1 | 1 | -1485 | False | False |
| 2025 | 10 | Canadian Grand Prix | 3 | 4 | -1526 | False | True |
| 2025 | 11 | Austrian Grand Prix | 3 | 2 | -1280 | False | False |

**NOR** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2025 | 7 | Emilia Romagna Grand Prix | 4 | 2 | -986 | False | False |
| 2025 | 8 | Monaco Grand Prix | 1 | 1 | -2230 | False | False |
| 2025 | 9 | Spanish Grand Prix | 2 | 2 | -1481 | False | False |
| 2025 | 10 | Canadian Grand Prix | 7 | DNF | -1798 | True | True |
| 2025 | 11 | Austrian Grand Prix | 1 | 1 | -1158 | False | False |

**VERDICT — GAINED; the variant was right.** NOR won from grid 3. Baseline picked pole-adjacent PIA (grid 2, model-rank 1); the variant promoted NOR to model-rank 1 and the blend correctly put NOR first (1.90 vs PIA 2.00). The new features netted NOR a larger boost (sum of 7 = +0.96 vs PIA +0.70; ewma_finish_3 +0.20, team_ewma +0.26/+0.33). Note honestly: NOR's raw ewma_finish_3 value (4.63) is WORSE than PIA's (2.69) because of NOR's wet Canada DNF (R10) — yet the learned contribution still favored NOR, so the gain here comes from the aggregate of the 7 features and how they re-weighted shared pace features (pace_delta_5 collapsed for both), not from a clean 'NOR's form average was better' story. The variant caught that NOR (Monaco + Austria winner, both from pole) was the man in form at Silverstone. Correct.

---

## 2025 R14 — Hungarian Grand Prix

- Actual winner: **NOR** (grid 3, finish 1)
- Baseline predicted winner: **NOR**  | variant predicted winner: **PIA**
- Actual winner blend rank — baseline 1, variant 3

### Contender scores (baseline model -> variant model)

| driver | model_rank B->V | raw B->V | blend_rank B->V | blend_score B->V |
|---|---|---|---|---|
| NOR | 1 -> 2 | +1.846 -> +1.645 | 1 -> 3 | 1.900 -> 2.450 |
| PIA | 2 -> 1 | +1.668 -> +1.651 | 2 -> 1 | 2.000 -> 1.450 |

### NOR — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 3.494 | +0.0534 |
| ewma_pace_3 | -1555.641 | +0.2287 |
| team_ewma_finish_3 | 2.861 | +0.2947 |
| team_ewma_pace_3 | -1527.489 | +0.3852 |
| pace_trend_5 | -65.650 | -0.0656 |
| finish_trend_5 | -1.700 | +0.0424 |
| team_pace_trend_5 | -59.400 | +0.0072 |

Sum of 7 new-feature contribs = +0.9460. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| team_form_finish_5 | +0.4048 | +0.1263 | -0.2785 |
| pace_delta_5 | +0.1866 | -0.0687 | -0.2553 |
| quali_gap_to_pole_ms | +0.9717 | +0.8249 | -0.1467 |

### PIA — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 2.229 | -0.0019 |
| ewma_pace_3 | -1499.336 | +0.3409 |
| team_ewma_finish_3 | 2.861 | +0.2674 |
| team_ewma_pace_3 | -1527.489 | +0.4264 |
| pace_trend_5 | -53.150 | -0.0350 |
| finish_trend_5 | -0.200 | -0.0845 |
| team_pace_trend_5 | -59.400 | -0.0039 |

Sum of 7 new-feature contribs = +0.9093. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| pace_delta_5 | +0.2788 | -0.0128 | -0.2916 |
| team_form_finish_5 | +0.3585 | +0.1038 | -0.2546 |
| season_rank | +0.4031 | +0.2762 | -0.1269 |

### Context feature values

| feature | NOR | PIA |
|---|---|---|
| ewma_finish_3 | 3.494 | 2.229 |
| form_finish_5 | 4.800 | 2.000 |
| form_finish_3 | 1.333 | 1.667 |
| ewma_pace_3 | -1555.641 | -1499.336 |
| pace_delta_5 | -1657.200 | -1563.800 |
| finish_trend_5 | -1.700 | -0.200 |
| pace_trend_5 | -65.650 | -53.150 |
| team_ewma_finish_3 | 2.861 | 2.861 |
| team_form_finish_5 | 3.400 | 3.400 |
| quali_pos | 3.000 | 2.000 |
| grid_pos | 3.000 | 2.000 |
| season_rank | 2.000 | 1.000 |

### Recent race history before this race (last 5, from race_results)

**PIA** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2025 | 9 | Spanish Grand Prix | 1 | 1 | -1485 | False | False |
| 2025 | 10 | Canadian Grand Prix | 3 | 4 | -1526 | False | True |
| 2025 | 11 | Austrian Grand Prix | 3 | 2 | -1280 | False | False |
| 2025 | 12 | British Grand Prix | 2 | 2 | -2029 | False | True |
| 2025 | 13 | Belgian Grand Prix | 2 | 1 | -1499 | False | True |

**NOR** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2025 | 9 | Spanish Grand Prix | 2 | 2 | -1481 | False | False |
| 2025 | 10 | Canadian Grand Prix | 7 | DNF | -1798 | True | True |
| 2025 | 11 | Austrian Grand Prix | 1 | 1 | -1158 | False | False |
| 2025 | 12 | British Grand Prix | 3 | 1 | -2279 | False | True |
| 2025 | 13 | Belgian Grand Prix | 1 | 2 | -1569 | False | True |

**VERDICT — LOST; baseline was right.** NOR won from grid 3. Raw scores were essentially TIED (NOR +1.645 vs PIA +1.651 — 0.006 apart), so the pick is a coin-flip the EWMA tipped the wrong way: variant demoted NOR 1->2, promoted PIA 2->1, blend then PIA 1.45 vs NOR 2.45. Mechanism: PIA's ewma_finish_3=2.23 looks cleaner than NOR's 3.49, because NOR's recent average is inflated by his wet Canada DNF (R10: grid 7 -> DNF, collision). Strip that one wet incident and NOR's run is 2,1,1,2 — better than PIA. The EWMA over-reacted to a single wet-race retirement on a near-tie and lost the winner.

---

## 2025 R16 — Italian Grand Prix

- Actual winner: **VER** (grid 1, finish 1)
- Baseline predicted winner: **VER**  | variant predicted winner: **NOR**
- Actual winner blend rank — baseline 1, variant 2

### Contender scores (baseline model -> variant model)

| driver | model_rank B->V | raw B->V | blend_rank B->V | blend_score B->V |
|---|---|---|---|---|
| VER | 2 -> 3 | +1.078 -> +0.907 | 1 -> 2 | 1.550 -> 2.100 |
| NOR | 3 -> 1 | +0.858 -> +1.589 | 3 -> 1 | 2.550 -> 1.450 |

### VER — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 5.666 | +0.1799 |
| ewma_pace_3 | -961.116 | +0.2327 |
| team_ewma_finish_3 | 9.473 | -0.0905 |
| team_ewma_pace_3 | -298.316 | -0.0141 |
| pace_trend_5 | -68.075 | -0.0437 |
| finish_trend_5 | -2.800 | +0.0804 |
| team_pace_trend_5 | -204.275 | -0.0130 |

Sum of 7 new-feature contribs = +0.3317. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| team_track_finish_avg | +0.2391 | +0.0850 | -0.1541 |
| quali_pos | +1.6116 | +1.7473 | +0.1357 |
| form_finish_3 | +0.1569 | +0.0359 | -0.1210 |

### NOR — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 6.078 | +0.0558 |
| ewma_pace_3 | -1540.191 | +0.2985 |
| team_ewma_finish_3 | 4.008 | +0.2665 |
| team_ewma_pace_3 | -1505.919 | +0.3538 |
| pace_trend_5 | 18.025 | -0.0373 |
| finish_trend_5 | 3.400 | -0.0267 |
| team_pace_trend_5 | 27.325 | +0.0146 |

Sum of 7 new-feature contribs = +0.9252. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| pace_delta_5 | +0.3198 | +0.0774 | -0.2424 |
| team_form_finish_5 | +0.3362 | +0.1688 | -0.1673 |
| driver_overtakes_10 | -0.3185 | -0.1560 | +0.1625 |

### Context feature values

| feature | VER | NOR |
|---|---|---|
| ewma_finish_3 | 5.666 | 6.078 |
| form_finish_5 | 7.600 | 4.600 |
| form_finish_3 | 5.000 | 7.000 |
| ewma_pace_3 | -961.116 | -1540.191 |
| pace_delta_5 | -638.625 | -1615.700 |
| finish_trend_5 | -2.800 | 3.400 |
| pace_trend_5 | -68.075 | 18.025 |
| team_ewma_finish_3 | 9.473 | 4.008 |
| team_form_finish_5 | 10.800 | 3.100 |
| quali_pos | 1.000 | 2.000 |
| grid_pos | 1.000 | 2.000 |
| season_rank | 3.000 | 2.000 |

### Recent race history before this race (last 5, from race_results)

**VER** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2025 | 11 | Austrian Grand Prix | 7 | DNF | NaN | True | False |
| 2025 | 12 | British Grand Prix | 1 | 5 | 488 | False | True |
| 2025 | 13 | Belgian Grand Prix | 4 | 4 | -1216 | False | True |
| 2025 | 14 | Hungarian Grand Prix | 8 | 9 | -700 | False | False |
| 2025 | 15 | Dutch Grand Prix | 3 | 2 | -1127 | False | False |

**NOR** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2025 | 11 | Austrian Grand Prix | 1 | 1 | -1158 | False | False |
| 2025 | 12 | British Grand Prix | 3 | 1 | -2279 | False | True |
| 2025 | 13 | Belgian Grand Prix | 1 | 2 | -1569 | False | True |
| 2025 | 14 | Hungarian Grand Prix | 3 | 1 | -1728 | False | False |
| 2025 | 15 | Dutch Grand Prix | 2 | DNF | -1344 | True | False |

**VERDICT — LOST; baseline was right — but a DIFFERENT failure mode.** VER won from pole (grid 1) at Monza. The variant promoted NOR from model-rank 3 to 1 (a large raw jump +0.86 -> +1.59) and dropped VER 2->3; blend then NOR 1.45 vs VER 2.10. Here the EWMA's read was NOT noise: VER genuinely was off form (recent DNF,5,4,9,2; ewma_finish_3=5.67) while McLaren/NOR had the fastest car all season (NOR ewma_pace_3 -1540 vs VER -961; new-feature sum NOR +0.93 vs VER +0.33). The momentum features correctly identified NOR as the faster car in general — but Monza is a low-downforce, pole-decides track that suited VER, and he converted pole to a win. This flip is the cost of momentum overriding a track-specific pole advantage, not of reacting to a single outlier. Baseline (anchored harder to grid) got it right.

---

## 2025 R23 — Qatar Grand Prix

- Actual winner: **VER** (grid 3, finish 1)
- Baseline predicted winner: **PIA**  | variant predicted winner: **VER**
- Actual winner blend rank — baseline 2, variant 1

### Contender scores (baseline model -> variant model)

| driver | model_rank B->V | raw B->V | blend_rank B->V | blend_score B->V |
|---|---|---|---|---|
| PIA | 2 -> 3 | +1.731 -> +0.918 | 1 -> 2 | 1.550 -> 2.100 |
| VER | 1 -> 1 | +1.863 -> +2.097 | 2 -> 1 | 1.900 -> 1.900 |

### PIA — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 8.277 | -0.0557 |
| ewma_pace_3 | -860.690 | +0.1547 |
| team_ewma_finish_3 | 7.353 | -0.0284 |
| team_ewma_pace_3 | -790.659 | +0.0998 |
| pace_trend_5 | 156.050 | -0.0096 |
| finish_trend_5 | 3.200 | -0.1115 |
| team_pace_trend_5 | 220.900 | -0.0506 |

Sum of 7 new-feature contribs = -0.0013. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| pace_delta_5 | +0.2578 | +0.0507 | -0.2070 |
| quali_gap_to_pole_ms | +1.0702 | +0.8801 | -0.1902 |
| season_rank | +0.2565 | +0.1360 | -0.1205 |

### VER — variant pred_contrib for the 7 new features

| feature | value | contrib (variant) |
|---|---|---|
| ewma_finish_3 | 2.595 | +0.0527 |
| ewma_pace_3 | -832.336 | +0.2048 |
| team_ewma_finish_3 | 7.342 | -0.0269 |
| team_ewma_pace_3 | -310.495 | +0.1127 |
| pace_trend_5 | 184.250 | +0.2116 |
| finish_trend_5 | -0.000 | +0.0590 |
| team_pace_trend_5 | 143.150 | +0.0161 |

Sum of 7 new-feature contribs = +0.6300. Top 3 shared-feature contribution shifts (baseline->variant):

| feature | contrib B | contrib V | delta |
|---|---|---|---|
| pace_delta_5 | +0.3678 | +0.1180 | -0.2499 |
| quali_gap_to_pole_ms | +0.5582 | +0.4671 | -0.0912 |
| season_points | +0.3363 | +0.2549 | -0.0814 |

### Context feature values

| feature | PIA | VER |
|---|---|---|
| ewma_finish_3 | 8.277 | 2.595 |
| form_finish_5 | 7.800 | 2.000 |
| form_finish_3 | 10.000 | 2.333 |
| ewma_pace_3 | -860.690 | -832.336 |
| pace_delta_5 | -712.050 | -859.750 |
| finish_trend_5 | 3.200 | -0.000 |
| pace_trend_5 | 156.050 | 184.250 |
| team_ewma_finish_3 | 7.353 | 7.342 |
| team_form_finish_5 | 6.500 | 6.900 |
| quali_pos | 1.000 | 3.000 |
| grid_pos | 1.000 | 3.000 |
| season_rank | 3.000 | 2.000 |

### Recent race history before this race (last 5, from race_results)

**VER** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2025 | 18 | Singapore Grand Prix | 2 | 2 | -1861 | False | True |
| 2025 | 19 | United States Grand Prix | 1 | 1 | -720 | False | False |
| 2025 | 20 | Mexico City Grand Prix | 5 | 3 | 50 | False | False |
| 2025 | 21 | São Paulo Grand Prix | 19 | 3 | -934 | False | False |
| 2025 | 22 | Las Vegas Grand Prix | 2 | 1 | -833 | False | False |

**PIA** last 5 races:
| season | round | event | grid | finish | pace_delta_ms | dnf | wet |
|---|---|---|---|---|---|---|---|
| 2025 | 18 | Singapore Grand Prix | 3 | 4 | -1639 | False | True |
| 2025 | 19 | United States Grand Prix | 6 | 5 | -306 | False | False |
| 2025 | 20 | Mexico City Grand Prix | 7 | 5 | -296 | False | False |
| 2025 | 21 | São Paulo Grand Prix | 4 | 5 | -614 | False | False |
| 2025 | 22 | Las Vegas Grand Prix | 5 | 20 | -705 | False | False |

**VERDICT — GAINED; the variant was right (cleanest win).** VER won from grid 3. Baseline picked pole-sitter PIA (grid 1, model-rank 1); the variant collapsed PIA's model score (raw +1.73 -> +0.92, model-rank 2->3) and held VER at model-rank 1, so the blend flipped to VER (1.90 vs PIA 2.10). This is exactly what momentum is for: PIA was in a genuine late-season COLLAPSE (last 5: 4,5,5,5,P20; ewma_finish_3=8.28, form_finish_5=7.80, finish_trend +3.2 worsening) while VER was surging (2,1,3,3,1; ewma_finish_3=2.60; winner of US GP and Las Vegas). The baseline, leaning on grid + cumulative season standing, kept faith with the pole-sitter and was wrong. The EWMA caught the regime change. Correct.

---

## Synthesis — why the EWMA nets negative for winner-picking

**Scoreboard.** Across 71 walk-forward races the momentum variant changed the
predicted winner in 9 races: 2 GAINED (2025 British R12, 2025 Qatar R23), 4 LOST
(2024 Mexico R20, 2024 Qatar R23, 2025 Hungary R14, 2025 Italy R16), and 3
neutral churns between already-wrong picks (2023 US, 2024 Hungary, 2024 Belgium).
Net winner_hit 46/71 -> 44/71 (-2), which is the whole regression.

**The dominant failure pattern (3 of the 4 losses): a DNF or wet-race result
poisons the half-life-3 EWMA.** The EWMA weights the most recent 1-3 races
heavily, and F1's recent-history stream is full of non-representative events:
- Mexico 2024: SAI's ewma_finish_3 (6.59) is inflated by an Azerbaijan DNF; the
  model preferred hot-streak NOR and lost the actual pole-winner SAI.
- Hungary 2025: NOR's ewma_finish_3 (3.49) is inflated by a wet Canada DNF; on a
  near-tied raw score the EWMA tipped to PIA and lost NOR.
- Qatar 2024: RUS's single prior-race win (Las Vegas) was recency-weighted over
  VER's much stronger sustained pace; a wet Sao Paulo outlier flattened VER's
  trend. Lost VER.
In each, the recent races that moved the EWMA were DNFs, wet chaos, or a lone win
— noise, not signal about who is fastest THIS weekend.

**The fourth loss is a distinct, harder problem (Monza 2025):** the EWMA read
was *correct* (VER genuinely off-form, McLaren genuinely fastest) but form was
the wrong signal — pole at a power track decided the race. No amount of noise-
robustness fixes this one; it is momentum legitimately overriding a track/grid
advantage that happened to win.

**Why it still helps top-3 / MAE while hurting winner_hit.** Momentum genuinely
sharpens the *ordering* of the field (podium overlap, spearman, MAE) — the two
gains (Qatar 2025, British 2025) show it correctly demoting a slumping pole-
sitter (PIA's 4,5,5,5,P20 collapse) and elevating a surging driver. But winner-
picking is a knife-edge: the blend decides P1 by tiny margins (the losses flipped
on raw gaps of 0.006-0.2 and single blend-rank steps). Improving the average
ordering while adding variance at the very top trades a few exact-winner hits for
broadly better ranks. On a 71-race sample, -2 winners outweighs the diffuse top-3
gains under the winner_hit adoption gate.

**What a better feature would need (UNTESTED SPECULATION — flagged as such).**
The clean fix suggested by the evidence is a *noise-robust* momentum signal:
compute the finish/pace EWMA over *classified, dry* races only — i.e. exclude
DNFs and wet races (or the driver's own wet/incident results) from the EWMA
window so a single retirement can't swing ewma_finish_3 by 3-4 positions. Three
of the four losses (Mexico, Hungary, Qatar-24) trace directly to a DNF/wet result
in the window, so masking them should recover most of the losses while keeping the
two legitimate gains (both driven by *sustained* form, not single events). The
Monza loss would remain, so even a perfect noise filter likely nets only ~neutral-
to-slightly-positive on winner_hit — this needs a fresh walk-forward to confirm
and must not be assumed. A second, cheaper option: down-weight momentum in the
blend for pole-sitters specifically (protect a strong qualifier), which would
address Mexico and Monza but is a blunt instrument. Neither has been run; both are
hypotheses, not results.
