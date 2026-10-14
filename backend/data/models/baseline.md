# F1 Race-Winner Model - BASELINE (frozen reference)

- blend weight (w): 0.55
- sigma (Monte Carlo noise): 4.5253
- eval races: 71
- model: /Users/yahirbahena/GitHub/F1 Predicitor Project/backend/data/models/model.joblib  (33 features)
- win-prob sim: 10000 draws, seed 42

## Overall

| metric | value |
|---|---|
| winner_hit_rate | 0.6761 (48/71) |
| top3_hit_rate | 0.9155 |
| pairwise_accuracy | 0.8191 |
| podium_overlap_avg | 2.0845 |
| position_mae | 3.2620 |
| spearman_avg | 0.6791 |
| dnf_adjusted_winner_hit_rate | 0.7059 (48/68) |
| avg_predicted_winner_prob | 0.3013 |
| avg_actual_winner_prob | 0.2639 |

## Breakdowns

### By season

| group | n | winner_hit | top3_hit | podium | mae | spearman |
|---|---|---|---|---|---|---|
| 2023 | 14 | 0.8571 (12/14) | 1.0000 | 1.9286 | 3.4286 | 0.6371 |
| 2024 | 24 | 0.6250 (15/24) | 0.8333 | 2.0833 | 2.8594 | 0.7525 |
| 2025 | 24 | 0.6250 (15/24) | 0.9583 | 2.2917 | 3.3285 | 0.6579 |
| 2026 | 9 | 0.6667 (6/9) | 0.8889 | 1.7778 | 3.8990 | 0.6055 |

### Wet vs dry

| group | n | winner_hit | top3_hit | podium | mae | spearman |
|---|---|---|---|---|---|---|
| wet | 16 | 0.5000 (8/16) | 0.8125 | 1.7500 | 3.3420 | 0.6714 |
| dry | 55 | 0.7273 (40/55) | 0.9455 | 2.1818 | 3.2387 | 0.6814 |

### Actual winner: pole vs not

| group | n | winner_hit | top3_hit | podium | mae | spearman |
|---|---|---|---|---|---|---|
| pole | 41 | 0.9512 (39/41) | 1.0000 | 2.1707 | 3.1601 | 0.6912 |
| not_pole | 30 | 0.3000 (9/30) | 0.8000 | 1.9667 | 3.4012 | 0.6627 |

### Actual winner: grid P3+ vs front-row

| group | n | winner_hit | top3_hit | podium | mae | spearman |
|---|---|---|---|---|---|---|
| grid>=3 | 13 | 0.1538 (2/13) | 0.6154 | 1.6923 | 3.2538 | 0.6774 |
| grid<=2 | 58 | 0.7931 (46/58) | 0.9828 | 2.1724 | 3.2638 | 0.6795 |
