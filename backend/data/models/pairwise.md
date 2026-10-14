# Pairwise ranking report

71 backtest races, blend w=0.55. Clean = neither driver DNF'd.
CIs are race-clustered bootstrap (95%).

| bucket | pairs | accuracy | 95% CI |
|---|---|---|---|
| all_clean | 10307 | 82.3% | 80.5%–84.0% |
| teammates | 540 | 73.7% | 69.9%–77.6% |
| similar_cars | 2097 | 74.3% | 71.6%–77.1% |
| adjacent_1_2 | 2251 | 63.9% | 62.2%–65.6% |
| mid_3_5 | 2858 | 78.5% | 76.7%–80.7% |
| distant_6plus | 5198 | 92.4% | 90.3%–94.5% |
| top4_internal | 426 | 80.0% | 75.7%–84.6% |
| midfield_internal | 3195 | 75.5% | 72.9%–78.2% |
| backmarker_internal | 402 | 56.0% | 49.2%–63.1% |
| cross_tier | 6284 | 87.7% | 85.4%–89.6% |
| comeback_involved | 2735 | 71.9% | 68.8%–74.7% |
| contaminated_dnf | 3514 | 61.8% | 58.0%–65.9% |

## Most-reversed recurring pairs (min 8 clean meetings)

| pair | meetings | reversal rate |
|---|---|---|
| BOR–HUL | 20 | 60% |
| BEA–TSU | 22 | 55% |
| BOT–ZHO | 30 | 53% |
| BEA–HUL | 25 | 48% |
| BEA–STR | 23 | 48% |
| RIC–STR | 20 | 45% |
| MAG–OCO | 25 | 44% |
| BEA–HAD | 23 | 43% |
| HAD–STR | 21 | 43% |
| HUL–RIC | 21 | 43% |
| GAS–HUL | 52 | 42% |
| BOT–RIC | 22 | 41% |
| RIC–SAR | 15 | 40% |
| COL–HUL | 25 | 40% |
| HUL–STR | 48 | 40% |
