# Four sequential experiments — 2026-07-18 (all vs frozen baseline 48/71, 33 feats, w=0.55)

Run one-by-one to isolate effects. All four negative/neutral; production unchanged.

## 1. Grid penalty delta (`grid_pos − quali_pos`) — NEUTRAL, not adopted
Signal exists (24.1% of rows nonzero; 4/22 historical P3+ winners carried penalties)
but the model doesn't convert it: 48/71 tie at w=0.55, P3+ bucket unchanged 2/13,
pole guardrail 39→38, worse at higher w. Top3 +0.014 and MAE −0.010 don't clear
any gate. Comeback winners aren't blocked by missing who's-out-of-position info —
the grid anchor still vetoes acting on it.

## 2. Per-driver simulation sigma (scaled by prior finish-pos volatility) — REFUTED
Calibration worse on every metric: winner log-loss 1.407→1.566 (clip 0.7–1.4) /
1.653 (0.5–1.8); P(actual winner) 0.264→0.230/0.219; win Brier and podium
log-loss both worse; display ranking churns in 61–65/71 races. Cause: finish-pos
volatility is mostly DNF/incident-driven, which the sim already models via
per-driver dnf_probs — per-driver sigma double-counts that variance. Keep global sigma.

## 3. Track-type generalization (3 overtaking-character clusters, driver form at
similar tracks, leakage-free expanding means) — REFUTED
Coverage improves (90.6% non-NaN vs 64.2% for exact-track history) but winner_hit
drops: 46/71 (finish only), 44/71 (finish+gain); P3+ unchanged. Track character
terciles are too crude a similarity measure — dilutes rather than transfers signal.

## 4. Dual blend weights (winner pick at model-heavy w, display at 0.55) — REFUTED/OBSOLETE
The w-sweep on the CURRENT 33-feat model is flat: 48/71 for every w in [0.55, 1.0].
The historical "model-heavy finds 1–2 more winners" pattern (28/32-feat models)
disappeared when quali_surprise was adopted — it absorbed the signal model-only
weighting used to exploit. Tuning winner-w is harmful out-of-sample (LOO 44/71,
reverse split −1). No dual-weight gain exists anymore.

## Net conclusion
The 33-feature model at w=0.55 is at a local optimum for this dataset: post-hoc
weighting tricks are exhausted (w-curve flat), cheap derived features are
saturated, and simulation tweaks degrade calibration. Remaining paths that could
move 48/71: more seasons of training/backtest data (2018–2022 ingest), and
practice-session long-run pace as a genuinely new pre-race signal.
