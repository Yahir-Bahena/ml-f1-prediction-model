# Context-aware blend weighting — REFUTED (2026-07-18)

Tested: per-race blend weight w driven by pre-race context (track_overtakes_avg,
P2-gap-to-pole, track_sc_rate) instead of fixed w=0.55, as post-processing on the
cached 33-feature production walk-forward (baseline 48/71 reproduced from cache).

## Exploratory tercile evidence (in-sample)
- track_overtakes_avg: high-overtaking tracks gain +1 hit at w>=0.75; low/mid flat.
- P2 pole gap: tight-margin races gain +1 at w>=0.75, but mid tercile prefers
  w=0.45 (non-monotonic -> noise flag); dominant-pole races are w-insensitive.
- track_sc_rate: no signal (+/-1 noise).
Oracle per-tercile tuning caps at ~50/71 — the whole effect is 1-race-sized cells.

## Constrained rule (<=3 params): w = w_hi when track_ot >= t or pole_gap <= g, else 0.55
- In-sample best: track_ot >= 30, w_hi = 0.85 (fit dropped the pole-gap term) -> 50/71 (+2).
- Forward time-split (fit on races 1-35, frozen on 36-71): +1 OOS.
- Reverse time-split: **-1 OOS** (sign flips).
- Leave-one-race-out: **48/71 — exactly the fixed-w baseline, zero net gain**.
- Guardrails under the rule: podium 2.085 -> 2.000, MAE 3.262 -> 3.349 (both worse).

## Verdict
REFUTE per pre-registered criteria (needed LOO >= 49 and both split deltas >= 0).
The in-sample +2 is threshold-fitting noise; it does not survive any validation,
and the rule damages whole-order metrics even when winner-neutral. The shipped
weather gate (p_wet >= 0.5) plus fixed w=0.55 already captures the capturable
context signal; residual context effects on 71 races are +/-1-race noise.

Do not retry with: more thresholds, other track-history covariates, or flexible
learners — the sample cannot support them. A context-aware w only becomes worth
revisiting when the backtest grows by ~50+ races or a genuinely new pre-race
signal (e.g. practice-session long-run pace) is ingested.

Independently confirmed twice: an orchestrator-run rule fit (results above) and a
separate implementation (`context_blend_experiment.py`, results in
`context_blend.json`) reached the same verdict — LOO gain zero, time-splits
disagree in sign, plateau alone insufficient.

Reproduce: `./.venv/bin/python -m analysis.context_blend_experiment` (from backend/).
