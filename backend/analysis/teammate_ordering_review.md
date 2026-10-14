# Teammate-ordering research review (2026-07-19)

Companion to ranking_eval_review.md. Question: why is teammate pairwise accuracy
(73.7%) below overall (82.3%), and what — if anything — should be built about it.

## Part 1 — Why teammate ordering is "worse": the deficit is mostly a margin artifact

The comparison 73.7% vs 82.3% is confounded. Teammates share a car, so they
finish CLOSE together — teammate pairs are drawn disproportionately from the
adjacent-margin population (63.9%), not the overall population (which is 50%
distant pairs at 92.4%). Back-of-envelope: if teammate pairs are ~50% adjacent,
~30% mid, ~20% distant, the margin-implied expectation is
0.5(63.9) + 0.3(78.5) + 0.2(92.4) ≈ 74.0% — indistinguishable from the observed
73.7%. The similar-cars bucket (74.3%) sitting at the same level supports this:
closeness, not teammate-ness, is the driver.

THEREFORE: before any feature work, the pairwise framework must answer three
questions (all measurement, zero risk):
1. Margin-controlled comparison: teammate vs non-teammate accuracy AT THE SAME
   actual margin. The "real" teammate deficit is whatever survives this control.
2. Quali-order baseline: how often does pre-race qualifying order alone predict
   teammate race order? If quali alone ≈ 71-73%, the model adds almost nothing
   within teams and the information ceiling given current features is close.
3. Oracle ceiling: "always pick the season head-to-head leader" accuracy.
   Teammate h2h records run ~60/40 to 70/30 in equal machinery — the
   irreducible variance floor is HIGH; a realistic ceiling is likely ~78-80%,
   not 92%.

Cause attribution, conditional on a residual gap surviving the margin control:
- Missing features: PARTLY. The only within-team discriminator that survives
  car cancellation is the driver-level signal, and the model has exactly one
  relative feature (quali_gap_to_teammate). Within-team RACE pace is absent.
- Feature transformations: PARTLY. Trees are poor at computing cross-row
  differences; teammates share every car feature, so within-group splits rely
  on the few driver-level absolutes.
- Target formulation / LambdaRank behavior: YES for lower teams. With
  2^(25-pos) gains, a P12/P13 teammate swap moves NDCG negligibly — near-zero
  gradient. Teammate resolution at the back was never trained for. (Same
  mechanism as the 56% backmarker bucket.)
- Car effects: definitionally canceled — which is precisely why the bucket is
  hard: everything the model is best at is useless here.
- Variance: LARGE. Same-car ordering flips on tenths; strategy asymmetry
  (teams split strategies, undercut priority to the lead car) injects
  pre-race-unknowable swaps. This bounds any fix.
- Data: 540 eval pairs is powerful for MEASUREMENT (±4pp), but the number of
  training examples where teammates' order must be resolved is similarly
  modest.

Strategic note on value: teammate pairs are only ~5% of all pairs, so even
+5pp in-bucket is +0.26pp overall pairwise. BUT the subset that matters
disproportionately is front-running teammates — several remaining winner
misses (Hungary '25, Spa '25) were literally McLaren intra-team ordering
errors at P1/P2. The bucket should be split by team tier: top-team teammate
pairs are the winner-relevant sliver.

## Part 2/4 — Ideas, criticized and tiered

### Tier 1 (high confidence, measurable now)

T1.0 MARGIN-CONTROLLED TEAMMATE DIAGNOSTIC (measurement, do first).
The three analyses above + teammate bucket split by team tier and by margin.
Why: decides whether Tier-1 features are warranted at all; calibrates the
ceiling. Risks: none. Success: a number — the residual teammate deficit after
margin control. Improves: nothing directly; prevents wasted work. If the
residual deficit is ~0, STOP here: teammate ordering is already at its
margin-implied level and the research direction closes.

T1.1 ROLLING TEAMMATE RACE-PACE DELTA.
Mean difference in pace_delta_ms between current teammates over their last ~5
shared classified races (sign flipped per driver; NaN if <2 shared races).
- Why it helps: it is the race-trim analog of quali_gap_to_teammate — the one
  proven within-team feature. Quali gap answers "who is faster on Saturday";
  nothing answers "who is faster on Sunday", and race ordering is a Sunday question.
- Weakness attacked: within-team resolution where every car feature cancels.
- Orthogonality: HIGH. No current feature encodes within-team race pace.
  (form_finish differences leak car+luck; this is lap-time-based.)
- Risks: halved sample per driver; breaks on lineup changes (NaN is correct);
  mild collinearity with quali gap (fast drivers are fast both days) — the
  orthogonal component is exactly the one-lap-vs-race-trim skill split, which
  is a real, known driver trait.
- Measure: teammate-bucket accuracy via pairwise compare() (540 pairs, ±4pp
  CI); guardrails: overall pairwise, MAE, winner on the standard panel.
- Expected movement: teammate bucket (target ≥ +3-4pp to call it real);
  adjacent pairs slightly (teammate pairs are a chunk of them); overall
  pairwise +0.2-0.4pp; winner ~nil except via front-team teammate calls.

T1.2 ROLLING HEAD-TO-HEAD FINISH RECORD (vs current teammate, last ~8 shared
classified races).
- Why: the empirical-Bayes answer to "who is currently ahead in this car";
  captures team-order/status effects no pace number sees.
- Orthogonality: MEDIUM — overlaps T1.1 (dominance shows up in both); test
  both, expect to keep ONE (whichever wins the bucket; keeping both invites
  the correlated-feature dilution seen repeatedly in this project).
- Risks: slow-moving; confounded by strategy roles; ~binary-ish signal.
- Measure/movement: same rubric as T1.1.

### Tier 2 (prototype only after Tier 1, and only if T1.0 shows a real gap)

T2.1 EXPLICIT WITHIN-TEAM FEATURE DELTAS (form, consistency as teammate
differences — a transformation, not new signal). Rationale: trees can't
subtract across rows. Uncertain because the information exists implicitly;
cap at 2-3 deltas, expect dilution risk. Judge on teammate bucket only.

T2.2 LABEL-GAIN REFORMULATION (hybrid/linear gains instead of 2^(25-pos)).
Targets the gradient starvation behind BOTH backmarker (56%) and lower-order
teammate pairs. This is the only idea that attacks the objective itself.
Honest framing: it is a whole-order trade — top-weighted gains are plausibly
WHY winner/podium are strong. Run as one controlled experiment on the full
panel; accept only if midfield/backmarker/teammate buckets rise with winner
and podium flat inside their noise bands. Coin-flip prior; high information
value either way.

T2.3 TEAM STRATEGY-ASYMMETRY PRIOR (per-team historical rate of race order
inverting quali order between teammates). Thin, team-level, slow signal;
likely noise at this sample. Only if T1.0 attributes the residual gap to
specific teams.

### Tier 3 (not recommended, with reasons)

- Teammate PRACTICE deltas: the fuel-cancellation logic is sound, but this is
  the practice pipeline under a new name — five integrations already failed on
  coverage/noise; the teammate variant inherits the same 60-80% coverage holes.
- "Improvement vs teammate" trajectories: rookie-thin data; momentum-family
  reasoning already refuted twice.
- A dedicated pairwise teammate classifier: 540 evaluation pairs but only ~500
  training teammate-pairs → guaranteed overfit; duplicates the ranker.
- Post-hoc teammate swap rules: post-processing was refuted three separate
  times in this project (rank gating, margin gating, tie-breakers) — the
  tie-breaker experiment specifically showed post-hoc swapping only loses.
- More quali-gap variants (medians, trends): redundant with the existing
  feature; the ms-space variant already regressed once (quali_gap_surprise).
- External team-orders data: label noise, scope creep, unverifiable.

## T1.0 RESULTS (2026-07-19, teammate_diagnostic.py) — DIRECTION CLOSED

1. Margin control: teammate 73.7% vs margin-implied expectation 74.7% —
   residual deficit +1.0% ≈ noise. The "teammate weakness" was ENTIRELY a
   margin-composition artifact. Detail: teammates BEAT other pairs on adjacent
   margins (69.1% vs 63.2% — quali_gap_to_teammate earning its keep) and lose
   on 6+ margins (84.3% vs 92.6% — big intra-team gaps are incident/strategy
   surprises, unknowable pre-race).
2. Quali-order baseline: 73.8%. Model on same subset: 73.8% — IDENTICAL.
   Within teams the model adds zero resolution beyond Saturday's order; the
   other 32 features contribute nothing to teammate ordering.
3. Oracle ceilings: prior-h2h 67.6%, hindsight-season-h2h 72.1% — BOTH BELOW
   the quali baseline. "Who is the better driver" predicts Sunday order worse
   than "who qualified ahead this weekend." T1.2 is dead on arrival; T1.1's
   dominance-flavored signal is capped by the same evidence — downgraded to
   low expected value, not built.
4. Tier split: top-4-car teammate pairs 81.2% (the winner-relevant sliver is
   fine), midfield/back 68.0% (strategy-split chaos).

CONCLUSION: teammate ordering is at its information limit given qualifying.
The remaining 26% of wrong teammate pairs are adjacent coin-flips plus
large-margin incident surprises. Next diagnostic target per the review: the
~395 wrong DISTANT (6+) non-teammate pairs — 7.6% error on pairs where being
wrong is never variance.

## Priority order

1. T1.0 diagnostic (an afternoon; may close the whole direction — that is a
   GOOD outcome, it would redirect effort to the distant-pair 7.6% error mass).
2. T1.1 + T1.2 as one experiment family judged on the teammate bucket.
3. T2.2 label-gain experiment (independent of teammates; highest information
   value per experiment among Tier 2).
Everything judged on the pairwise panel with compare(); winner hit rate is a
guardrail, never the objective.
