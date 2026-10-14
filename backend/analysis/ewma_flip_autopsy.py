"""Per-race autopsy of the EWMA/momentum experiment's winner-flips.

Read-only on production. Reproduces the walk-forward for baseline (32 prod
FEATURE_COLS) and variant (baseline + 7 EWMA/trend features) using the project
framework (ml.evaluate.run_walk_forward / _fit / _blend), diffs the per-race
predicted winner, then for every flipped race:

  * refits the exact walk-forward model (baseline + variant) for that race,
  * pulls raw ranker scores, model ranks, blend scores, blend ranks for the two
    contenders (baseline pick vs variant pick),
  * computes LightGBM pred_contrib for both models and reports the 7 new
    features' contributions plus the biggest shared-feature contribution deltas,
  * dumps the moving driver's recent race history (finishes / pace / DNF / wet)
    from race_results.parquet to judge whether the EWMA reacted to signal or noise.

Run from backend/:
    ./.venv/bin/python analysis/ewma_flip_autopsy.py
Writes analysis/ewma_flip_autopsy.md (and prints the same).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path("/Users/yahirbahena/GitHub/F1 Predicitor Project/backend")
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
from ml.dataset import DATASET_PARQUET  # noqa: E402
from ml.features import FEATURE_COLS  # noqa: E402
from ml.train import PARAMS, BLEND_W  # noqa: E402
from ml.evaluate import _fit, _blend, _load_or_run, run_walk_forward  # noqa: E402

NEW_COLS = [
    "ewma_finish_3", "ewma_pace_3", "team_ewma_finish_3", "team_ewma_pace_3",
    "pace_trend_5", "finish_trend_5", "team_pace_trend_5",
]
VAR_COLS = FEATURE_COLS + NEW_COLS
OUT_MD = BACKEND / "analysis" / "ewma_flip_autopsy.md"

# Features whose value we surface for context around the EWMA story.
CONTEXT_VALS = [
    "ewma_finish_3", "form_finish_5", "form_finish_3",
    "ewma_pace_3", "pace_delta_5",
    "finish_trend_5", "pace_trend_5",
    "team_ewma_finish_3", "team_form_finish_5",
    "quali_pos", "grid_pos", "season_rank",
]

# Authored, evidence-based verdicts keyed by (season, round). Every factual claim
# is drawn from the pred_contrib / feature-value / history tables emitted above.
VERDICTS: dict[tuple[int, int], str] = {
    (2023, 18): (
        "**VERDICT — neutral (both wrong; immaterial to winner_hit).** VER won from "
        "grid 6; both models rank him 3rd and neither picks him. The variant only "
        "reshuffled two wrong front-runners (NOR->LEC) via small EWMA-driven raw "
        "shifts. Not a real EWMA success or failure — a low-magnitude churn among "
        "already-wrong picks."
    ),
    (2024, 13): (
        "**VERDICT — neutral (both wrong; immaterial).** PIA won from grid 2; both "
        "models rank him 2nd. Baseline VER -> variant NOR is a churn between wrong "
        "picks and does not move winner_hit."
    ),
    (2024, 14): (
        "**VERDICT — neutral (both wrong; immaterial).** HAM won from grid 3 only "
        "after Russell's post-race disqualification (underweight car) — an outcome no "
        "form/pace model can predict. Baseline PIA -> variant NOR is irrelevant churn."
    ),
    (2024, 20): (
        "**VERDICT — LOST; baseline was right.** SAI took pole (grid 1) and won. The "
        "variant promoted NOR from model-rank 2 to 1 (SAI held rank 3); the 0.45*grid "
        "+ 0.55*model blend then flipped to NOR (1.90 vs SAI 2.10). NOR's momentum was "
        "genuinely hot (last 5: 1,3,4,1,4; ewma_finish_3=3.49) and the new features "
        "handed him a bigger boost (sum of 7 = +0.74 vs SAI +0.52; team_ewma_finish_3 "
        "+0.37, ewma_finish_3 +0.10). But SAI's momentum was DEFLATED by noise: his "
        "ewma_finish_3=6.59 / form_finish_5=7.20 are dragged up by a DNF at R17 "
        "Azerbaijan (running P-competitive, pace -1066) and a strategy-hit P7 at "
        "Singapore — not representative of his true pace. Strip the DNF and SAI's "
        "recent form is front-running. He was on pole at Mexico and won; the EWMA over-"
        "reacted to one retirement and a hot NOR streak."
    ),
    (2024, 23): (
        "**VERDICT — LOST; baseline was right.** VER won (grid 2, quali pole). The "
        "variant demoted VER two model ranks (1->3) and promoted RUS (2->1); blend "
        "then favored RUS 1.00 vs VER 2.55. The trigger is RUS's recency: he WON the "
        "immediately-prior race (Las Vegas, grid 1 -> P1), and the half-life-3 EWMA "
        "weights that single result heavily (new-feature sum RUS +0.84 vs VER +0.60). "
        "But VER's underlying pace was far superior all along (pace_delta_5 -1056 vs "
        "RUS -638; ewma_pace_3 -965 vs -709) and his ewma_finish_3 was actually better "
        "(3.70 vs 4.93). VER's finish_trend looked flat partly because R21 Sao Paulo "
        "was a WET win from grid 17 — an outlier that scrambles the trend. The EWMA "
        "over-weighted one RUS win over VER's deeper, faster race pace. VER won."
    ),
    (2025, 12): (
        "**VERDICT — GAINED; the variant was right.** NOR won from grid 3. Baseline "
        "picked pole-adjacent PIA (grid 2, model-rank 1); the variant promoted NOR to "
        "model-rank 1 and the blend correctly put NOR first (1.90 vs PIA 2.00). The "
        "new features netted NOR a larger boost (sum of 7 = +0.96 vs PIA +0.70; "
        "ewma_finish_3 +0.20, team_ewma +0.26/+0.33). Note honestly: NOR's raw "
        "ewma_finish_3 value (4.63) is WORSE than PIA's (2.69) because of NOR's wet "
        "Canada DNF (R10) — yet the learned contribution still favored NOR, so the "
        "gain here comes from the aggregate of the 7 features and how they re-weighted "
        "shared pace features (pace_delta_5 collapsed for both), not from a clean "
        "'NOR's form average was better' story. The variant caught that NOR (Monaco + "
        "Austria winner, both from pole) was the man in form at Silverstone. Correct."
    ),
    (2025, 14): (
        "**VERDICT — LOST; baseline was right.** NOR won from grid 3. Raw scores were "
        "essentially TIED (NOR +1.645 vs PIA +1.651 — 0.006 apart), so the pick is a "
        "coin-flip the EWMA tipped the wrong way: variant demoted NOR 1->2, promoted "
        "PIA 2->1, blend then PIA 1.45 vs NOR 2.45. Mechanism: PIA's ewma_finish_3="
        "2.23 looks cleaner than NOR's 3.49, because NOR's recent average is inflated "
        "by his wet Canada DNF (R10: grid 7 -> DNF, collision). Strip that one wet "
        "incident and NOR's run is 2,1,1,2 — better than PIA. The EWMA over-reacted to "
        "a single wet-race retirement on a near-tie and lost the winner."
    ),
    (2025, 16): (
        "**VERDICT — LOST; baseline was right — but a DIFFERENT failure mode.** VER "
        "won from pole (grid 1) at Monza. The variant promoted NOR from model-rank 3 "
        "to 1 (a large raw jump +0.86 -> +1.59) and dropped VER 2->3; blend then NOR "
        "1.45 vs VER 2.10. Here the EWMA's read was NOT noise: VER genuinely was off "
        "form (recent DNF,5,4,9,2; ewma_finish_3=5.67) while McLaren/NOR had the "
        "fastest car all season (NOR ewma_pace_3 -1540 vs VER -961; new-feature sum "
        "NOR +0.93 vs VER +0.33). The momentum features correctly identified NOR as "
        "the faster car in general — but Monza is a low-downforce, pole-decides track "
        "that suited VER, and he converted pole to a win. This flip is the cost of "
        "momentum overriding a track-specific pole advantage, not of reacting to a "
        "single outlier. Baseline (anchored harder to grid) got it right."
    ),
    (2025, 23): (
        "**VERDICT — GAINED; the variant was right (cleanest win).** VER won from grid "
        "3. Baseline picked pole-sitter PIA (grid 1, model-rank 1); the variant "
        "collapsed PIA's model score (raw +1.73 -> +0.92, model-rank 2->3) and held "
        "VER at model-rank 1, so the blend flipped to VER (1.90 vs PIA 2.10). This is "
        "exactly what momentum is for: PIA was in a genuine late-season COLLAPSE "
        "(last 5: 4,5,5,5,P20; ewma_finish_3=8.28, form_finish_5=7.80, finish_trend "
        "+3.2 worsening) while VER was surging (2,1,3,3,1; ewma_finish_3=2.60; winner "
        "of US GP and Las Vegas). The baseline, leaning on grid + cumulative season "
        "standing, kept faith with the pole-sitter and was wrong. The EWMA caught the "
        "regime change. Correct."
    ),
}

SYNTHESIS = """\
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
"""

lines: list[str] = []
def emit(s: str = "") -> None:
    print(s)
    lines.append(s)


def blend_rank(raw: np.ndarray, grid: np.ndarray, w: float = BLEND_W) -> np.ndarray:
    """1 = predicted winner (argmin of blend score)."""
    sc = _blend(raw, grid, w)
    return sc.argsort().argsort() + 1, sc


def main() -> None:
    df = pd.read_parquet(DATASET_PARQUET)
    results = pd.read_parquet(config.RESULTS_PARQUET)

    # --- ordinal race index (mirror run_walk_forward) so we can rebuild the
    #     exact train subset for any flipped race ---
    race_keys = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    race_keys = list(race_keys.itertuples(index=False, name=None))
    ord_map = {k: i for i, k in enumerate(race_keys)}

    # --- walk-forwards ---
    emit("# EWMA / momentum experiment — per-race winner-flip autopsy\n")
    emit(f"- baseline: {len(FEATURE_COLS)} production FEATURE_COLS")
    emit(f"- variant : baseline + {len(NEW_COLS)} EWMA/trend features "
         f"({', '.join(NEW_COLS)})")
    emit(f"- blend w = {BLEND_W} (production); predicted winner = argmin blend score\n")

    base_races, base_hit = _load_or_run(df, FEATURE_COLS, PARAMS, use_cache=True)
    emit(f"Baseline walk-forward: {len(base_races)} races ({'cache' if base_hit else 'refit'}).")
    var_races = run_walk_forward(df, VAR_COLS, params=PARAMS)
    emit(f"Variant  walk-forward: {len(var_races)} races (fresh refit).\n")

    base_by = {(r["season"], r["round"]): r for r in base_races}
    var_by = {(r["season"], r["round"]): r for r in var_races}

    # --- diff predicted winners ---
    flips = []
    b_hits = v_hits = 0
    for key in sorted(base_by):
        rb, rv = base_by[key], var_by[key]
        codes = rb["driver_codes"]
        aw_idx = int(np.where(rb["actual"] == 1)[0][0])
        b_rank, b_sc = blend_rank(rb["raw"], rb["grid"])
        v_rank, v_sc = blend_rank(rv["raw"], rv["grid"])
        b_pick = codes[int(b_rank.argmin())]
        v_pick = codes[int(v_rank.argmin())]
        b_hits += int(b_rank[aw_idx] == 1)
        v_hits += int(v_rank[aw_idx] == 1)
        if b_pick != v_pick:
            flips.append((key, rb, rv, aw_idx, b_pick, v_pick))

    emit(f"Overall winner_hit — baseline {b_hits}/{len(base_races)}, "
         f"variant {v_hits}/{len(var_races)}.\n")
    emit(f"## Definitive flip list — {len(flips)} races where the predicted winner changed\n")
    emit("| season | round | event | actual winner | baseline pick | variant pick | outcome |")
    emit("|---|---|---|---|---|---|---|")
    for key, rb, rv, aw_idx, b_pick, v_pick in flips:
        codes = rb["driver_codes"]
        aw = codes[aw_idx]
        b_ok = (b_pick == aw)
        v_ok = (v_pick == aw)
        if b_ok and not v_ok:
            outcome = "LOST (baseline right)"
        elif v_ok and not b_ok:
            outcome = "GAINED (variant right)"
        else:
            outcome = "neutral (both wrong)"
        emit(f"| {key[0]} | {key[1]} | {rb['event_name']} | {aw} | {b_pick} | {v_pick} | {outcome} |")
    emit("")

    # --- per-race deep dive ---
    for key, rb, rv, aw_idx, b_pick, v_pick in flips:
        season, rnd = key
        i = ord_map[key]
        codes = rb["driver_codes"]
        aw = codes[aw_idx]

        # refit the exact walk-forward models for this race
        train = df[df.apply(lambda r: ord_map[(r["season"], r["round"])] < i, axis=1)]
        test = df[(df["season"] == season) & (df["round"] == rnd)].copy()
        base_model = _fit(train, FEATURE_COLS, PARAMS)
        var_model = _fit(train, VAR_COLS, PARAMS)

        Xb = test[FEATURE_COLS]
        Xv = test[VAR_COLS]
        raw_b = base_model.predict(Xb)
        raw_v = var_model.predict(Xv)
        grid = test["grid_pos"].fillna(len(test)).to_numpy().astype(float)

        b_rank, b_sc = blend_rank(raw_b, grid)
        v_rank, v_sc = blend_rank(raw_v, grid)
        b_mrank = (-raw_b).argsort().argsort() + 1
        v_mrank = (-raw_v).argsort().argsort() + 1

        # pred_contrib (last col = bias)
        cb = base_model.predict(Xb, pred_contrib=True)
        cv = var_model.predict(Xv, pred_contrib=True)

        idx_map = {c: k for k, c in enumerate(codes)}
        contenders = [b_pick, v_pick]
        if aw not in contenders:
            contenders.append(aw)

        emit(f"---\n\n## {season} R{rnd} — {rb['event_name']}\n")
        # header facts
        def finfo(code):
            ri = idx_map[code]
            row = test.iloc[ri]
            fin = int(row["target_pos"])
            return int(row["grid_pos"]) if not pd.isna(row["grid_pos"]) else -1, fin, bool(row["dnf"])
        aw_grid, aw_fin, aw_dnf = finfo(aw)
        emit(f"- Actual winner: **{aw}** (grid {aw_grid}, finish 1)")
        emit(f"- Baseline predicted winner: **{b_pick}**  | variant predicted winner: **{v_pick}**")
        emit(f"- Actual winner blend rank — baseline {int(b_rank[aw_idx])}, variant {int(v_rank[aw_idx])}\n")

        # contender score table
        emit("### Contender scores (baseline model -> variant model)\n")
        emit("| driver | model_rank B->V | raw B->V | blend_rank B->V | blend_score B->V |")
        emit("|---|---|---|---|---|")
        for code in contenders:
            ri = idx_map[code]
            emit(f"| {code} | {int(b_mrank[ri])} -> {int(v_mrank[ri])} | "
                 f"{raw_b[ri]:+.3f} -> {raw_v[ri]:+.3f} | "
                 f"{int(b_rank[ri])} -> {int(v_rank[ri])} | "
                 f"{b_sc[ri]:.3f} -> {v_sc[ri]:.3f} |")
        emit("")

        # new-feature contributions + top shared deltas per contender
        for code in contenders:
            ri = idx_map[code]
            emit(f"### {code} — variant pred_contrib for the 7 new features\n")
            emit("| feature | value | contrib (variant) |")
            emit("|---|---|---|")
            for f in NEW_COLS:
                fi = VAR_COLS.index(f)
                val = test.iloc[ri][f]
                vs = "NaN" if pd.isna(val) else f"{val:.3f}"
                emit(f"| {f} | {vs} | {cv[ri, fi]:+.4f} |")
            emit("")
            # top shared-feature contribution deltas baseline->variant
            deltas = []
            for f in FEATURE_COLS:
                bi = FEATURE_COLS.index(f)
                vi = VAR_COLS.index(f)
                d = cv[ri, vi] - cb[ri, bi]
                deltas.append((abs(d), d, f, cb[ri, bi], cv[ri, vi]))
            deltas.sort(reverse=True)
            new_total = sum(cv[ri, VAR_COLS.index(f)] for f in NEW_COLS)
            emit(f"Sum of 7 new-feature contribs = {new_total:+.4f}. "
                 f"Top 3 shared-feature contribution shifts (baseline->variant):\n")
            emit("| feature | contrib B | contrib V | delta |")
            emit("|---|---|---|---|")
            for _, d, f, b0, v0 in deltas[:3]:
                emit(f"| {f} | {b0:+.4f} | {v0:+.4f} | {d:+.4f} |")
            emit("")

        # context feature values for the contenders
        emit("### Context feature values\n")
        emit("| feature | " + " | ".join(contenders) + " |")
        emit("|---|" + "---|" * len(contenders))
        for f in CONTEXT_VALS:
            cells = []
            for code in contenders:
                ri = idx_map[code]
                if f == "grid_pos":
                    val = test.iloc[ri]["grid_pos"]
                else:
                    val = test.iloc[ri][f]
                cells.append("NaN" if pd.isna(val) else f"{val:.3f}")
            emit(f"| {f} | " + " | ".join(cells) + " |")
        emit("")

        # recent history for the two picks (the moving drivers) from race_results
        emit("### Recent race history before this race (last 5, from race_results)\n")
        for code in set([b_pick, v_pick]):
            h = results[(results["driver_code"] == code)]
            # strictly prior weekends
            h = h[h.apply(lambda r: (r["season"], r["round"]) in ord_map
                          and ord_map[(r["season"], r["round"])] < i, axis=1)]
            h = h.sort_values(["season", "round"]).tail(5)
            emit(f"**{code}** last {len(h)} races:")
            emit("| season | round | event | grid | finish | pace_delta_ms | dnf | wet |")
            emit("|---|---|---|---|---|---|---|---|")
            for _, r in h.iterrows():
                fin = "DNF" if r["dnf"] else (int(r["finish_pos"]) if not pd.isna(r["finish_pos"]) else "-")
                pace = "NaN" if pd.isna(r["pace_delta_ms"]) else f"{r['pace_delta_ms']:.0f}"
                grd = int(r["grid_pos"]) if not pd.isna(r["grid_pos"]) else "-"
                emit(f"| {int(r['season'])} | {int(r['round'])} | {r['event_name']} | {grd} | "
                     f"{fin} | {pace} | {bool(r['dnf'])} | {bool(r['wet_race'])} |")
            emit("")

        if key in VERDICTS:
            emit(VERDICTS[key])
            emit("")

    emit("---\n")
    emit(SYNTHESIS)

    OUT_MD.write_text("\n".join(lines))
    print(f"\n[written] {OUT_MD}")


if __name__ == "__main__":
    main()
