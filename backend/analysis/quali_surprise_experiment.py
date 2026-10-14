"""QUALI-SURPRISE experiment for the F1 race-winner model.

Hypothesis: in the 5 "wrong pole-override" misses the pole-sitter WON but the
model ranked them ~3rd because their recent form said they shouldn't be on pole;
the blend then flipped to another front-row starter. A pole-sitter who
out-qualifies their own recent form has real weekend pace the form features
cannot see. This must come as a FEATURE (post-processing fixes are dead).

Two derived features, computed here on the loaded dataframe (no dataset rebuild):
  1. quali_surprise      = form_quali_5 - quali_pos
       positive = qualified better than recent 5-race form. NaN if either NaN.
  2. quali_gap_surprise  = prior_mean(quali_gap_to_pole_ms) - quali_gap_to_pole_ms
       prior mean = groupby(driver).shift(1).rolling(5, min_periods=2).mean();
       shift(1) drops the current race so the mean is STRICTLY-PRIOR (leakage
       free). positive = closer to pole than usual (in ms).

Variants: (a) quali_surprise alone; (b) both.

Scored with the PROJECT-STANDARD framework (ml.evaluate.run_walk_forward /
evaluate / to_dict / render_comparison). Nothing here mutates production.

Run from backend/ with the project venv:
    ./.venv/bin/python analysis/quali_surprise_experiment.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path("/Users/yahirbahena/GitHub/F1 Predicitor Project/backend")
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
from ml.features import FEATURE_COLS  # noqa: E402
from ml.evaluate import (  # noqa: E402
    BASELINE_JSON_PATH,
    run_walk_forward,
    evaluate,
    to_dict,
    render_comparison,
    _blend,
)

QS = "quali_surprise"
QGS = "quali_gap_surprise"
ANALYSIS = BACKEND / "analysis"

# The 5 "wrong pole-override" target races (season, round) -> winning pole-sitter.
TARGETS = [
    (2024, 15, "Dutch GP (Zandvoort)"),
    (2024, 22, "Las Vegas GP"),
    (2025, 3, "Japanese GP (Suzuka)"),
    (2025, 10, "Canadian GP (Montreal)"),
    (2025, 16, "Italian GP (Monza)"),
]
TARGET_KEYS = {(s, r) for s, r, _ in TARGETS}


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add quali_surprise + quali_gap_surprise. Leakage-free by construction:
    form_quali_5 already uses only strictly-prior quali; quali_pos and
    quali_gap_to_pole_ms are pre-race (qualifying precedes the race); the gap
    prior mean uses shift(1) so the current race is excluded."""
    df = df.sort_values(["season", "round"], kind="stable").reset_index(drop=True)
    df[QS] = df["form_quali_5"] - df["quali_pos"]
    prior_gap_mean = (
        df.groupby("driver_code")["quali_gap_to_pole_ms"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    )
    df[QGS] = prior_gap_mean - df["quali_gap_to_pole_ms"]
    return df


# ---------------------------------------------------------------------------
# Leakage check (independent from-scratch recompute using strictly-prior rows)
# ---------------------------------------------------------------------------
def leakage_check(df: pd.DataFrame, samples: list[tuple[int, int, str]]) -> tuple[str, bool]:
    lines = ["Leakage check (hand-recompute both features from STRICTLY-PRIOR rows only):"]
    all_ok = True
    for season, rnd, code in samples:
        d_all = df[df["driver_code"] == code].sort_values(["season", "round"])
        # strictly-prior races for this driver (before the current (season, round))
        cur_mask = (d_all["season"] == season) & (d_all["round"] == rnd)
        cur_pos = d_all.index[cur_mask][0]
        prior = d_all.loc[:cur_pos].iloc[:-1]  # everything before the current row
        cur = d_all[cur_mask].iloc[0]

        # quali_surprise: mean of last-5 prior quali_pos minus current quali_pos
        prior_q = prior["quali_pos"].dropna().tail(5)
        exp_qs = (float(prior_q.mean()) - cur["quali_pos"]) if len(prior_q) else np.nan

        # quali_gap_surprise: mean of last-5 prior gap (>=2 non-nan) minus current
        prior_gap = prior["quali_gap_to_pole_ms"].dropna().tail(5)
        exp_qgs = (
            float(prior_gap.mean()) - cur["quali_gap_to_pole_ms"]
            if len(prior_gap) >= 2
            else np.nan
        )

        got = df[(df["season"] == season) & (df["round"] == rnd) &
                 (df["driver_code"] == code)].iloc[0]
        lines.append(f"  {season} R{rnd} {code} ({len(prior)} prior driver races):")
        for name, e, g in ((QS, exp_qs, got[QS]), (QGS, exp_qgs, got[QGS])):
            g = float(g)
            ok = (np.isnan(e) and np.isnan(g)) or np.isclose(e, g, rtol=1e-9, atol=1e-9)
            all_ok &= ok
            es = "NaN" if np.isnan(e) else f"{e:.6f}"
            gs = "NaN" if np.isnan(g) else f"{g:.6f}"
            lines.append(f"    {name:18s} hand={es:>12s}  parquet={gs:>12s}  {'OK' if ok else 'MISMATCH'}")
    lines.append(f"  => {'ALL MATCH (no leakage)' if all_ok else 'MISMATCH FOUND'}")
    lines.append("  quali_pos / quali_gap_to_pole_ms are pre-race (qualifying precedes the")
    lines.append("  race) -- same information status as the existing quali_* FEATURE_COLS.")
    return "\n".join(lines), all_ok


# ---------------------------------------------------------------------------
# Per-race scorecard from walk-forward RaceResults (predicted winner + hit)
# ---------------------------------------------------------------------------
def per_race(races: list, blend_w: float) -> dict[tuple[int, int], dict]:
    out = {}
    for rc in races:
        scores = _blend(rc["raw"], rc["grid"], blend_w)  # lower = better
        pred_rank = scores.argsort().argsort() + 1
        pw_idx = int(pred_rank.argmin())              # predicted winner
        actual = rc["actual"]
        aw_idx = int(np.where(actual == 1)[0][0]) if (actual == 1).any() else -1
        codes = rc["driver_codes"]
        grid = rc["grid"]
        pole_idx = int(np.argmin(grid))               # grid==1
        out[(rc["season"], rc["round"])] = {
            "event": rc["event_name"],
            "pred_winner": codes[pw_idx],
            "actual_winner": codes[aw_idx] if aw_idx >= 0 else "?",
            "actual_winner_grid": float(grid[aw_idx]) if aw_idx >= 0 else float("nan"),
            "pole_driver": codes[pole_idx],
            "winner_hit": int(pw_idx == aw_idx),
        }
    return out


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset.parquet")
    df = add_derived(df)

    # ---- Mechanism check: target-race feature values (BEFORE model output) ----
    tgt_win = {(2024, 15): "NOR", (2024, 22): "RUS", (2025, 3): "VER",
               (2025, 10): "RUS", (2025, 16): "VER"}
    print("=" * 78)
    print("MECHANISM CHECK -- winning pole-sitter's feature values in the 5 targets")
    print("(hypothesis predicts distinctly POSITIVE quali_surprise / quali_gap_surprise)")
    print("=" * 78)
    print(f"{'race':26s} {'drv':4s} {'quali_pos':>9s} {'form_q5':>8s} "
          f"{'q_surprise':>11s} {'gap_surprise(ms)':>17s}")
    mech_ok = True
    for s, r, label in TARGETS:
        c = tgt_win[(s, r)]
        row = df[(df["season"] == s) & (df["round"] == r) & (df["driver_code"] == c)].iloc[0]
        qs, qgs = row[QS], row[QGS]
        mech_ok &= (qs > 0) and (qgs > 0)
        print(f"{label:26s} {c:4s} {row['quali_pos']:>9.1f} {row['form_quali_5']:>8.2f} "
              f"{qs:>+11.3f} {qgs:>+17.1f}")
    print(f"\n  => Data {'CONFIRMS' if mech_ok else 'DOES NOT confirm'} the mechanism "
          f"(all target quali_surprise AND gap_surprise strictly positive: {mech_ok}).")
    print()

    # ---- Leakage check (2 sampled rows, both features) ----
    samples = [(2024, 22, "RUS"), (2025, 16, "VER")]
    report, leak_ok = leakage_check(df, samples)
    print(report)
    print()

    baseline = json.loads(BASELINE_JSON_PATH.read_text())

    # ---- Walk-forward: baseline (for flips) + both variants ----
    # Base = the frozen 32-feature set. quali_surprise was ADOPTED into
    # production FEATURE_COLS after this experiment, so strip it (and the gap)
    # to keep the pre-adoption comparison reproducible.
    base = [c for c in FEATURE_COLS if c not in (QS, QGS)]
    variants = {
        "baseline_repro": base,
        "quali_surprise": base + [QS],
        "quali_surprise+gap": base + [QS, QGS],
    }
    wf = {}
    reps = {}
    for name, cols in variants.items():
        print(f"Running walk-forward: {name} ({len(cols)} features)...")
        wf[name] = run_walk_forward(df, cols)
        reps[name] = evaluate(wf[name])  # default blend_w = BLEND_W = 0.55

    # sanity: baseline repro must reproduce frozen baseline (46/71)
    br = reps["baseline_repro"].overall
    print(f"\n  baseline_repro @0.55: winner_hits={br['winner_hits']} "
          f"(frozen baseline: {baseline['overall']['winner_hits']})  "
          f"MAE={br['position_mae']:.4f} (frozen {baseline['overall']['position_mae']:.4f})")

    rep_a = to_dict(reps["quali_surprise"])
    rep_b = to_dict(reps["quali_surprise+gap"])
    (ANALYSIS / "quali_surprise.json").write_text(json.dumps(rep_a, indent=2))
    (ANALYSIS / "quali_surprise_gap.json").write_text(json.dumps(rep_b, indent=2))
    print(f"\n[written] {ANALYSIS/'quali_surprise.json'}")
    print(f"[written] {ANALYSIS/'quali_surprise_gap.json'}\n")

    # ---- Comparison table (both variants vs frozen baseline) ----
    print(render_comparison(baseline, [
        ("quali_surprise", rep_a),
        ("qs+gap", rep_b),
    ]))
    print()

    # ---- Blend sweep for each variant ----
    for name in ("quali_surprise", "quali_surprise+gap"):
        print(f"## Blend sweep -- {name}\n")
        print("| w | winner_hit_rate | winner_hits | podium | mae | pole-winner-hit |")
        print("|---|---|---|---|---|---|")
        for w in (0.55, 0.70, 0.85, 1.00):
            r = evaluate(wf[name], blend_w=w)
            o = r.overall
            pole = [x for x in r.breakdowns["winner_pole_vs_not"] if x["group"] == "pole"][0]
            print(f"| {w:.2f} | {o['winner_hit_rate']:.4f} | {o['winner_hits']} | "
                  f"{o['podium_overlap_avg']:.4f} | {o['position_mae']:.4f} | "
                  f"{pole['hits']}/{pole['n']} |")
        print()

    # ---- Target scorecard + guardrail + all flips (w=0.55) ----
    W = 0.55
    base_pr = per_race(wf["baseline_repro"], W)
    a_pr = per_race(wf["quali_surprise"], W)
    b_pr = per_race(wf["quali_surprise+gap"], W)

    print("## TARGET SCORECARD @ w=0.55 (the 5 wrong pole-override races)\n")
    print(f"{'race':26s} {'actual':6s} | {'baseline':14s} | {'quali_surprise':16s} | {'qs+gap':14s}")
    for s, r, label in TARGETS:
        k = (s, r)
        bp, ap, bbp = base_pr[k], a_pr[k], b_pr[k]
        aw = bp["actual_winner"]
        print(f"{label:26s} {aw:6s} | "
              f"{bp['pred_winner']:>4s} {'HIT ' if bp['winner_hit'] else 'miss':4s} "
              f"      | {ap['pred_winner']:>4s} {'HIT ' if ap['winner_hit'] else 'miss':4s} "
              f"        | {bbp['pred_winner']:>4s} {'HIT ' if bbp['winner_hit'] else 'miss':4s}")
    print()

    # GUARDRAIL: correct-override races (actual winner NOT on pole AND baseline hit)
    print("## GUARDRAIL -- correct-override races (winner NOT on pole, baseline HIT it)\n")
    print("Did any correct override get lost by a variant?\n")
    print(f"{'race':26s} {'winner(grid)':13s} | {'baseline':9s} {'qs':9s} {'qs+gap':9s}")
    n_co = a_lost = b_lost = 0
    for k in sorted(base_pr):
        bp = base_pr[k]
        if bp["actual_winner_grid"] != 1 and bp["winner_hit"] == 1:
            n_co += 1
            ap, bbp = a_pr[k], b_pr[k]
            a_lost += (ap["winner_hit"] == 0)
            b_lost += (bbp["winner_hit"] == 0)
            print(f"{bp['event'][:26]:26s} "
                  f"{bp['actual_winner']}(P{int(bp['actual_winner_grid'])}){'':6s} | "
                  f"{'HIT':9s} {'HIT' if ap['winner_hit'] else 'LOST':9s} "
                  f"{'HIT' if bbp['winner_hit'] else 'LOST':9s}")
    print(f"\n  correct-override races: {n_co} | lost by quali_surprise: {a_lost} | "
          f"lost by qs+gap: {b_lost}")

    # pole-winner hit guardrail
    def pole_hit(pr):
        rows = [v for v in pr.values() if v["actual_winner_grid"] == 1]
        return sum(v["winner_hit"] for v in rows), len(rows)
    print(f"\n  pole-winner hit  baseline {pole_hit(base_pr)[0]}/{pole_hit(base_pr)[1]} | "
          f"quali_surprise {pole_hit(a_pr)[0]}/{pole_hit(a_pr)[1]} | "
          f"qs+gap {pole_hit(b_pr)[0]}/{pole_hit(b_pr)[1]}  (frozen baseline 37/41)")
    print()

    # ALL flips
    for name, pr in (("quali_surprise", a_pr), ("qs+gap", b_pr)):
        print(f"## ALL flips vs baseline -- {name} (w=0.55)\n")
        gained, lost = [], []
        for k in sorted(base_pr):
            b0, v0 = base_pr[k]["winner_hit"], pr[k]["winner_hit"]
            if b0 == 0 and v0 == 1:
                gained.append(k)
            elif b0 == 1 and v0 == 0:
                lost.append(k)
        for tag, ks in (("GAINED", gained), ("LOST", lost)):
            print(f"  {tag} ({len(ks)}):")
            for k in ks:
                bp, vp = base_pr[k], pr[k]
                tflag = "  <-- TARGET" if k in TARGET_KEYS else ""
                print(f"    {bp['event'][:26]:26s} actual={bp['actual_winner']}"
                      f"(P{int(bp['actual_winner_grid'])})  "
                      f"base->{bp['pred_winner']}  {name}->{vp['pred_winner']}{tflag}")
        print(f"  net winner_hit delta: {len(gained) - len(lost):+d}\n")

    # ---- Adoption rule ----
    print("=" * 78)
    print("ADOPTION RULE @ w=0.55: winner_hit >= 47/71 AND podium >= 2.05 AND")
    print("  MAE <= 3.33 AND pole-winner hit >= 37/41")
    print("=" * 78)
    for name, rep in (("quali_surprise", reps["quali_surprise"]),
                      ("quali_surprise+gap", reps["quali_surprise+gap"])):
        o = rep.overall
        ph = pole_hit(a_pr if name == "quali_surprise" else b_pr)
        wh_hits = int(o["winner_hits"].split("/")[0])
        c1 = wh_hits >= 47
        c2 = o["podium_overlap_avg"] >= 2.05
        c3 = o["position_mae"] <= 3.33
        c4 = ph[0] >= 37
        print(f"\n  {name}:")
        print(f"    winner_hit    {o['winner_hits']:>6s}  >= 47/71 : {c1}")
        print(f"    podium        {o['podium_overlap_avg']:.4f}  >= 2.05  : {c2}")
        print(f"    MAE           {o['position_mae']:.4f}  <= 3.33  : {c3}")
        print(f"    pole-winner   {ph[0]}/{ph[1]}   >= 37/41 : {c4}")
        print(f"    leakage OK    {leak_ok}")
        print(f"    => {'ADOPT' if (c1 and c2 and c3 and c4 and leak_ok) else 'DO NOT ADOPT'}")


if __name__ == "__main__":
    main()
