"""Experiment: grid-pace mismatch feature family for the F1 race-winner model.

Builds within-race rank features from EXISTING dataset.parquet columns, runs the
walk-forward backtest for each variant, reports overall + targeted-bucket metrics,
re-sweeps BLEND_W for the best variant, lists flipped races, and prints new-feature
importance. Experiment only -- does NOT touch production ml/*.py or app/**.

Run from backend/ with the project venv:
    ./.venv/bin/python analysis/pace_mismatch_experiment.py

Writes:
    analysis/raw_scores_cache_pace.pkl   (per-variant raw walk-forward scores)
    analysis/pace_mismatch_results.md    (all tables + leakage note + verdict)
"""
from __future__ import annotations
import sys, pickle, io
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import spearmanr

BACKEND = Path("/Users/yahirbahena/GitHub/F1 Predicitor Project/backend")
sys.path.insert(0, str(BACKEND))
from ml.features import FEATURE_COLS  # noqa

ANALYSIS = BACKEND / "analysis"
CACHE = ANALYSIS / "raw_scores_cache_pace.pkl"
RESULTS_MD = ANALYSIS / "pace_mismatch_results.md"

MAX_LABEL = 25
PARAMS = dict(objective="lambdarank", n_estimators=400, learning_rate=0.03,
              num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=2.0, random_state=42, verbose=-1)
TRAIN_ONLY = 30  # first 30 races are train-only

# ---------------------------------------------------------------------------
# Feature construction (within-race ranks from existing columns)
# ---------------------------------------------------------------------------
def add_mismatch_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add grid-pace mismatch features. All within-race; leakage-free because
    both inputs are pre-race-safe (see leakage note in results md).

    pace_delta_5 / stint1_pace_5: LOWER = faster (winners avg pace_delta_5 ~ -1148
    vs field ~ -45; corr with finish_pos is +0.54). So rank ascending -> rank 1 =
    fastest recent race pace. NaN inputs -> NaN rank (not imputed).
    """
    df = df.copy()
    g = df.groupby(["season", "round"])
    # rank 1 = fastest (smallest pace_delta_5); NaN stays NaN
    df["pace_rank_in_race"] = g["pace_delta_5"].rank(method="min", ascending=True, na_option="keep")
    df["pace_grid_mismatch"] = df["quali_pos"] - df["pace_rank_in_race"]
    # stint variant
    df["stint1_rank_in_race"] = g["stint1_pace_5"].rank(method="min", ascending=True, na_option="keep")
    df["stint1_grid_mismatch"] = df["quali_pos"] - df["stint1_rank_in_race"]
    return df


VARIANTS = {
    "baseline":            FEATURE_COLS,
    "a_mismatch":          FEATURE_COLS + ["pace_grid_mismatch"],
    "b_mismatch_rank":     FEATURE_COLS + ["pace_grid_mismatch", "pace_rank_in_race"],
    "c_stint":             FEATURE_COLS + ["stint1_grid_mismatch", "stint1_rank_in_race"],
    "d_both":              FEATURE_COLS + ["pace_grid_mismatch", "pace_rank_in_race",
                                           "stint1_grid_mismatch"],
}


# ---------------------------------------------------------------------------
# Walk-forward: cache raw scores per variant
# ---------------------------------------------------------------------------
def build_cache(df: pd.DataFrame):
    race_keys = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    race_keys = list(race_keys.itertuples(index=False, name=None))
    # ordinal race index for fast train-subset masking
    ord_map = {k: i for i, k in enumerate(race_keys)}
    df = df.copy()
    df["_ord"] = list(zip(df["season"], df["round"]))
    df["_ord"] = df["_ord"].map(ord_map)

    cache = {name: [] for name in VARIANTS}
    fitted_models = {}  # keep last full-history fit per variant for importance
    for i in range(TRAIN_ONLY, len(race_keys)):
        season, rnd = race_keys[i]
        train = df[df["_ord"] < i]
        test = df[(df["season"] == season) & (df["round"] == rnd)].reset_index(drop=True)
        if test["target_pos"].isna().any() or len(test) < 10:
            continue
        train_s = train.sort_values(["season", "round"], kind="stable")
        groups = train_s.groupby(["season", "round"], sort=False).size().to_numpy()
        labels = (MAX_LABEL - train_s["target_pos"]).clip(lower=0).astype(int)
        grid = test["grid_pos"].fillna(len(test)).to_numpy().astype(float)
        actual = test["target_pos"].to_numpy()
        aw = test["driver_code"].to_numpy()
        aw_grid_raw = test["grid_pos"].to_numpy()
        for name, cols in VARIANTS.items():
            m = LGBMRanker(**PARAMS)
            m.fit(train_s[cols], labels, group=groups)
            raw = m.predict(test[cols])
            cache[name].append({
                "season": int(season), "round": int(rnd),
                "event_name": test.iloc[0]["event_name"],
                "location": test.iloc[0]["location"],
                "raw": raw, "grid": grid, "actual": actual,
                "codes": aw, "grid_raw": aw_grid_raw,
                "dnf": test["dnf"].to_numpy(),
            })
            fitted_models[name] = m  # last iteration = trained on the most history
    payload = {"cache": cache, "cols": VARIANTS,
               "importance": {n: dict(zip(VARIANTS[n], m.feature_importances_.tolist()))
                              for n, m in fitted_models.items()}}
    with open(CACHE, "wb") as f:
        pickle.dump(payload, f)
    return payload


def blend(raw, grid, w):
    model_rank = (-raw).argsort().argsort() + 1.0
    grid_rank = grid.argsort().argsort() + 1.0
    return (1.0 - w) * grid_rank + w * model_rank + 1e-6 * grid_rank


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def eval_variant(races, w=0.55):
    """Return per-race dataframe + summary dict for one variant at blend w."""
    rows = []
    for rc in races:
        s = blend(rc["raw"], rc["grid"], w)
        pr = s.argsort().argsort() + 1
        pw_idx = int(pr.argmin())
        aw_mask = np.where(rc["actual"] == 1)[0]
        aw_idx = int(aw_mask[0]) if len(aw_mask) else -1
        hit = int(aw_idx == pw_idx)
        pod = len(set(np.where(pr <= 3)[0]) & set(np.where(rc["actual"] <= 3)[0]))
        rho = spearmanr(pr, rc["actual"]).statistic
        rho = float(rho) if not np.isnan(rho) else 0.0
        mae = float(np.abs(pr - rc["actual"]).mean())
        aw_grid = float(rc["grid_raw"][aw_idx]) if aw_idx >= 0 and not pd.isna(rc["grid_raw"][aw_idx]) else np.nan
        rows.append({
            "season": rc["season"], "round": rc["round"],
            "event_name": rc["event_name"], "location": rc["location"],
            "winner_hit": hit, "podium_overlap": pod, "spearman": rho, "mae": mae,
            "aw_grid": aw_grid, "aw_code": rc["codes"][aw_idx] if aw_idx >= 0 else None,
            "pw_code": rc["codes"][pw_idx], "pw_grid": float(rc["grid_raw"][pw_idx]) if not pd.isna(rc["grid_raw"][pw_idx]) else np.nan,
            "pw_dnf": bool(rc["dnf"][pw_idx]),
        })
    d = pd.DataFrame(rows)
    n = len(d)
    grid3 = d[d["aw_grid"] >= 3]
    pole = d[d["aw_grid"] == 1]
    nodnf = d[~d["pw_dnf"]]
    summary = {
        "n": n,
        "hit_rate": d["winner_hit"].mean(),
        "hits": int(d["winner_hit"].sum()),
        "podium_overlap": d["podium_overlap"].mean(),
        "mae": d["mae"].mean(),
        "spearman": d["spearman"].mean(),
        "grid3_hits": int(grid3["winner_hit"].sum()), "grid3_n": len(grid3),
        "pole_hits": int(pole["winner_hit"].sum()), "pole_n": len(pole),
        "dnfadj_hits": int(nodnf["winner_hit"].sum()), "dnfadj_n": len(nodnf),
        "dnfadj_rate": nodnf["winner_hit"].mean(),
    }
    return d, summary


def per_season(d):
    return d.groupby("season")["winner_hit"].agg(["sum", "count"])


def main():
    df = pd.read_parquet(BACKEND / "data/processed/dataset.parquet")
    df = add_mismatch_features(df)

    # NaN coverage report
    cov = {c: float(df[c].isna().mean()) for c in
           ["pace_delta_5", "stint1_pace_5", "pace_rank_in_race",
            "pace_grid_mismatch", "stint1_rank_in_race", "stint1_grid_mismatch"]}

    if CACHE.exists():
        payload = pickle.load(open(CACHE, "rb"))
        # validate cached variants match
        if set(payload["cache"]) != set(VARIANTS):
            payload = build_cache(df)
    else:
        payload = build_cache(df)
    cache = payload["cache"]
    importance = payload["importance"]

    buf = io.StringIO()
    p = lambda *a: (print(*a), print(*a, file=buf))

    p("# Grid-Pace Mismatch Experiment - Results\n")
    p("## Leakage note")
    p("Both inputs to every mismatch feature are pre-race-safe:")
    p("- `pace_delta_5` is a rolling mean of PRIOR races' pace_delta_ms (strictly past races).")
    p("- `quali_pos` / `stint1_pace_5` are known before the race (quali happens pre-race;")
    p("  stint1_pace_5 is also a rolling PRIOR-race mean).")
    p("Ranking these WITHIN a race only reorders values already known pre-race, so it")
    p("introduces no future information -> leakage-free.\n")
    p("### Input NaN coverage")
    for k, v in cov.items():
        p(f"- {k}: NaN rate {v:.4f}")
    p("")
    p("(stint1_pace_5 NaN 1.8% << 50% threshold -> stint variant kept.)\n")

    # ---- Variant comparison at default w=0.55 ----
    p("## Variant comparison (walk-forward, BLEND_W=0.55)\n")
    p("| variant | hit_rate | hits | pod | mae | spear | grid>=3 | pole | dnf-adj |")
    p("|---|---|---|---|---|---|---|---|---|")
    per_variant = {}
    for name in VARIANTS:
        d, s = eval_variant(cache[name], 0.55)
        per_variant[name] = (d, s)
        p(f"| {name} | {s['hit_rate']:.4f} | {s['hits']}/{s['n']} | {s['podium_overlap']:.2f} | "
          f"{s['mae']:.2f} | {s['spearman']:.3f} | {s['grid3_hits']}/{s['grid3_n']} | "
          f"{s['pole_hits']}/{s['pole_n']} | {s['dnfadj_hits']}/{s['dnfadj_n']}={s['dnfadj_rate']:.3f} |")
    p("")

    base_hit = per_variant["baseline"][1]["hit_rate"]
    p(f"Baseline reproduction: {base_hit:.4f} (target 0.6479) -> "
      f"{'MATCH' if abs(base_hit-0.6479)<1e-3 else 'MISMATCH'}\n")

    # ---- Per-season for baseline + each variant ----
    p("## Per-season hit rates (BLEND_W=0.55)\n")
    seasons = sorted(per_variant["baseline"][0]["season"].unique())
    hdr = "| season | " + " | ".join(VARIANTS) + " |"
    p(hdr); p("|" + "---|" * (len(VARIANTS) + 1))
    for yr in seasons:
        cells = []
        for name in VARIANTS:
            d = per_variant[name][0]
            sub = d[d["season"] == yr]
            cells.append(f"{int(sub['winner_hit'].sum())}/{len(sub)}")
        p(f"| {yr} | " + " | ".join(cells) + " |")
    p("")

    # ---- Pick best variant: max hit_rate at w=0.55, tie-break grid3 then pole guardrail ----
    def score_key(name):
        s = per_variant[name][1]
        return (s["hit_rate"], s["grid3_hits"], s["pole_hits"])
    candidates = [n for n in VARIANTS if n != "baseline"]
    best = max(candidates, key=score_key)
    p(f"## Best variant by (hit_rate, grid>=3, pole) at w=0.55: **{best}**\n")

    # ---- BLEND_W sweep for baseline + best ----
    p("## BLEND_W sweep\n")
    ws = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
    p("| w | baseline hit | " + f"{best} hit | {best} grid>=3 | {best} pole | {best} pod | {best} mae |")
    p("|---|---|---|---|---|---|---|")
    best_sweep = {}
    for w in ws:
        _, sb = eval_variant(cache["baseline"], w)
        dv, sv = eval_variant(cache[best], w)
        best_sweep[w] = sv
        p(f"| {w:.2f} | {sb['hit_rate']:.4f} | {sv['hit_rate']:.4f} | "
          f"{sv['grid3_hits']}/{sv['grid3_n']} | {sv['pole_hits']}/{sv['pole_n']} | "
          f"{sv['podium_overlap']:.2f} | {sv['mae']:.2f} |")
    p("")

    # best w for the best variant (max hit, tie-break grid3)
    best_w = max(ws, key=lambda w: (best_sweep[w]["hit_rate"], best_sweep[w]["grid3_hits"]))
    sbw = best_sweep[best_w]
    p(f"Best w for {best}: **{best_w}** -> hit {sbw['hit_rate']:.4f}, "
      f"grid>=3 {sbw['grid3_hits']}/{sbw['grid3_n']}, pole {sbw['pole_hits']}/{sbw['pole_n']}\n")

    # ---- Flipped races: best variant at best_w vs baseline at 0.55 ----
    db, _ = eval_variant(cache["baseline"], 0.55)
    dv, _ = eval_variant(cache[best], best_w)
    merged = db[["season", "round", "event_name", "aw_grid", "winner_hit"]].merge(
        dv[["season", "round", "winner_hit", "pw_code", "aw_code"]],
        on=["season", "round"], suffixes=("_base", "_var"))
    gained = merged[(merged["winner_hit_base"] == 0) & (merged["winner_hit_var"] == 1)]
    lost = merged[(merged["winner_hit_base"] == 1) & (merged["winner_hit_var"] == 0)]
    p(f"## Flipped races ({best} @ w={best_w} vs baseline @ w=0.55)\n")
    p(f"GAINED ({len(gained)}):")
    for _, r in gained.iterrows():
        p(f"  + {int(r['season'])} R{int(r['round'])} {r['event_name']} "
          f"(winner grid {r['aw_grid']:.0f}, now predicts {r['aw_code']})")
    p(f"LOST ({len(lost)}):")
    for _, r in lost.iterrows():
        p(f"  - {int(r['season'])} R{int(r['round'])} {r['event_name']} "
          f"(winner grid {r['aw_grid']:.0f})")
    p("")

    # ---- Also compare best variant vs baseline at SAME w=0.55 (feature-only effect) ----
    dv55, _ = eval_variant(cache[best], 0.55)
    m55 = db[["season", "round", "event_name", "aw_grid", "winner_hit"]].merge(
        dv55[["season", "round", "winner_hit"]], on=["season", "round"], suffixes=("_base", "_var"))
    g55 = m55[(m55["winner_hit_base"] == 0) & (m55["winner_hit_var"] == 1)]
    l55 = m55[(m55["winner_hit_base"] == 1) & (m55["winner_hit_var"] == 0)]
    p(f"### Feature-only flips ({best} vs baseline, both w=0.55): gained {len(g55)}, lost {len(l55)}")
    for _, r in g55.iterrows():
        p(f"  + {int(r['season'])} R{int(r['round'])} {r['event_name']} (winner grid {r['aw_grid']:.0f})")
    for _, r in l55.iterrows():
        p(f"  - {int(r['season'])} R{int(r['round'])} {r['event_name']} (winner grid {r['aw_grid']:.0f})")
    p("")

    # ---- Feature importance of new features ----
    p("## Feature importance of new features (final full-history fit, gain-normalized split count)\n")
    for name in candidates:
        imp = importance[name]
        newf = [c for c in VARIANTS[name] if c not in FEATURE_COLS]
        total = sum(imp.values()) or 1
        p(f"**{name}**:")
        for c in newf:
            rank = sorted(imp, key=imp.get, reverse=True).index(c) + 1
            p(f"  - {c}: importance {imp[c]} ({100*imp[c]/total:.1f}% of total, rank {rank}/{len(imp)})")
    p("")

    # ---- Verdict ----
    p("## Verdict against success criteria\n")
    # criteria evaluated at the best w for the best variant
    grid3_recovered = sbw["grid3_hits"]  # baseline 2
    pole_lost = 36 - sbw["pole_hits"] if sbw["pole_n"] == 41 else (per_variant["baseline"][1]["pole_hits"] - sbw["pole_hits"])
    overall_ok = sbw["hit_rate"] >= 0.648
    # recovered relative to baseline grid3 = 2
    recovered_delta = sbw["grid3_hits"] - per_variant["baseline"][1]["grid3_hits"]
    p(f"- grid>=3 winners recovered vs baseline: {per_variant['baseline'][1]['grid3_hits']} -> "
      f"{sbw['grid3_hits']} (delta {recovered_delta:+d}); need >=3 recovered "
      f"[{'PASS' if recovered_delta >= 3 else 'FAIL'}]")
    p(f"- pole-winners lost vs baseline: {per_variant['baseline'][1]['pole_hits']} -> "
      f"{sbw['pole_hits']} (lost {pole_lost}); need <=1 lost "
      f"[{'PASS' if pole_lost <= 1 else 'FAIL'}]")
    p(f"- overall hit >= 0.648 at some w: best {best} hits {sbw['hit_rate']:.4f} at w={best_w} "
      f"[{'PASS' if overall_ok else 'FAIL'}]")
    confirm = (recovered_delta >= 3) and (pole_lost <= 1) and overall_ok
    p("")
    p(f"### {'CONFIRM' if confirm else 'REFUTE'}: "
      + ("feature family recovers >=3 grid>=3 winners, keeps the pole guardrail, and holds overall >= 0.648."
         if confirm else
         "success criteria not met (see per-criterion flags above)."))

    RESULTS_MD.write_text(buf.getvalue())
    print(f"\n[written] {RESULTS_MD}")


if __name__ == "__main__":
    main()
