"""Reproduce the walk-forward backtest and emit a per-race table for failure analysis.

Adapts backend/ml/train.py. Run with the project venv from backend/:
    ./.venv/bin/python <this>/backtest_repro.py
Writes backtest_races.csv (all 71 backtest races, one row each) to the scratchpad.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import spearmanr

BACKEND = Path("/Users/yahirbahena/GitHub/F1 Predicitor Project/backend")
sys.path.insert(0, str(BACKEND))
from ml.features import FEATURE_COLS  # noqa

SCRATCH = Path("/private/tmp/claude-501/-Users-yahirbahena-GitHub-F1-Predicitor-Project/23f5ad82-d8ac-4db8-b758-a0b3b205ee89/scratchpad")

BLEND_W = 0.55
MAX_LABEL = 25
PARAMS = dict(objective="lambdarank", n_estimators=400, learning_rate=0.03,
             num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=2.0, random_state=42, verbose=-1)


def _fit(train):
    train = train.sort_values(["season", "round"], kind="stable")
    groups = train.groupby(["season", "round"], sort=False).size().to_numpy()
    labels = (MAX_LABEL - train["target_pos"]).clip(lower=0).astype(int)
    m = LGBMRanker(**PARAMS)
    m.fit(train[FEATURE_COLS], labels, group=groups)
    return m


def blend_scores(raw, grid):
    model_rank = (-raw).argsort().argsort() + 1.0
    if grid is None:
        return model_rank
    grid = np.asarray(grid, float)
    if np.all(np.isnan(grid)):
        return model_rank
    grid = np.where(np.isnan(grid), float(len(grid)), grid)
    grid_rank = grid.argsort().argsort() + 1.0
    return (1.0 - BLEND_W) * grid_rank + BLEND_W * model_rank + 1e-6 * grid_rank


def main():
    df = pd.read_parquet(BACKEND / "data/processed/dataset.parquet")
    meta = pd.read_parquet(BACKEND / "data/processed/race_meta.parquet")
    metacols = ["season", "round", "wet_race", "sc_laps", "vsc_laps",
                "total_overtakes_clean", "n_starters", "n_dnf", "track_temp", "air_temp"]
    meta = meta[metacols]

    race_keys = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    race_keys = list(race_keys.itertuples(index=False, name=None))

    # season-average track overtakes for a "low overtaking" flag
    track_ot = df.groupby("location")["track_overtakes_avg"].mean()

    rows = []
    model_hits, base_hits, pod, mae_l, sp_l = [], [], [], [], []
    for i in range(30, len(race_keys)):
        season, rnd = race_keys[i]
        train_keys = set(race_keys[:i])
        train = df[df.apply(lambda r: (r["season"], r["round"]) in train_keys, axis=1)]
        test = df[(df["season"] == season) & (df["round"] == rnd)].copy().reset_index(drop=True)
        if test["target_pos"].isna().any() or len(test) < 10:
            continue

        model = _fit(train)
        grid = test["grid_pos"].fillna(len(test)).to_numpy().astype(float)
        raw = model.predict(test[FEATURE_COLS])
        scores = blend_scores(raw, grid)
        actual = test["target_pos"].to_numpy()

        pred_rank = scores.argsort().argsort() + 1          # 1 = predicted winner (blended)
        model_only_rank = (-raw).argsort().argsort() + 1    # 1 = model favourite (no grid)
        grid_rank = grid.argsort().argsort() + 1

        pw_idx = int(pred_rank.argmin())                    # predicted winner row
        aw_mask = np.where(actual == 1)[0]
        aw_idx = int(aw_mask[0]) if len(aw_mask) else -1    # actual winner row

        winner_hit = int(aw_idx == pw_idx)
        pred_top3 = set(np.where(pred_rank <= 3)[0])
        actual_top3 = set(np.where(actual <= 3)[0])
        rho = spearmanr(pred_rank, actual).statistic
        rho = float(rho) if not np.isnan(rho) else 0.0
        overlap = len(pred_top3 & actual_top3)
        mae = float(np.abs(pred_rank - actual).mean())

        # grid baseline winner hit
        base_pred = grid_rank.argmin()
        base_hit = int(actual[base_pred] == 1)

        def rec(idx, pfx):
            if idx < 0:
                return {}
            r = test.iloc[idx]
            return {
                f"{pfx}_code": r["driver_code"], f"{pfx}_team": r["team"],
                f"{pfx}_grid": float(r["grid_pos"]) if not pd.isna(r["grid_pos"]) else np.nan,
                f"{pfx}_finish": float(actual[idx]),
                f"{pfx}_pred_rank": int(pred_rank[idx]),
                f"{pfx}_model_only_rank": int(model_only_rank[idx]),
                f"{pfx}_grid_rank": int(grid_rank[idx]),
                f"{pfx}_dnf": bool(r["dnf"]),
            }

        row = {"season": int(season), "round": int(rnd),
               "event_name": test.iloc[0]["event_name"], "location": test.iloc[0]["location"],
               "winner_hit": winner_hit, "base_grid_hit": base_hit,
               "podium_overlap": overlap, "spearman": rho, "mae": mae,
               "n_starters_in_test": len(test)}
        row.update(rec(aw_idx, "aw"))   # actual winner
        row.update(rec(pw_idx, "pw"))   # predicted winner
        rows.append(row)

        model_hits.append(winner_hit); base_hits.append(base_hit)
        pod.append(overlap); mae_l.append(mae); sp_l.append(rho)

    out = pd.DataFrame(rows)
    m = meta.rename(columns={"wet_race": "meta_wet", "n_dnf": "meta_n_dnf",
                             "n_starters": "meta_n_starters"})
    out = out.merge(m, on=["season", "round"], how="left")
    out["low_overtake_track"] = out["location"].map(track_ot) < track_ot.median()
    out["track_overtakes_avg"] = out["location"].map(track_ot)

    out.to_csv(SCRATCH / "backtest_races.csv", index=False)
    print("RACES", len(out))
    print("model winner_hit_rate", np.mean(model_hits))
    print("base  winner_hit_rate", np.mean(base_hits))
    print("podium_overlap_avg", np.mean(pod))
    print("position_mae", np.mean(mae_l))
    print("spearman_avg", np.mean(sp_l))


if __name__ == "__main__":
    main()
