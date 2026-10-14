"""Experiment: add the strategy features already present in dataset.parquet
(pit_gain_5, pit_stop_delta_5, undercut_rate_10, stint1_pace_5) to FEATURE_COLS
and re-run the walk-forward backtest. Hypothesis: helps the strategy-variance
miss bucket (winner from grid>=3 in dry races)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

BACKEND = Path("/Users/yahirbahena/GitHub/F1 Predicitor Project/backend")
sys.path.insert(0, str(BACKEND))
from ml.features import FEATURE_COLS  # noqa

EXTRA = ["pit_gain_5", "pit_stop_delta_5", "undercut_rate_10", "stint1_pace_5"]
MAX_LABEL = 25
BLEND_W = 0.55
PARAMS = dict(objective="lambdarank", n_estimators=400, learning_rate=0.03,
             num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=2.0, random_state=42, verbose=-1)


def run(feats):
    df = pd.read_parquet(BACKEND / "data/processed/dataset.parquet")
    keys = list(df[["season", "round"]].drop_duplicates()
                .sort_values(["season", "round"]).itertuples(index=False, name=None))
    hits, pods, maes = [], [], []
    per_race = []
    for i in range(30, len(keys)):
        season, rnd = keys[i]
        tk = set(keys[:i])
        train = df[df.apply(lambda r: (r["season"], r["round"]) in tk, axis=1)]
        test = df[(df["season"] == season) & (df["round"] == rnd)].reset_index(drop=True)
        if test["target_pos"].isna().any() or len(test) < 10:
            continue
        train = train.sort_values(["season", "round"], kind="stable")
        groups = train.groupby(["season", "round"], sort=False).size().to_numpy()
        labels = (MAX_LABEL - train["target_pos"]).clip(lower=0).astype(int)
        m = LGBMRanker(**PARAMS)
        m.fit(train[feats], labels, group=groups)
        raw = m.predict(test[feats])
        grid = test["grid_pos"].fillna(len(test)).to_numpy().astype(float)
        mr = (-raw).argsort().argsort() + 1.0
        gr = grid.argsort().argsort() + 1.0
        s = (1 - BLEND_W) * gr + BLEND_W * mr + 1e-6 * gr
        pr = s.argsort().argsort() + 1
        actual = test["target_pos"].to_numpy()
        h = int(actual[pr.argmin()] == 1)
        hits.append(h)
        pods.append(len(set(np.where(pr <= 3)[0]) & set(np.where(actual <= 3)[0])))
        maes.append(float(np.abs(pr - actual).mean()))
        per_race.append((int(season), int(rnd), h))
    return np.mean(hits), np.mean(pods), np.mean(maes), per_race


if __name__ == "__main__":
    h0, p0, m0, base_races = run(FEATURE_COLS)
    print(f"BASE (28 feats):      hit={h0:.3f} pod={p0:.2f} mae={m0:.2f}")
    h1, p1, m1, ext_races = run(FEATURE_COLS + EXTRA)
    print(f"+strategy (32 feats): hit={h1:.3f} pod={p1:.2f} mae={m1:.2f}")
    flips = [(a[0], a[1], a[2], b[2]) for a, b in zip(base_races, ext_races) if a[2] != b[2]]
    print("flipped races (season, round, base_hit, ext_hit):", flips)
