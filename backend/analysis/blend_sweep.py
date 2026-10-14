"""Experiment 1: cache raw walk-forward scores once, then sweep BLEND_W.
Also: override-vs-follow analysis and per-condition (wet/dry) blend behavior.
Run from backend/ with the project venv."""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

BACKEND = Path("/Users/yahirbahena/GitHub/F1 Predicitor Project/backend")
sys.path.insert(0, str(BACKEND))
from ml.features import FEATURE_COLS  # noqa

SCRATCH = Path("/private/tmp/claude-501/-Users-yahirbahena-GitHub-F1-Predicitor-Project/23f5ad82-d8ac-4db8-b758-a0b3b205ee89/scratchpad")
CACHE = SCRATCH / "raw_scores_cache.pkl"

MAX_LABEL = 25
PARAMS = dict(objective="lambdarank", n_estimators=400, learning_rate=0.03,
             num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=2.0, random_state=42, verbose=-1)


def build_cache():
    df = pd.read_parquet(BACKEND / "data/processed/dataset.parquet")
    race_keys = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    race_keys = list(race_keys.itertuples(index=False, name=None))
    races = []
    for i in range(30, len(race_keys)):
        season, rnd = race_keys[i]
        train_keys = set(race_keys[:i])
        train = df[df.apply(lambda r: (r["season"], r["round"]) in train_keys, axis=1)]
        test = df[(df["season"] == season) & (df["round"] == rnd)].reset_index(drop=True)
        if test["target_pos"].isna().any() or len(test) < 10:
            continue
        train = train.sort_values(["season", "round"], kind="stable")
        groups = train.groupby(["season", "round"], sort=False).size().to_numpy()
        labels = (MAX_LABEL - train["target_pos"]).clip(lower=0).astype(int)
        m = LGBMRanker(**PARAMS)
        m.fit(train[FEATURE_COLS], labels, group=groups)
        raw = m.predict(test[FEATURE_COLS])
        races.append({
            "season": int(season), "round": int(rnd),
            "raw": raw,
            "grid": test["grid_pos"].fillna(len(test)).to_numpy().astype(float),
            "actual": test["target_pos"].to_numpy(),
        })
    with open(CACHE, "wb") as f:
        pickle.dump(races, f)
    return races


def blend(raw, grid, w):
    model_rank = (-raw).argsort().argsort() + 1.0
    grid_rank = grid.argsort().argsort() + 1.0
    return (1.0 - w) * grid_rank + w * model_rank + 1e-6 * grid_rank


def main():
    if CACHE.exists():
        races = pickle.load(open(CACHE, "rb"))
    else:
        races = build_cache()
    print(f"{len(races)} cached backtest races")

    meta = pd.read_parquet(BACKEND / "data/processed/race_meta.parquet")
    wet = {(int(r.season), int(r.round)): bool(r.wet_race) for r in meta.itertuples()}

    print("\n=== BLEND_W SWEEP (winner_hit_rate | podium_overlap | mae) ===")
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]:
        hits, pods, maes = [], [], []
        wet_hits, dry_hits = [], []
        for rc in races:
            s = blend(rc["raw"], rc["grid"], w)
            pr = s.argsort().argsort() + 1
            h = int(rc["actual"][pr.argmin()] == 1)
            hits.append(h)
            pods.append(len(set(np.where(pr <= 3)[0]) & set(np.where(rc["actual"] <= 3)[0])))
            maes.append(float(np.abs(pr - rc["actual"]).mean()))
            (wet_hits if wet[(rc["season"], rc["round"])] else dry_hits).append(h)
        print(f"  w={w:4.2f}  hit={np.mean(hits):.3f}  pod={np.mean(pods):.2f}  mae={np.mean(maes):.2f}"
              f"  | wet_hit={np.mean(wet_hits):.3f} (n={len(wet_hits)})  dry_hit={np.mean(dry_hits):.3f}")

    # Override analysis at w=0.55
    print("\n=== OVERRIDE vs FOLLOW at BLEND_W=0.55 ===")
    ov_hit, ov_n, fo_hit, fo_n = 0, 0, 0, 0
    ov_rows = []
    for rc in races:
        s = blend(rc["raw"], rc["grid"], 0.55)
        pr = s.argsort().argsort() + 1
        grid_rank = rc["grid"].argsort().argsort() + 1
        pred_idx = pr.argmin(); grid_idx = grid_rank.argmin()
        hit = int(rc["actual"][pred_idx] == 1)
        if pred_idx != grid_idx:
            ov_n += 1; ov_hit += hit
            ov_rows.append((rc["season"], rc["round"], hit,
                            int(rc["actual"][grid_idx] == 1)))
        else:
            fo_n += 1; fo_hit += hit
    print(f"  overrides: {ov_n} races, hit {ov_hit} ({ov_hit/ov_n:.3f})")
    print(f"  follows grid leader: {fo_n} races, hit {fo_hit} ({fo_hit/fo_n:.3f})")
    grid_would = sum(g for _, _, _, g in ov_rows)
    print(f"  in the {ov_n} overridden races, grid leader actually won {grid_would} times")
    print(f"  net gain from overriding: {ov_hit - grid_would:+d} wins")

    # 'pole-protection' rule: only allow the blend to override the pole-sitter
    # when the model STRONGLY demotes them (model rank of pole > K).
    print("\n=== RULE: revert override to pole-sitter unless model demotes pole below rank K ===")
    for K in [2, 3, 4, 5]:
        hits = []
        for rc in races:
            s = blend(rc["raw"], rc["grid"], 0.55)
            pr = s.argsort().argsort() + 1
            grid_rank = rc["grid"].argsort().argsort() + 1
            model_rank = (-rc["raw"]).argsort().argsort() + 1
            pred_idx = pr.argmin(); pole_idx = grid_rank.argmin()
            if pred_idx != pole_idx and model_rank[pole_idx] <= K:
                pred_idx = pole_idx  # weak demotion -> trust the pole
            hits.append(int(rc["actual"][pred_idx] == 1))
        print(f"  K={K}: hit_rate={np.mean(hits):.3f}")

    # Per-condition blend: wet -> model-only, dry -> 0.55
    print("\n=== PER-CONDITION BLEND (wet w=1.0, dry w=0.55) ===")
    hits, pods, maes = [], [], []
    for rc in races:
        w = 1.0 if wet[(rc["season"], rc["round"])] else 0.55
        s = blend(rc["raw"], rc["grid"], w)
        pr = s.argsort().argsort() + 1
        hits.append(int(rc["actual"][pr.argmin()] == 1))
        pods.append(len(set(np.where(pr <= 3)[0]) & set(np.where(rc["actual"] <= 3)[0])))
        maes.append(float(np.abs(pr - rc["actual"]).mean()))
    print(f"  hit={np.mean(hits):.3f} pod={np.mean(pods):.2f} mae={np.mean(maes):.2f}")
    # NOTE: wet_race is not knowable pre-race with certainty; this is an upper
    # bound assuming a good pre-race rain forecast (is_wet feature analogue).


if __name__ == "__main__":
    main()
