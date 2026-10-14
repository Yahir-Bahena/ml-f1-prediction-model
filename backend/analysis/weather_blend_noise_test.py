"""Gate experiment: forecast-noise robustness of the weather-conditional blend.

Runs on the SHIPPED 32-feature model (FEATURE_COLS after activating the 4 dormant
strategy features). Caches the raw walk-forward ranker scores once (same cache
pattern as pace_mismatch_experiment.py) and does all noise post-processing on the
cached raws -- cheap, so >=500 Monte Carlo draws per level is trivial.

The weather-conditional blend raises the model weight w in wet races:
    boolean oracle : w = 1.0 if realized wet else 0.55
    continuous     : w = 0.55 + 0.45 * p_wet         (the version actually shipped)

We never get a perfect pre-race forecast, so we simulate an imperfect one from the
realized wet flag and measure how the blend's winner_hit degrades with forecast
accuracy a (prob the forecast agrees with reality).

Run from backend/ with the project venv:
    ./.venv/bin/python analysis/weather_blend_noise_test.py
"""
from __future__ import annotations
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import spearmanr

BACKEND = Path("/Users/yahirbahena/GitHub/F1 Predicitor Project/backend")
sys.path.insert(0, str(BACKEND))
from ml.features import FEATURE_COLS  # noqa: E402

ANALYSIS = BACKEND / "analysis"
CACHE = ANALYSIS / "raw_scores_cache_weather.pkl"

DRY_W = 0.55        # BLEND_W
WET_W = 1.0         # oracle model weight in wet races
MAX_LABEL = 25
TRAIN_ONLY = 30     # first 30 races are train-only (matches ml.train backtest)
PARAMS = dict(objective="lambdarank", n_estimators=400, learning_rate=0.03,
              num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=2.0, random_state=42, verbose=-1)

ACCURACIES = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7]
N_DRAWS = 2000
SEED = 12345


# ---------------------------------------------------------------------------
# Walk-forward: cache raw scores for the 32-feature model (one variant)
# ---------------------------------------------------------------------------
def build_cache(df: pd.DataFrame, wet_by_race: dict) -> list[dict]:
    race_keys = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    race_keys = list(race_keys.itertuples(index=False, name=None))
    df = df.copy()
    ord_map = {k: i for i, k in enumerate(race_keys)}
    df["_ord"] = list(zip(df["season"], df["round"]))
    df["_ord"] = df["_ord"].map(ord_map)

    cache = []
    for i in range(TRAIN_ONLY, len(race_keys)):
        season, rnd = race_keys[i]
        train = df[df["_ord"] < i]
        test = df[(df["season"] == season) & (df["round"] == rnd)].reset_index(drop=True)
        if test["target_pos"].isna().any() or len(test) < 10:
            continue
        train_s = train.sort_values(["season", "round"], kind="stable")
        groups = train_s.groupby(["season", "round"], sort=False).size().to_numpy()
        labels = (MAX_LABEL - train_s["target_pos"]).clip(lower=0).astype(int)
        m = LGBMRanker(**PARAMS)
        m.fit(train_s[FEATURE_COLS], labels, group=groups)
        cache.append({
            "season": int(season), "round": int(rnd),
            "event_name": test.iloc[0]["event_name"],
            "raw": m.predict(test[FEATURE_COLS]),
            "grid": test["grid_pos"].fillna(len(test)).to_numpy().astype(float),
            "actual": test["target_pos"].to_numpy(),
            "wet": bool(wet_by_race.get((int(season), int(rnd)), False)),
        })
    with open(CACHE, "wb") as f:
        pickle.dump({"cache": cache, "features": FEATURE_COLS}, f)
    return cache


def blend(raw: np.ndarray, grid: np.ndarray, w: float) -> np.ndarray:
    model_rank = (-raw).argsort().argsort() + 1.0
    grid_rank = grid.argsort().argsort() + 1.0
    return (1.0 - w) * grid_rank + w * model_rank + 1e-6 * grid_rank


def race_metrics(pred_scores: np.ndarray, actual: np.ndarray) -> tuple[int, int, float, float]:
    pr = pred_scores.argsort().argsort() + 1
    winner_hit = int(actual[pr.argmin()] == 1)
    pod = len(set(np.where(pr <= 3)[0]) & set(np.where(actual <= 3)[0]))
    mae = float(np.abs(pr - actual).mean())
    rho = spearmanr(pr, actual).statistic
    rho = float(rho) if not np.isnan(rho) else 0.0
    return winner_hit, pod, mae, rho


def eval_fixed(cache: list[dict], w_of_race) -> dict:
    """Deterministic evaluation; w_of_race(rc) -> blend weight for that race."""
    hits, pods, maes, rhos = [], [], [], []
    for rc in cache:
        s = blend(rc["raw"], rc["grid"], float(w_of_race(rc)))
        h, p, m, r = race_metrics(s, rc["actual"])
        hits.append(h); pods.append(p); maes.append(m); rhos.append(r)
    return {
        "n": len(cache),
        "winner_hit": float(np.mean(hits)),
        "hits": int(np.sum(hits)),
        "podium": float(np.mean(pods)),
        "mae": float(np.mean(maes)),
        "spearman": float(np.mean(rhos)),
    }


def main() -> None:
    df = pd.read_parquet(BACKEND / "data/processed/dataset.parquet")
    rm = pd.read_parquet(BACKEND / "data/processed/race_meta.parquet")
    wet_by_race = {(int(r.season), int(r.round)): bool(r.wet_race)
                   for r in rm.itertuples(index=False)}

    if CACHE.exists():
        payload = pickle.load(open(CACHE, "rb"))
        cache = payload["cache"] if payload.get("features") == FEATURE_COLS else build_cache(df, wet_by_race)
    else:
        cache = build_cache(df, wet_by_race)

    print(f"# Weather-Conditional Blend: Forecast-Noise Robustness (32-feature model)\n")
    print(f"Eval races: {len(cache)}   Wet races: {sum(rc['wet'] for rc in cache)}   "
          f"Features: {len(FEATURE_COLS)}\n")

    # ---- Reference points -------------------------------------------------
    base = eval_fixed(cache, lambda rc: DRY_W)                 # always w=0.55
    oracle = eval_fixed(cache, lambda rc: WET_W if rc["wet"] else DRY_W)
    print("## Reference (deterministic)")
    print(f"- Static blend  (w=0.55 always) : hit {base['winner_hit']:.4f} "
          f"({base['hits']}/{base['n']}), podium {base['podium']:.4f}, mae {base['mae']:.4f}")
    print(f"- Oracle wet-cond (w=1.0 wet)   : hit {oracle['winner_hit']:.4f} "
          f"({oracle['hits']}/{oracle['n']}), podium {oracle['podium']:.4f}, mae {oracle['mae']:.4f}")
    print(f"  (28-feat oracle was 0.676 hit)\n")

    # ---- Precompute per-race winner_hit under dry (0.55) and wet (1.0) -----
    realized = np.array([rc["wet"] for rc in cache], dtype=bool)
    hit_dry = np.array([race_metrics(blend(rc["raw"], rc["grid"], DRY_W), rc["actual"])[0] for rc in cache])
    hit_wet = np.array([race_metrics(blend(rc["raw"], rc["grid"], WET_W), rc["actual"])[0] for rc in cache])
    n_races = len(cache)
    rng = np.random.default_rng(SEED)

    def mc_boolean(miss_rate_fn) -> tuple[float, float]:
        """miss_rate_fn(realized_bool_array) -> per-race flip probability."""
        flip_prob = miss_rate_fn(realized)                     # (n_races,)
        rand = rng.random((N_DRAWS, n_races))
        flip = rand < flip_prob                                 # forecast disagrees
        pred_wet = realized[None, :] ^ flip                     # predicted wet flag
        selected = np.where(pred_wet, hit_wet[None, :], hit_dry[None, :])
        draw_hit = selected.mean(axis=1)
        return float(draw_hit.mean()), float(np.percentile(draw_hit, 5))

    # ---- Symmetric boolean noise -----------------------------------------
    print("## Boolean oracle blend under symmetric forecast noise "
          f"(flip prob = 1-a, {N_DRAWS} draws)")
    print("| accuracy a | mean winner_hit | p5 winner_hit |")
    print("|---|---|---|")
    bool_results = {}
    for a in ACCURACIES:
        mean, p5 = mc_boolean(lambda rel, a=a: np.full(rel.shape, 1.0 - a))
        bool_results[a] = (mean, p5)
        print(f"| {a:.2f} | {mean:.4f} | {p5:.4f} |")
    print()

    # ---- Asymmetric boolean noise (false-alarm = 2x miss) -----------------
    print("## Boolean oracle blend under asymmetric noise "
          "(false-alarm rate = 2x miss rate; miss=1-a on wet, fa=2(1-a) on dry)")
    print("| accuracy a | miss rate | false-alarm rate | mean winner_hit | p5 winner_hit |")
    print("|---|---|---|---|---|")
    for a in [0.9, 0.8]:
        miss = 1.0 - a
        fa = min(2.0 * miss, 1.0)
        mean, p5 = mc_boolean(lambda rel, miss=miss, fa=fa: np.where(rel, miss, fa))
        print(f"| {a:.2f} | {miss:.3f} | {fa:.3f} | {mean:.4f} | {p5:.4f} |")
    print()

    # ---- Continuous shipped version: w = 0.55 + 0.45*p_wet ---------------
    # p_wet proxy = a if realized wet else (1-a). Deterministic given a.
    print("## Continuous shipped blend  w = 0.55 + 0.45*p_wet, "
          "p_wet = a if wet else (1-a)")
    print("| accuracy a | winner_hit | podium | mae |")
    print("|---|---|---|---|")
    cont_results = {}
    for a in ACCURACIES:
        def w_of(rc, a=a):
            p_wet = a if rc["wet"] else (1.0 - a)
            return DRY_W + (1.0 - DRY_W) * p_wet
        r = eval_fixed(cache, w_of)
        cont_results[a] = r
        print(f"| {a:.2f} | {r['winner_hit']:.4f} | {r['podium']:.4f} | {r['mae']:.4f} |")
    print()

    # ---- PASS / FAIL ------------------------------------------------------
    # Criteria on the shipped (continuous) blend, mirrored by boolean.
    def pass_check(res_at):
        return (res_at(0.9) >= 0.655) and (res_at(0.8) >= 0.648)

    bool_90, bool_80 = bool_results[0.9][0], bool_results[0.8][0]
    cont_90, cont_80 = cont_results[0.9]["winner_hit"], cont_results[0.8]["winner_hit"]
    print("## PASS/FAIL (mean winner_hit >= 0.655 @ a=0.90 AND >= 0.648 @ a=0.80)")
    print(f"- Boolean    : a=0.90 {bool_90:.4f}  a=0.80 {bool_80:.4f}  -> "
          f"{'PASS' if pass_check(lambda a: bool_results[a][0]) else 'FAIL'}")
    print(f"- Continuous : a=0.90 {cont_90:.4f}  a=0.80 {cont_80:.4f}  -> "
          f"{'PASS' if pass_check(lambda a: cont_results[a]['winner_hit']) else 'FAIL'}")
    overall = pass_check(lambda a: cont_results[a]["winner_hit"]) and \
              pass_check(lambda a: bool_results[a][0])
    print(f"\n### GATE: {'PASS -> wire change 2' if overall else 'FAIL -> stop after change 1'}")


if __name__ == "__main__":
    main()
