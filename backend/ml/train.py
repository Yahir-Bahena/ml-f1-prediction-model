"""Train the race-ranking model with a walk-forward backtest.

Run from backend/:  python -m ml.train
Saves model.joblib + metrics.json to data/models/.

Architecture (chosen via walk-forward ablation, see metrics.json):
- LightGBM lambdarank ranker, grouped by race, label = quality of finish
  (higher = better finish). Learns *within-race ordering* rather than
  absolute finishing position, which is what the evaluation cares about.
- The final score blends the model's rank with the grid-position prior:
      score = (1 - w) * grid_rank + w * model_rank,  w = BLEND_W
  On ~2k rows the model cannot fully re-learn the dominant grid signal,
  so anchoring to it beats both the raw model and the grid baseline.
- In the wet the grid prior is unreliable, so when rain is likely (pre-race
  rain probability p_wet >= WET_BLEND_THRESHOLD) the model weight is raised:
  w = BLEND_W + (1 - BLEND_W) * clip(p_wet, 0, 1). The backtest keeps
  w = BLEND_W (no realized-wet leakage); the wet-conditional upper bound is
  tracked separately as metrics["wet_conditional_oracle"].
- Before qualifying (no grid yet) the model rank is used alone.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import spearmanr

import config
from ml.dataset import DATASET_PARQUET
from ml.features import FEATURE_COLS

MODEL_PATH = config.MODELS_DIR / "model.joblib"
METRICS_PATH = config.MODELS_DIR / "metrics.json"

# Weight of the model rank vs. the grid-rank prior in the final score.
# Chosen for stability across seeds in the walk-forward backtest.
BLEND_W = 0.55

# Rain probability below which the blend stays at BLEND_W. Mid-range weights
# (w ~ 0.65-0.8) measurably hurt dry races, so marginal rain chances must not
# nudge w upward; 0.5 also matches weather.WET_PROB_THRESHOLD.
WET_BLEND_THRESHOLD = 0.5

# Ranking label: higher = better finish. 25 keeps every classified position
# positive (max grid ~22) and stays within LightGBM's default label_gain range.
MAX_LABEL = 25

PARAMS = dict(
    objective="lambdarank",
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    random_state=42,
    verbose=-1,
)


def _fit(train: pd.DataFrame) -> LGBMRanker:
    """Fit the lambdarank ranker with one group per race (time-ordered)."""
    train = train.sort_values(["season", "round"], kind="stable")
    groups = train.groupby(["season", "round"], sort=False).size().to_numpy()
    labels = (MAX_LABEL - train["target_pos"]).clip(lower=0).astype(int)
    model = LGBMRanker(**PARAMS)
    model.fit(train[FEATURE_COLS], labels, group=groups)
    return model


def blend_scores(
    raw_scores: np.ndarray, grid: np.ndarray | None, p_wet: float | None = None
) -> np.ndarray:
    """Combine raw ranker output (higher = better) with the grid prior.

    Returns a position-like score (lower = better, ~1..n) suitable for the
    Monte Carlo simulation. `grid` may be None or all-NaN (pre-qualifying),
    in which case the model rank is used alone.

    `p_wet` is an optional rain probability in [0, 1]. When rain is likely
    (p_wet >= WET_BLEND_THRESHOLD) the model weight is raised toward 1.0:
        w = BLEND_W + (1 - BLEND_W) * clip(p_wet, 0, 1)
    The grid prior is a poor guide in the wet, where results scramble; leaning on
    the model there recovers wet-race winners (see analysis/weather_blend_noise_test).
    Below the threshold the default blend applies unchanged — mid-range weights
    hurt dry races, so marginal rain chances must not move w. Default p_wet=None
    reproduces the dry-weather blend byte-for-byte.
    """
    model_rank = (-raw_scores).argsort().argsort() + 1.0  # 1 = model's favourite
    if grid is None:
        return model_rank
    grid = np.asarray(grid, dtype=float)
    if np.all(np.isnan(grid)):
        return model_rank
    w = BLEND_W
    if p_wet is not None and not np.isnan(p_wet) and p_wet >= WET_BLEND_THRESHOLD:
        w = BLEND_W + (1.0 - BLEND_W) * float(np.clip(p_wet, 0.0, 1.0))
    grid = np.where(np.isnan(grid), float(len(grid)), grid)
    grid_rank = grid.argsort().argsort() + 1.0
    # tiny grid term breaks exact blend ties toward the better qualifier
    return (1.0 - w) * grid_rank + w * model_rank + 1e-6 * grid_rank


def predict_scores(
    model: LGBMRanker, X: pd.DataFrame, grid: np.ndarray | None, p_wet: float | None = None
) -> np.ndarray:
    """Full prediction path: ranker + grid blend. Lower score = better.

    `p_wet` (rain probability) is threaded to blend_scores to raise the model
    weight in the wet; None keeps the default dry-weather blend.
    """
    return blend_scores(model.predict(X), grid, p_wet)


def _race_eval(pred_scores: np.ndarray, actual: np.ndarray) -> dict:
    """Compare a predicted ordering against actual finishing positions."""
    pred_rank = pred_scores.argsort().argsort() + 1  # 1 = predicted winner
    winner_hit = int(actual[pred_rank.argmin()] == 1)
    pred_top3 = set(np.where(pred_rank <= 3)[0])
    actual_top3 = set(np.where(actual <= 3)[0])
    rho = spearmanr(pred_rank, actual).statistic
    return {
        "winner_hit": winner_hit,
        "podium_overlap": len(pred_top3 & actual_top3),
        "mae": float(np.abs(pred_rank - actual).mean()),
        "spearman": float(rho) if not np.isnan(rho) else 0.0,
        "residuals": (actual - pred_scores).tolist(),
    }


def backtest(df: pd.DataFrame, min_train_races: int = 30) -> tuple[dict, dict, dict, float]:
    """Walk forward race by race; also score the grid-position baseline.

    Returns (model, baseline_grid, wet_conditional_oracle, sigma). The oracle is
    a secondary upper-bound only: it blends with p_wet = the realized wet flag
    (w=1.0 wet, BLEND_W dry). That flag is NOT available pre-race, so it never
    feeds the honest `model` metrics -- it just tracks how much a perfect rain
    forecast would be worth across retrains.
    """
    race_keys = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    race_keys = list(race_keys.itertuples(index=False, name=None))

    model_evals, base_evals, oracle_evals, residuals = [], [], [], []
    for i in range(min_train_races, len(race_keys)):
        season, rnd = race_keys[i]
        train_keys = set(race_keys[:i])
        train = df[df.apply(lambda r: (r["season"], r["round"]) in train_keys, axis=1)]
        test = df[(df["season"] == season) & (df["round"] == rnd)]
        if test["target_pos"].isna().any() or len(test) < 10:
            continue

        model = _fit(train)
        grid = test["grid_pos"].fillna(len(test)).to_numpy().astype(float)
        raw = model.predict(test[FEATURE_COLS])
        scores = blend_scores(raw, grid)
        actual = test["target_pos"].to_numpy()

        ev = _race_eval(scores, actual)
        residuals.extend(ev.pop("residuals"))
        model_evals.append(ev)

        bev = _race_eval(grid, actual)
        bev.pop("residuals")
        base_evals.append(bev)

        # secondary: perfect-forecast oracle (realized wet flag; leaks -> not honest)
        p_wet = float(test["is_wet"].iloc[0]) if not pd.isna(test["is_wet"].iloc[0]) else 0.0
        oev = _race_eval(blend_scores(raw, grid, p_wet), actual)
        oev.pop("residuals")
        oracle_evals.append(oev)

    def agg(evals: list[dict]) -> dict:
        return {
            "races": len(evals),
            "winner_hit_rate": float(np.mean([e["winner_hit"] for e in evals])),
            "podium_overlap_avg": float(np.mean([e["podium_overlap"] for e in evals])),
            "position_mae": float(np.mean([e["mae"] for e in evals])),
            "spearman_avg": float(np.mean([e["spearman"] for e in evals])),
        }

    sigma = float(np.std(residuals)) if residuals else 4.0
    return agg(model_evals), agg(base_evals), agg(oracle_evals), sigma


def main() -> None:
    config.ensure_dirs()
    df = pd.read_parquet(DATASET_PARQUET)
    print(f"Dataset: {len(df)} rows, {df.groupby(['season', 'round']).ngroups} races")

    model_metrics, baseline_metrics, oracle_metrics, sigma = backtest(df)
    print("Backtest (model):   ", json.dumps(model_metrics, indent=2))
    print("Backtest (grid=pos):", json.dumps(baseline_metrics, indent=2))
    print("Backtest (wet oracle):", json.dumps(oracle_metrics, indent=2))
    print(f"Residual sigma: {sigma:.2f} positions")

    final = _fit(df)
    importance = sorted(
        zip(FEATURE_COLS, final.feature_importances_.tolist()), key=lambda x: -x[1]
    )

    joblib.dump(
        {
            "model": final,
            "features": FEATURE_COLS,
            "sigma": sigma,
            "model_type": "lambdarank_grid_blend",
            "blend_w": BLEND_W,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(
        json.dumps(
            {
                "trained_at": pd.Timestamp.utcnow().isoformat(),
                "n_rows": len(df),
                "n_races": int(df.groupby(["season", "round"]).ngroups),
                "sigma": sigma,
                "model": model_metrics,
                "baseline_grid": baseline_metrics,
                "wet_conditional_oracle": oracle_metrics,
                "feature_importance": importance,
            },
            indent=2,
        )
    )
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    main()
