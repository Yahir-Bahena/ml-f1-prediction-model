"""Fit + validate Platt calibration for win probabilities on the backtest.

Prints the reliability table (raw and calibrated), out-of-sample time-split
validation, and the PLATT_A / PLATT_B parameters to paste into
ml/calibration.py. Rerun after major model changes.

Run from backend/:  ./.venv/bin/python analysis/calibration_fit.py
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from ml.dataset import DATASET_PARQUET
from ml.evaluate import _blend, _dnf_probs, _load_or_run
from ml.features import FEATURE_COLS
from ml.simulate import simulate_race
from ml.train import PARAMS


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(rs: list) -> tuple[float, float]:
    """2-parameter logistic regression on log-odds via IRLS."""
    X = np.concatenate([logit(w) for w, _ in rs])
    y = np.concatenate([(np.arange(len(w)) == aw).astype(float) for w, aw in rs])
    a, b = 1.0, 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(a * X + b)))
        wd = np.clip(p * (1 - p), 1e-6, None)
        g = np.array([np.sum((p - y) * X), np.sum(p - y)])
        h = np.array([[np.sum(wd * X * X), np.sum(wd * X)], [np.sum(wd * X), np.sum(wd)]])
        da, db = np.linalg.solve(h, g)
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-9:
            break
    return float(a), float(b)


def apply_calib(w: np.ndarray, a: float, b: float) -> np.ndarray:
    p = 1 / (1 + np.exp(-(a * logit(w) + b)))
    return p / p.sum()


def brier(rs: list, a: float | None = None, b: float | None = None) -> tuple[float, float, float]:
    bs, tp, tw = [], [], []
    for w, aw in rs:
        p = apply_calib(w, a, b) if a is not None else w
        y = (np.arange(len(p)) == aw).astype(float)
        bs.append(np.mean((p - y) ** 2))
        tp.append(p.max())
        tw.append(int(np.argmax(p) == aw))
    return float(np.mean(bs)), float(np.mean(tp)), float(np.mean(tw))


def main() -> None:
    df = pd.read_parquet(DATASET_PARQUET)
    races, _ = _load_or_run(df, FEATURE_COLS, PARAMS, use_cache=True)
    sigma = float(joblib.load("data/models/model.joblib")["sigma"])

    rs = []
    for rc in races:
        s = _blend(rc["raw"], rc["grid"], 0.55)
        sim = simulate_race(s, _dnf_probs(rc, None), sigma, n_sims=20000, seed=42)
        aw = int(np.where(rc["actual"] == 1)[0][0])
        rs.append((sim["win_prob"], aw))

    b0, tp0, tw0 = brier(rs)
    a, b = fit_platt(rs)
    b1, tp1, tw1 = brier(rs, a, b)
    print(f"raw:        Brier {b0:.5f} | top-pick prob {tp0:.3f} vs actual {tw0:.3f}")
    print(f"calibrated: Brier {b1:.5f} | top-pick prob {tp1:.3f} vs actual {tw1:.3f}")

    half = len(rs) // 2
    for name, fit_rs, test_rs in [("FORWARD", rs[:half], rs[half:]), ("REVERSE", rs[half:], rs[:half])]:
        af, bf = fit_platt(fit_rs)
        braw = brier(test_rs)[0]
        bcal = brier(test_rs, af, bf)[0]
        print(f"{name} split: OOS Brier {braw:.5f} -> {bcal:.5f} (a={af:.2f}, b={bf:.2f})")

    print(f"\nPLATT_A = {a:.4f}\nPLATT_B = {b:.4f}")


if __name__ == "__main__":
    main()
