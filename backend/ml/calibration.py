"""Win-probability calibration (Platt scaling).

The Monte Carlo simulation's single global sigma flattens win probabilities:
the backtest reliability curve showed drivers given 30-40% actually won 69% of
the time, while 5-20% drivers won far less than stated. A two-parameter
logistic map on the log-odds, fit on the 71-race walk-forward backtest,
corrects this — validated out-of-sample in both time-split directions
(Brier 0.0329 -> ~0.026).

Applied at serving time to win probabilities only (podium/points probabilities
are not yet calibrated). Refit after major model changes with:
    ./.venv/bin/python analysis/calibration_fit.py
and paste the printed parameters here.
"""
from __future__ import annotations

import numpy as np

# Platt parameters from analysis/calibration_fit.py on the 33-feature model
# (2026-07-19): calibrated = sigmoid(A * logit(raw) + B), renormalized per race.
PLATT_A = 2.9026
PLATT_B = 2.7299


def calibrate_win_probs(win_prob: np.ndarray) -> np.ndarray:
    """Map raw simulated win probabilities to calibrated ones.

    Renormalizes so the field's probabilities sum to 1 (exactly one winner),
    which also keeps the map safe for degenerate inputs.
    """
    p = np.clip(np.asarray(win_prob, dtype=float), 1e-6, 1 - 1e-6)
    z = PLATT_A * np.log(p / (1 - p)) + PLATT_B
    cal = 1.0 / (1.0 + np.exp(-z))
    total = cal.sum()
    return cal / total if total > 0 else cal
