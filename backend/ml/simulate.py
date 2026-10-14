"""Monte Carlo race simulation: model scores -> win/podium/points probabilities."""
from __future__ import annotations

import numpy as np

# Two projected finishes closer than this (positions) are treated as a tie: a
# ~0.1-0.2 gap in a 10k-sim mean is noise, not a real ordering. Within a tie
# cluster the higher win probability ranks first, so the board never shows a
# 3%-to-win driver above a 65%-to-win one just because their means differ by 0.01.
MEAN_POS_TIE = 0.25


def rank_order(mean_pos: np.ndarray, win: np.ndarray) -> np.ndarray:
    """Final board order: by projected finish, near-ties (within MEAN_POS_TIE)
    broken by win %.

    Drivers are grouped while within MEAN_POS_TIE of the group's best mean, and
    each group is ordered by descending win probability. Outside a tie cluster the
    ordering is exactly argsort(mean_pos), so clear finishing gaps are unaffected.
    Shared by the live predictor and the history backfill so the two never drift.
    """
    by_mean = sorted(range(len(mean_pos)), key=lambda i: mean_pos[i])
    order: list[int] = []
    group: list[int] = []
    for i in by_mean:
        if group and mean_pos[i] - mean_pos[group[0]] > MEAN_POS_TIE:
            order.extend(sorted(group, key=lambda j: -win[j]))
            group = []
        group.append(i)
    order.extend(sorted(group, key=lambda j: -win[j]))
    return np.array(order, dtype=int)


def simulate_race(
    scores: np.ndarray,
    dnf_probs: np.ndarray,
    sigma: float,
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Simulate a race n_sims times.

    scores: predicted finishing-position score per driver (lower = better).
    dnf_probs: per-driver probability of not finishing.
    sigma: race-day noise (std of backtest residuals, in positions).

    Finishers are ranked by noisy score; DNFs fill the back of the field.
    Returns arrays aligned with the input order.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)

    noisy = scores[None, :] + rng.normal(0.0, sigma, size=(n_sims, n))
    dnf = rng.random((n_sims, n)) < dnf_probs[None, :]
    # DNF cars sort behind every finisher (keep noise as tiebreak among DNFs)
    noisy = np.where(dnf, noisy + 1000.0, noisy)

    order = np.argsort(noisy, axis=1)                # order[s, k] = driver index at position k+1
    positions = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    positions[rows, order] = np.arange(1, n + 1)[None, :]

    return {
        "win_prob": (positions == 1).mean(axis=0),
        "podium_prob": (positions <= 3).mean(axis=0),
        "points_prob": ((positions <= 10) & ~dnf).mean(axis=0),
        "mean_pos": positions.mean(axis=0),
        "p10_pos": np.percentile(positions, 10, axis=0),
        "p90_pos": np.percentile(positions, 90, axis=0),
        "dnf_prob": dnf.mean(axis=0),
    }
