"""Build the prediction-history record: for every completed race from the first
live prediction (2026 Belgian GP) onward, reconstruct what the model predicted
vs what actually happened, plus race-control highlights.

The prediction is reconstructed on the SAME leakage-free basis the model uses in
production: train on races strictly before the target, then predict the target
from its own pre-race feature row (features are history-only + that weekend's
qualifying, exactly as build_training_dataset computes them). So this is a
faithful record of what the model knew before lights out — not hindsight.

Race-control highlights are parsed from FastF1's race_control_messages and are
purely descriptive; they never touch the model.

Idempotent + incremental: existing races are kept, only new ones are added.
Run from backend/:  python -m scripts.backfill_predictions
Writes data/processed/predictions_history.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
from ml.calibration import calibrate_win_probs
from ml.dataset import DATASET_PARQUET
from ml.features import FEATURE_COLS
from ml.simulate import rank_order, simulate_race
from ml.train import METRICS_PATH, _fit, predict_scores

HISTORY_PATH = config.PROCESSED_DIR / "predictions_history.json"
# first race predicted live after the model was built
FIRST_SEASON, FIRST_ROUND = 2026, 10

_SIGMA_CACHE: float | None = None


def _prod_sigma() -> float:
    """Residual sigma used by the live predictor, so the reconstructed board's
    simulation noise matches production (not a per-race std of scores, which is a
    different, larger number and changes near-tie ordering)."""
    global _SIGMA_CACHE
    if _SIGMA_CACHE is None:
        try:
            _SIGMA_CACHE = float(json.loads(METRICS_PATH.read_text())["sigma"])
        except (OSError, ValueError, KeyError):
            _SIGMA_CACHE = 4.5
    return _SIGMA_CACHE


# --------------------------------------------------------------------------- #
# Prediction reconstruction (walk-forward, leakage-free)
# --------------------------------------------------------------------------- #
def _reconstruct(df: pd.DataFrame, rr: pd.DataFrame, season: int, rnd: int) -> list[dict]:
    """Predicted order + win probabilities for one race, model trained on prior races."""
    keys = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    keys = list(keys.itertuples(index=False, name=None))
    idx = keys.index((season, rnd))
    train = df[df.apply(lambda r: (r["season"], r["round"]) in set(keys[:idx]), axis=1)]
    test = df[(df["season"] == season) & (df["round"] == rnd)].reset_index(drop=True)

    model = _fit(train)
    grid = test["grid_pos"].fillna(len(test)).to_numpy(dtype=float)
    scores = predict_scores(model, test[FEATURE_COLS], grid)  # p_wet=None (matches backtest)
    dnf_p = (0.5 * test["driver_dnf_rate"].fillna(0.1)
             + 0.5 * test["team_dnf_rate"].fillna(0.1)).clip(0.02, 0.35).to_numpy()
    sim = simulate_race(scores, dnf_p, _prod_sigma())
    win = calibrate_win_probs(sim["win_prob"])

    # driver display fields from results
    meta = rr[(rr.season == season) & (rr["round"] == rnd)].set_index("driver_code")
    # same ordering as the live board: projected finish, near-ties broken by win %
    order = rank_order(sim["mean_pos"], win)
    rows = []
    for rank, i in enumerate(order, start=1):
        code = test.iloc[i]["driver_code"]
        m = meta.loc[code] if code in meta.index else None
        rows.append({
            "driver_code": code,
            "driver_name": test.iloc[i]["driver_name"],
            "team": test.iloc[i]["team"],
            "team_color": str(m["team_color"]) if m is not None else "888888",
            "driver_number": str(m["driver_number"]) if m is not None else "",
            "predicted_rank": rank,
            "win_prob": round(float(win[i]), 4),
            "grid_pos": None if m is None or pd.isna(m["grid_pos"]) else int(m["grid_pos"]),
        })
    return rows


# --------------------------------------------------------------------------- #
# Race-control highlights (descriptive only)
# --------------------------------------------------------------------------- #
# driver abbreviations appear as "(HAM)" right after each car number in FIA text
_CODE = re.compile(r"\(([A-Z]{3})\)")
_SECS = re.compile(r"(\d+)\s*SECOND")


def _highlights(season: int, rnd: int, num_to_code: dict[str, str]) -> list[dict]:
    import fastf1

    try:
        s = fastf1.get_session(season, rnd, "R")
        s.load(laps=False, telemetry=False, weather=False, messages=True)
        rc = s.race_control_messages
    except Exception:
        return []
    if rc is None or len(rc) == 0:
        return []

    out, seen = [], set()
    for _, r in rc.iterrows():
        msg = str(r.get("Message", "")).strip()
        up = msg.upper()
        lap = r.get("Lap")
        lap = int(lap) if lap is not None and not pd.isna(lap) else None
        cat = str(r.get("Category", ""))
        codes = _CODE.findall(msg)

        kind = text = None
        if re.search(r"\bRED FLAG\b", up) and "CHEQUERED" not in up:
            kind, text = "red_flag", "Red flag — race suspended"
        elif cat == "SafetyCar" and "DEPLOYED" in up and "VIRTUAL" not in up:
            kind, text = "safety_car", "Safety car deployed"
        elif "VIRTUAL SAFETY CAR" in up and "DEPLOYED" in up:
            kind, text = "vsc", "Virtual safety car deployed"
        elif "CAUSING A COLLISION" in up and "STEWARDS" not in up:
            kind = "collision"
            text = (f"Collision involving {' and '.join(codes)}" if codes
                    else "On-track collision")
        elif "TIME PENALTY" in up and "STEWARDS" in up and "SERVED" not in up:
            kind = "penalty"
            secs = _SECS.search(up)
            who = codes[0] if codes else "a driver"
            text = f"{secs.group(1)}s time penalty for {who}" if secs else f"Time penalty for {who}"
        elif "GRID PENALTY" in up:
            kind = "penalty"
            text = f"Grid penalty for {codes[0]}" if codes else "Grid penalty"
        if kind is None:
            continue

        key = (kind, tuple(codes))
        if key in seen:
            continue
        seen.add(key)
        out.append({"lap": lap, "type": kind, "drivers": codes, "text": text})

    out.sort(key=lambda h: (h["lap"] is None, h["lap"] or 0))
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _spearman(pred_rank: np.ndarray, actual: np.ndarray) -> float:
    from scipy.stats import spearmanr
    rho = spearmanr(pred_rank, actual).statistic
    return float(rho) if not np.isnan(rho) else 0.0


def build_race(df: pd.DataFrame, rr: pd.DataFrame, season: int, rnd: int) -> dict | None:
    res = rr[(rr.season == season) & (rr["round"] == rnd)]
    if len(res) == 0 or res["finish_pos"].isna().all():
        return None

    pred = _reconstruct(df, rr, season, rnd)
    finish = res.set_index("driver_code")
    num_to_code = {str(r.driver_number): r.driver_code for r in res.itertuples()}

    drivers = []
    for p in pred:
        f = finish.loc[p["driver_code"]] if p["driver_code"] in finish.index else None
        actual = None if f is None or pd.isna(f["finish_pos"]) else int(f["finish_pos"])
        drivers.append({
            **p,
            "actual_finish": actual,
            "dnf": bool(f["dnf"]) if f is not None else False,
            "status": str(f["status"]) if f is not None else "",
            "delta": None if actual is None else actual - p["predicted_rank"],
        })

    classified = [d for d in drivers if d["actual_finish"] is not None]
    pred_rank = np.array([d["predicted_rank"] for d in classified])
    act = np.array([d["actual_finish"] for d in classified])
    pred_winner = min(drivers, key=lambda d: d["predicted_rank"])
    actual_winner = min(classified, key=lambda d: d["actual_finish"]) if classified else None
    pred_top3 = {d["driver_code"] for d in drivers if d["predicted_rank"] <= 3}
    act_top3 = {d["driver_code"] for d in classified if d["actual_finish"] <= 3}

    meta = res.iloc[0]
    return {
        "season": season, "round": rnd,
        "event_name": str(meta["event_name"]), "location": str(meta["location"]),
        "country": str(meta["country"]),
        "date": pd.Timestamp(meta["date"]).isoformat() if not pd.isna(meta["date"]) else None,
        "wet_race": bool(meta["wet_race"]),
        "metrics": {
            "winner_hit": actual_winner is not None
            and pred_winner["driver_code"] == actual_winner["driver_code"],
            "podium_overlap": len(pred_top3 & act_top3),
            "mae": round(float(np.abs(pred_rank - act).mean()), 2) if len(act) else None,
            "spearman": round(_spearman(pred_rank, act), 3) if len(act) else None,
        },
        "predicted_winner": pred_winner["driver_code"],
        "actual_winner": actual_winner["driver_code"] if actual_winner else None,
        "drivers": drivers,
        "highlights": _highlights(season, rnd, num_to_code),
    }


def main() -> None:
    config.ensure_dirs()
    df = pd.read_parquet(DATASET_PARQUET)
    rr = pd.read_parquet(config.RESULTS_PARQUET)

    existing = {}
    if HISTORY_PATH.exists():
        for r in json.loads(HISTORY_PATH.read_text())["races"]:
            existing[(r["season"], r["round"])] = r

    keys = rr[["season", "round"]].drop_duplicates()
    targets = [
        (int(s), int(rd)) for s, rd in keys.itertuples(index=False)
        if (s, rd) >= (FIRST_SEASON, FIRST_ROUND)
    ]
    for season, rnd in sorted(targets):
        if (season, rnd) in existing:
            continue
        print(f"building prediction record: {season} R{rnd} ...", flush=True)
        race = build_race(df, rr, season, rnd)
        if race:
            existing[(season, rnd)] = race

    races = [existing[k] for k in sorted(existing)]
    HISTORY_PATH.write_text(json.dumps({"races": races}, indent=2))
    print(f"wrote {len(races)} race records -> {HISTORY_PATH}")


if __name__ == "__main__":
    main()
