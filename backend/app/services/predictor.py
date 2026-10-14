"""Next-race prediction: model scores -> Monte Carlo probabilities + explanations."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

import config
from app.services import store, weather
from ml.calibration import calibrate_win_probs
from ml.features import FEATURE_COLS, build_prediction_rows
from ml.simulate import rank_order, simulate_race
from ml.train import MODEL_PATH, predict_scores

FRIENDLY = {
    "quali_pos": "Qualifying position",
    "quali_gap_to_pole_ms": "Gap to pole",
    "quali_gap_to_teammate": "Quali vs teammate",
    "form_finish_3": "Recent finishes (3)",
    "form_finish_5": "Recent finishes (5)",
    "form_points_5": "Recent points haul",
    "form_gain_5": "Recent places gained",
    "form_quali_5": "Recent quali form",
    "season_points": "Season points",
    "season_rank": "Championship position",
    "pace_delta_5": "Race pace vs field",
    "consistency_5": "Lap-time consistency",
    "deg_slope_5": "Tyre degradation",
    "pit_gain_5": "Pit-cycle gains",
    "pit_stop_delta_5": "Pit-stop speed",
    "undercut_rate_10": "Undercut success",
    "stint1_pace_5": "Opening-stint pace",
    "driver_overtakes_10": "Overtaking record",
    "lap1_gain_5": "Lap-1 starts",
    "track_overtakes_avg": "Track overtaking difficulty",
    "track_sc_rate": "Safety-car likelihood",
    "track_finish_avg": "History at this track",
    "track_gain_avg": "Places gained at this track",
    "track_races": "Experience at this track",
    "team_track_finish_avg": "Team record at this track",
    "team_form_finish_5": "Team form",
    "team_season_points": "Team season points",
    "driver_dnf_rate": "Reliability (driver)",
    "team_dnf_rate": "Reliability (car)",
    "wet_delta": "Wet-weather skill",
    "is_wet": "Wet race",
    "n_races": "Experience",
}

_model_cache: dict = {}


def load_model() -> dict:
    mtime = MODEL_PATH.stat().st_mtime
    if _model_cache.get("mtime") != mtime:
        _model_cache.update(mtime=mtime, bundle=joblib.load(MODEL_PATH))
    return _model_cache["bundle"]


def _display_value(feature: str, value: float) -> str:
    if feature.endswith("_ms") or feature in ("pace_delta_5", "consistency_5"):
        return f"{value / 1000:+.2f}s" if "delta" in feature or "gap" in feature else f"{value / 1000:.2f}s"
    if feature in ("quali_pos", "form_finish_3", "form_finish_5", "form_quali_5",
                   "season_rank", "track_finish_avg", "team_track_finish_avg"):
        return f"P{value:.0f}" if value == int(value) else f"P{value:.1f}"
    if "rate" in feature or feature == "is_wet":
        return f"{value * 100:.0f}%"
    return f"{value:.1f}"


def _explanations(model, X: pd.DataFrame, top_n: int = 4) -> list[list[dict]]:
    """Per-driver top SHAP contributions on the ranker score.

    The lambdarank model outputs higher = better, so a positive contribution
    pushes the driver toward P1 (impact is reported with + = helps).
    """
    contribs = model.predict(X, pred_contrib=True)[:, :-1]  # drop bias column
    out = []
    for i in range(len(X)):
        row = contribs[i]
        order = np.argsort(np.abs(row))[::-1]
        picks = []
        for j in order:
            if len(picks) >= top_n or abs(row[j]) < 0.05:
                break
            feature = FEATURE_COLS[j]
            value = X.iloc[i, j]
            # a quali "reason" with no quali data yet only confuses the reader
            if pd.isna(value) and feature.startswith("quali"):
                continue
            picks.append(
                {
                    "feature": feature,
                    "label": FRIENDLY.get(feature, feature),
                    "impact": round(float(row[j]), 2),  # + = helps, - = hurts
                    "value": None if pd.isna(value) else _display_value(feature, float(value)),
                }
            )
        out.append(picks)
    return out


def _apply_grid_penalties(
    codes: list[str], quali: np.ndarray, penalties: dict[str, int]
) -> np.ndarray:
    """Post-penalty starting position (1..n) per driver, aligned to `codes`.

    Follows the intuitive rule the grid is actually published by: a penalized
    driver starts at (qualifying position + penalty places); everyone else keeps
    their qualifying order and fills the slots around them. If two drivers would
    land on the same slot, or a target lands past the back, it cascades to the
    next free slot. Drivers with no qualifying time (NaN) sit at the back.
    Returns a float array (never NaN) suitable as the blend's grid prior.
    """
    n = len(codes)
    order = sorted(  # qualifying order, best first; NaN quali to the back
        range(n), key=lambda i: (np.isnan(quali[i]), quali[i] if not np.isnan(quali[i]) else 0.0, i)
    )
    qrank = {i: r + 1 for r, i in enumerate(order)}  # dense 1..n qualifying rank
    penalized = [i for i in order if penalties.get(codes[i], 0) > 0]
    non_pen = [i for i in order if penalties.get(codes[i], 0) == 0]

    slot: list[int | None] = [None] * n
    for i in sorted(penalized, key=lambda i: qrank[i] + penalties[codes[i]]):
        t = min(qrank[i] + penalties[codes[i]], n) - 1  # 0-indexed target
        while t < n and slot[t] is not None:
            t += 1
        if t >= n:  # overflow past the back: take the last free slot
            t = max(j for j in range(n) if slot[j] is None)
        slot[t] = i
    fill = iter(non_pen)
    for s in range(n):
        if slot[s] is None:
            slot[s] = next(fill)

    start = np.empty(n, dtype=float)
    for pos0, i in enumerate(slot):
        start[i] = pos0 + 1
    return start


def _confidence(
    win: np.ndarray,
    rows: pd.DataFrame,
    forecast: dict | None,
    quali_available: bool,
    penalties: dict[str, int] | None = None,
) -> dict:
    """Race-level prediction confidence: the calibrated probability of the top
    pick, plus human-readable reasons pushing it up or down. `win` is the
    calibrated win-probability array aligned with `rows`."""
    order = np.argsort(-win)
    p1, p2 = float(win[order[0]]), float(win[order[1]])
    gap = p1 - p2
    within5 = int((win >= p1 - 0.05).sum())

    sc_rate = rows["track_sc_rate"].dropna()
    sc_rate = float(sc_rate.iloc[0]) if len(sc_rate) else None
    overtakes = rows["track_overtakes_avg"].dropna()
    overtakes = float(overtakes.iloc[0]) if len(overtakes) else None
    rain = forecast.get("rain_prob") if forecast else None
    pick = rows.iloc[order[0]]

    reasons: list[dict] = []
    if gap >= 0.15:
        reasons.append({"kind": "good", "text": f"Clear favourite — {gap * 100:.0f} points ahead of the next driver"})
    if quali_available and not pd.isna(pick["quali_pos"]) and int(pick["quali_pos"]) == 1:
        reasons.append({"kind": "good", "text": "Pick starts from pole position"})
    if overtakes is not None and overtakes < 25 and quali_available:
        reasons.append({"kind": "good", "text": "Low-overtaking circuit — grid order tends to hold"})
    if rain is not None and rain < 0.2:
        reasons.append({"kind": "good", "text": "Dry race expected"})
    if quali_available:
        reasons.append({"kind": "good", "text": "Qualifying results included"})
    if penalties:
        drivers = ", ".join(f"{c} +{p}" for c, p in penalties.items())
        reasons.append({"kind": "warn", "text": f"Grid penalties applied ({drivers})"})

    if rain is not None and rain >= 0.3:
        reasons.append({"kind": "warn", "text": f"{rain * 100:.0f}% rain probability — wet races scramble the field"})
    if sc_rate is not None and sc_rate >= 0.5:
        reasons.append({"kind": "warn", "text": f"Safety car in {sc_rate * 100:.0f}% of past races here"})
    if within5 >= 3:
        reasons.append({"kind": "warn", "text": f"Top {within5} drivers within 5 points of win probability"})
    if not quali_available:
        reasons.append({"kind": "warn", "text": "Pre-qualifying forecast — grid not yet known"})

    score = round(p1 * 100)
    level = "high" if score >= 55 else "medium" if score >= 35 else "low"
    return {"score": score, "level": level, "reasons": reasons}


def predict_next_race() -> dict | None:
    event = store.next_event()
    if event is None:
        return None

    bundle = load_model()
    model, sigma = bundle["model"], bundle["sigma"]

    results, races = store.results(), store.races()
    quali = store.completed_quali(event)
    # normalize the live schedule label so renamed circuits (e.g. Monte Carlo
    # vs Monaco) still find their full track history
    location = config.normalize_location(event["Location"])
    rows = build_prediction_rows(results, races, location, quali)

    # live rain forecast closes the is_wet train/serve gap (to the extent the
    # forecast is right); on any failure it stays NaN as before
    forecast = weather.race_rain_forecast(location, event.get("Session5DateUtc"))
    if forecast is not None and forecast.get("is_wet") is not None:
        rows["is_wet"] = float(forecast["is_wet"])

    X = rows[FEATURE_COLS]
    # Blend the ranker with the qualifying grid when available; before
    # qualifying the grid is all-NaN and the model rank is used alone. In the
    # wet the grid prior is unreliable, so lean on the model via the rain
    # probability (rain_prob preferred; fall back to the binary is_wet flag).
    p_wet = (
        forecast["rain_prob"]
        if forecast and forecast.get("rain_prob") is not None
        else (forecast["is_wet"] if forecast else None)
    )

    # Manual grid-place penalties (data-only, event-scoped: see store.grid_penalties).
    # They shift ONLY the starting-grid prior fed to the blend, never the
    # qualifying-pace features in X -- a penalty changes where a driver starts,
    # not the lap they set.
    penalties = store.grid_penalties(event)
    start_pos = _apply_grid_penalties(
        rows["driver_code"].tolist(),
        rows["quali_pos"].to_numpy(dtype=float),
        penalties,
    )
    scores = predict_scores(model, X, start_pos, p_wet=p_wet)

    dnf = (
        0.5 * rows["driver_dnf_rate"].fillna(0.1) + 0.5 * rows["team_dnf_rate"].fillna(0.1)
    ).clip(0.02, 0.35).to_numpy()

    sim = simulate_race(scores, dnf, sigma)
    # the raw simulation is badly underconfident at the top (see ml/calibration)
    win = calibrate_win_probs(sim["win_prob"])
    expl = _explanations(model, X)

    order = rank_order(sim["mean_pos"], win)
    drivers = []
    for rank, i in enumerate(order, start=1):
        r = rows.iloc[i]
        drivers.append(
            {
                "predicted_rank": rank,
                "driver_code": r["driver_code"],
                "driver_name": r["driver_name"],
                "driver_number": r["driver_number"],
                "team": r["team"],
                "team_color": r["team_color"],
                "quali_pos": r["quali_pos"],
                "start_pos": float(start_pos[i]) if not pd.isna(r["quali_pos"]) else None,
                "grid_penalty": int(penalties.get(r["driver_code"], 0)),
                "score": round(float(scores[i]), 2),
                "win_prob": round(float(win[i]), 4),
                "podium_prob": round(float(sim["podium_prob"][i]), 4),
                "points_prob": round(float(sim["points_prob"][i]), 4),
                "dnf_prob": round(float(sim["dnf_prob"][i]), 4),
                "mean_pos": round(float(sim["mean_pos"][i]), 2),
                "p10_pos": float(sim["p10_pos"][i]),
                "p90_pos": float(sim["p90_pos"][i]),
                "why": expl[i],
                "form_finish_5": r["form_finish_5"],
                "track_finish_avg": r["track_finish_avg"],
            }
        )

    return store.clean(
        {
            "event": {
                "season": config.CURRENT_SEASON,
                "round": int(event["RoundNumber"]),
                "name": event["EventName"],
                "official_name": event["OfficialEventName"],
                "location": location,
                "country": config.normalize_country(location, event["Country"]),
                "race_start_utc": store.iso_utc(event["Session5DateUtc"]),
            },
            "quali_available": quali is not None,
            "grid_penalties": penalties,  # {} when none apply
            "weather_forecast": forecast,  # null when unavailable
            "sigma": sigma,
            "confidence": _confidence(win, rows, forecast, quali is not None, penalties),
            "drivers": drivers,
        }
    )
