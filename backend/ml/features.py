"""Feature engineering for the race predictor.

Every feature for a given race is computed ONLY from races that happened before
it (plus that race's own qualifying, which takes place before the race), so the
same code path is valid for both training rows and next-race prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    # qualifying (known pre-race; NaN if quali hasn't happened yet)
    "quali_pos",
    "quali_gap_to_pole_ms",
    "quali_gap_to_teammate",
    # qualifying surprise: this weekend's quali vs the driver's recent quali form
    # (form_quali_5 - quali_pos). Positive = out-qualified recent form, a signal
    # of real weekend pace the form features cannot see. Recovers "wrong pole-
    # override" misses (QUALI-SURPRISE experiment). Derived post-pass in
    # build_training_dataset / build_prediction_rows.
    "quali_surprise",
    # driver form
    "form_finish_3",
    "form_finish_5",
    "form_points_5",
    "form_gain_5",
    "form_quali_5",
    "season_points",
    "season_rank",
    # pace / consistency / tyre life (rolling)
    "pace_delta_5",
    "consistency_5",
    "deg_slope_5",
    # strategy / racecraft (rolling; from the racecraft backfill)
    "pit_gain_5",
    "pit_stop_delta_5",
    "undercut_rate_10",
    "stint1_pace_5",
    # overtaking
    "driver_overtakes_10",
    "lap1_gain_5",
    "track_overtakes_avg",
    "track_sc_rate",
    # track history
    "track_finish_avg",
    "track_gain_avg",
    "track_races",
    "team_track_finish_avg",
    # team strength
    "team_form_finish_5",
    "team_season_points",
    # reliability
    "driver_dnf_rate",
    "team_dnf_rate",
    # conditions
    "wet_delta",
    "is_wet",
    # experience
    "n_races",
]

# Source columns for the overtaking features. The "_clean" variants count only
# genuine on-track passes (see scripts/ingest.py:_clean_overtakes); the raw
# columns also include pit-cycle/retirement/SC position changes. A/B-tested in
# the walk-forward backtest: clean beat raw on winner_hit_rate and position_mae.
OVERTAKES_COL = "overtakes_clean"
TRACK_OVERTAKES_COL = "total_overtakes_clean"


def _mean(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean()) if len(s) else np.nan


def _ewma(s: pd.Series, halflife: float) -> float:
    """Exponentially-weighted mean weighting the most recent prior race highest.

    `s` must already be in chronological order (oldest -> newest). NaN is dropped,
    and the result is NaN when no prior observation exists (insufficient history).
    half-life is expressed in observations (races), so a 3-race half-life halves
    the weight of a result three races back relative to the latest one.
    """
    s = s.dropna()
    if not len(s):
        return np.nan
    return float(s.ewm(halflife=halflife, adjust=True).mean().iloc[-1])


# Retirement causes (FastF1 `status`) that are DRIVER-CAUSED — a crash is genuine
# negative form signal, so these DNFs are KEPT in the noise-robust finish EWMA.
# Every other DNF cause (mechanical failures AND the generic "Retired"/"Withdrew"/
# "Did not start" buckets) is treated as non-representative and DROPPED. NOTE: the
# dominant DNF bucket is the generic "Retired" (191/289), which does not attribute
# fault, so most real driver-caused incidents cannot be recovered as crashes here.
CRASH_STATUSES = frozenset({"Accident", "Collision", "Collision damage", "Spun off"})


def _ewma_weighted(values: np.ndarray, extra_w: np.ndarray, halflife: float) -> float:
    """Exponentially-weighted mean with an extra per-observation weight.

    `values`/`extra_w` are chronological (oldest -> newest) and NaN-free. Uses the
    same adjust=True convention as _ewma (newest observation gets base weight 1),
    but multiplies each observation's base weight by `extra_w[i]` (e.g. 0.5 to
    downweight a wet race). Returns NaN when empty or all weights vanish.
    """
    n = len(values)
    if n == 0:
        return np.nan
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    base = (1.0 - alpha) ** (n - 1 - np.arange(n, dtype=float))
    w = base * np.asarray(extra_w, dtype=float)
    sw = w.sum()
    if sw == 0:
        return np.nan
    return float((w * np.asarray(values, dtype=float)).sum() / sw)


def _ewma_downweight_wet(
    values: pd.Series, wet: pd.Series, halflife: float, wet_w: float = 0.5
) -> float:
    """EWMA (half-life in observations) over `values`, downweighting wet races.

    `values` and `wet` must share an index and be chronological. NaN values are
    dropped (with their wet flag); each retained observation's EWMA weight is
    multiplied by `wet_w` when its race was wet. NaN when no clean observation.
    """
    v = pd.to_numeric(values, errors="coerce")
    m = v.notna()
    if not m.any():
        return np.nan
    vv = v[m].to_numpy(dtype=float)
    wetv = wet[m].astype(bool).to_numpy()
    extra = np.where(wetv, wet_w, 1.0)
    return _ewma_weighted(vv, extra, halflife)


def _slope(s: pd.Series, n: int = 5) -> float:
    """OLS slope of the last `n` prior observations against race order (0..k-1).

    `s` must be chronological. NaN is dropped first; returns NaN with fewer than
    3 points (a line through <3 points is noise, not a trend). For finish/pace a
    negative slope means improving/getting faster over the recent window.
    """
    y = s.dropna().tail(n).to_numpy(dtype=float)
    if len(y) < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _with_championship_points(results: pd.DataFrame) -> pd.DataFrame:
    """Fold sprint points into the points column so points-based features
    (season_points, season_rank, form_points_5, team_season_points) reflect
    real championship points.

    No leakage: features for race N only ever see weekends strictly before N
    (hist), which are fully complete. The current weekend's sprint is used
    neither in training nor in the pre-race prediction path.
    """
    if "sprint_points" not in results.columns:
        return results
    return results.assign(points=results["points"] + results["sprint_points"].fillna(0.0))


def driver_features(
    hist: pd.DataFrame,
    races_hist: pd.DataFrame,
    code: str,
    team: str,
    location: str,
) -> dict:
    """Features for one driver ahead of a race at `location`, from history only."""
    d = hist[hist["driver_code"] == code].sort_values(["season", "round"])
    t = hist[hist["team"] == team].sort_values(["season", "round"])
    season = hist["season"].max() if len(hist) else np.nan
    d_season = d[d["season"] == season]
    cur = hist[hist["season"] == season]

    gains = d["grid_pos"] - d["finish_pos"]

    # season standings so far
    season_points = float(d_season["points"].sum()) if len(d_season) else np.nan
    season_rank = np.nan
    if len(cur):
        standings = cur.groupby("driver_code")["points"].sum().sort_values(ascending=False)
        if code in standings.index:
            season_rank = float(standings.index.get_loc(code) + 1)

    d_track = d[d["location"] == location]
    t_track = t[t["location"] == location]
    r_track = races_hist[races_hist["location"] == location] if len(races_hist) else pd.DataFrame()

    # per-race team aggregates (chronological): one value per prior race = mean
    # over the team's cars that weekend. Grouping by race first (rather than
    # ewma-ing the ~2-rows-per-race stream at half-life 6) gives every race equal
    # weight even when a team fielded a single car (reserve driver) that weekend,
    # and makes the half-life a clean "3 races" for both the ewma and the slope.
    if len(t):
        t_race_finish = t.groupby(["season", "round"], sort=True)["finish_pos"].mean()
        t_race_pace = t.groupby(["season", "round"], sort=True)["pace_delta_ms"].mean()
    else:
        t_race_finish = pd.Series(dtype=float)
        t_race_pace = pd.Series(dtype=float)

    wet = d[d["wet_race"] == True]  # noqa: E712
    dry = d[d["wet_race"] == False]  # noqa: E712
    wet_delta = (
        _mean(wet["finish_pos"]) - _mean(dry["finish_pos"])
        if len(wet) >= 2 and len(dry) >= 2
        else np.nan
    )

    # ---- noise-robust ("adjusted") momentum ------------------------------
    # Strips the events that poisoned the plain EWMA (E3 autopsy): mechanical /
    # generic DNFs are dropped, driver-caused crashes are kept (crashing is form
    # signal), and wet races are downweighted 0.5. Pace features drop ALL DNFs
    # (partial-stint pace from an aborted race is noise regardless of cause).
    d_dnf = d["dnf"].astype(bool) if len(d) else pd.Series(dtype=bool)
    d_status = d["status"] if "status" in d.columns else pd.Series(index=d.index, dtype=object)
    d_is_crash = d_dnf & d_status.isin(CRASH_STATUSES)
    d_keep_finish = d[(~d_dnf) | d_is_crash]   # classified finishers + crashes
    d_keep_pace = d[~d_dnf]                     # all DNFs excluded for pace

    if len(t):
        t_nodnf = t[~t["dnf"].astype(bool)]
        if len(t_nodnf):
            t_grp = t_nodnf.groupby(["season", "round"], sort=True)
            t_race_pace_clean = t_grp["pace_delta_ms"].mean()
            t_race_wet_clean = t_grp["wet_race"].max()  # race-level flag
        else:
            t_race_pace_clean = pd.Series(dtype=float)
            t_race_wet_clean = pd.Series(dtype=bool)
    else:
        t_race_pace_clean = pd.Series(dtype=float)
        t_race_wet_clean = pd.Series(dtype=bool)

    return {
        "form_finish_3": _mean(d["finish_pos"].tail(3)),
        "form_finish_5": _mean(d["finish_pos"].tail(5)),
        "form_points_5": _mean(d["points"].tail(5)),
        "form_gain_5": _mean(gains.tail(5)),
        "form_quali_5": _mean(d["quali_pos"].tail(5)),
        "season_points": season_points,
        "season_rank": season_rank,
        "pace_delta_5": _mean(d["pace_delta_ms"].tail(5)),
        "consistency_5": _mean(d["lap_std_ms"].tail(5)),
        "deg_slope_5": _mean(d["deg_slope_s_per_lap"].tail(5)),
        "driver_overtakes_10": _mean(d[OVERTAKES_COL].tail(10)),
        "lap1_gain_5": _mean(d["lap1_gain"].tail(5)),
        "track_overtakes_avg": _mean(r_track[TRACK_OVERTAKES_COL]) if len(r_track) else np.nan,
        "track_sc_rate": _mean((r_track["sc_laps"] > 0).astype(float)) if len(r_track) else np.nan,
        "track_finish_avg": _mean(d_track["finish_pos"]),
        "track_gain_avg": _mean(d_track["grid_pos"] - d_track["finish_pos"]),
        "track_races": float(len(d_track)),
        "team_track_finish_avg": _mean(t_track["finish_pos"]),
        "team_form_finish_5": _mean(t["finish_pos"].tail(10)),  # 10 rows = ~5 races x 2 cars
        "team_season_points": float(t[t["season"] == season]["points"].sum()) if len(t) else np.nan,
        "driver_dnf_rate": _mean(d["dnf"].tail(20).astype(float)),
        "team_dnf_rate": _mean(t["dnf"].tail(40).astype(float)),
        "wet_delta": wet_delta,
        "n_races": float(len(d)),
        # decomposed racecraft/strategy (rolling; columns exist since the
        # racecraft backfill - see scripts/ingest.py:_racecraft_metrics)
        "pit_gain_5": _mean(d["pit_cycle_gain"].tail(5)) if "pit_cycle_gain" in d else np.nan,
        "pit_stop_delta_5": _mean(d["pit_stop_delta_ms"].tail(5)) if "pit_stop_delta_ms" in d else np.nan,
        "undercut_rate_10": _undercut_rate(d),
        "stint1_pace_5": _mean(d["stint1_pace_delta_ms"].tail(5)) if "stint1_pace_delta_ms" in d else np.nan,
        # momentum / EWMA (half-life 3 races; most-recent-weighted) + recent
        # trend (OLS slope over the last 5 prior races). React faster to
        # mid-season competitive shifts than cumulative / flat-window features.
        "ewma_finish_3": _ewma(d["finish_pos"], 3),
        "ewma_pace_3": _ewma(d["pace_delta_ms"], 3),
        "team_ewma_finish_3": _ewma(t_race_finish, 3),
        "team_ewma_pace_3": _ewma(t_race_pace, 3),
        "pace_trend_5": _slope(d["pace_delta_ms"], 5),
        "finish_trend_5": _slope(d["finish_pos"], 5),
        "team_pace_trend_5": _slope(t_race_pace, 5),
        # noise-robust momentum (unused by production FEATURE_COLS; experiment only)
        "adj_ewma_finish_3": _ewma_downweight_wet(
            d_keep_finish["finish_pos"], d_keep_finish["wet_race"], 3
        ),
        "adj_ewma_pace_3": _ewma_downweight_wet(
            d_keep_pace["pace_delta_ms"], d_keep_pace["wet_race"], 3
        ),
        "adj_team_ewma_pace_3": _ewma_downweight_wet(
            t_race_pace_clean, t_race_wet_clean, 3
        ),
        # slope over the last 5 classified-for-finish results (mechanical DNFs
        # dropped, crashes kept); no wet downweighting on the trend (kept simple).
        "adj_finish_trend_5": _slope(d_keep_finish["finish_pos"], 5),
    }


def _undercut_rate(d: pd.DataFrame) -> float:
    """Undercut success rate over the last 10 races; NaN below 2 attempts
    (a rate from a single attempt is noise, not skill)."""
    if "undercut_attempts" not in d:
        return np.nan
    a = d["undercut_attempts"].tail(10).sum()
    s = d["undercut_successes"].tail(10).sum()
    return float(s / a) if a >= 2 else np.nan


def add_teammate_quali_gap(rows: pd.DataFrame) -> pd.DataFrame:
    """quali_gap_to_teammate = own quali best - teammate quali best (ms)."""
    rows = rows.copy()
    rows["quali_gap_to_teammate"] = np.nan
    for _, idx in rows.groupby("team").groups.items():
        grp = rows.loc[idx]
        if len(grp) == 2:
            a, b = grp.index
            va, vb = grp.loc[a, "quali_best_ms"], grp.loc[b, "quali_best_ms"]
            if not (pd.isna(va) or pd.isna(vb)):
                rows.loc[a, "quali_gap_to_teammate"] = va - vb
                rows.loc[b, "quali_gap_to_teammate"] = vb - va
    return rows


def build_training_dataset(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """One row per driver per race with features from prior races + target finish."""
    races = races.sort_values(["season", "round"]).reset_index(drop=True)
    results = _with_championship_points(results).sort_values(["season", "round"]).reset_index(drop=True)

    out = []
    for i, race in races.iterrows():
        season, rnd = race["season"], race["round"]
        prior_races = races.iloc[:i]
        hist = results.merge(
            prior_races[["season", "round"]], on=["season", "round"], how="inner"
        )
        rows = results[(results["season"] == season) & (results["round"] == rnd)].copy()
        rows = add_teammate_quali_gap(rows)

        n_starters = len(rows)
        for _, r in rows.iterrows():
            feats = driver_features(hist, prior_races, r["driver_code"], r["team"], r["location"])
            target = r["finish_pos"] if not pd.isna(r["finish_pos"]) else float(n_starters)
            out.append(
                {
                    "season": season,
                    "round": rnd,
                    "event_name": r["event_name"],
                    "location": r["location"],
                    "driver_code": r["driver_code"],
                    "driver_name": r["driver_name"],
                    "team": r["team"],
                    "quali_pos": r["quali_pos"],
                    "quali_gap_to_pole_ms": r["quali_gap_to_pole_ms"],
                    "quali_gap_to_teammate": r["quali_gap_to_teammate"],
                    "is_wet": float(r["wet_race"]),
                    **feats,
                    "target_pos": float(target),
                    "grid_pos": r["grid_pos"],
                    "dnf": bool(r["dnf"]),
                }
            )
    out_df = pd.DataFrame(out)
    # quali_surprise post-pass: form_quali_5 (strictly-prior quali form, arrives
    # via **feats) minus this weekend's quali_pos (pre-race). NaN if either NaN.
    out_df["quali_surprise"] = out_df["form_quali_5"] - out_df["quali_pos"]
    return out_df


def build_prediction_rows(
    results: pd.DataFrame,
    races: pd.DataFrame,
    location: str,
    quali: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature rows for the upcoming race at `location`, using full history.

    The driver entry list is taken from the most recent race. `quali` may carry
    columns driver_code / quali_pos / quali_best_ms / quali_gap_to_pole_ms when
    qualifying has already happened; otherwise those features stay NaN.
    """
    races = races.sort_values(["season", "round"]).reset_index(drop=True)
    results = _with_championship_points(results)
    latest = results.sort_values(["season", "round"]).groupby("driver_code").tail(1)
    last_season, last_round = races.iloc[-1]["season"], races.iloc[-1]["round"]
    entry = latest[(latest["season"] == last_season) & (latest["round"] == last_round)][
        ["driver_code", "driver_name", "driver_number", "team", "team_color"]
    ].copy()

    if quali is not None and len(quali):
        entry = entry.merge(quali, on="driver_code", how="left")
    for col in ("quali_pos", "quali_best_ms", "quali_gap_to_pole_ms"):
        if col not in entry.columns:
            entry[col] = np.nan
    entry = add_teammate_quali_gap(entry)

    rows = []
    for _, r in entry.iterrows():
        feats = driver_features(results, races, r["driver_code"], r["team"], location)
        rows.append(
            {
                "driver_code": r["driver_code"],
                "driver_name": r["driver_name"],
                "driver_number": r["driver_number"],
                "team": r["team"],
                "team_color": r["team_color"],
                "quali_pos": r["quali_pos"],
                "quali_gap_to_pole_ms": r["quali_gap_to_pole_ms"],
                "quali_gap_to_teammate": r["quali_gap_to_teammate"],
                "is_wet": np.nan,  # unknown before the race; LightGBM handles NaN
                **feats,
            }
        )
    rows_df = pd.DataFrame(rows)
    # quali_surprise post-pass (same derivation as build_training_dataset):
    # form_quali_5 (via **feats) minus quali_pos. Present in BOTH paths so the
    # feature exists identically at train and predict time.
    rows_df["quali_surprise"] = rows_df["form_quali_5"] - rows_df["quali_pos"]
    return rows_df
