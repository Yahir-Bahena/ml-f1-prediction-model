"""Aggregated driver / track / standings stats from the ingested tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from app.services import store


def _with_total_points(df: pd.DataFrame) -> pd.DataFrame:
    """Add total_points = GP points + sprint points (championship points)."""
    sprint = df["sprint_points"].fillna(0.0) if "sprint_points" in df.columns else 0.0
    return df.assign(total_points=df["points"] + sprint)


def _current_lineup() -> pd.DataFrame:
    """Latest row per driver from the most recent ingested race."""
    res = store.results().sort_values(["season", "round"])
    last = res[res["season"] == res["season"].max()]
    last = last[last["round"] == last["round"].max()]
    return last


def driver_list() -> list[dict]:
    res = store.results()
    season = res["season"].max()
    cur = _with_total_points(res[res["season"] == season])
    lineup = _current_lineup()

    out = []
    for _, drv in lineup.iterrows():
        code = drv["driver_code"]
        d = cur[cur["driver_code"] == code]
        out.append(
            {
                "driver_code": code,
                "driver_name": drv["driver_name"],
                "driver_number": drv["driver_number"],
                "team": drv["team"],
                "team_color": drv["team_color"],
                "season_points": float(d["total_points"].sum()),
                "wins": int((d["finish_pos"] == 1).sum()),
                "podiums": int((d["finish_pos"] <= 3).sum()),
                "avg_finish": float(d["finish_pos"].mean()) if len(d) else None,
                "avg_quali": float(d["quali_pos"].mean()) if len(d) else None,
            }
        )
    out.sort(key=lambda x: -x["season_points"])
    return store.clean(out)


def driver_stats(code: str) -> dict | None:
    res = store.results().sort_values(["season", "round"])
    d = _with_total_points(res[res["driver_code"] == code])
    if d.empty:
        return None
    latest = d.iloc[-1]
    season = res["season"].max()
    d_cur = d[d["season"] == season]

    wet, dry = d[d["wet_race"]], d[~d["wet_race"]]
    gains = d["grid_pos"] - d["finish_pos"]

    recent = d.tail(12)[
        ["season", "round", "event_name", "location", "grid_pos", "quali_pos",
         "finish_pos", "points", "dnf", "wet_race"]
    ].to_dict("records")

    track_history = (
        d.groupby("location")
        .agg(
            races=("finish_pos", "size"),
            avg_finish=("finish_pos", "mean"),
            best_finish=("finish_pos", "min"),
            avg_gain=("lap1_gain", "mean"),
        )
        .reset_index()
        .sort_values("avg_finish")
        .to_dict("records")
    )

    return store.clean(
        {
            "driver_code": code,
            "driver_name": latest["driver_name"],
            "driver_number": latest["driver_number"],
            "team": latest["team"],
            "team_color": latest["team_color"],
            "career": {
                "races": int(len(d)),
                "wins": int((d["finish_pos"] == 1).sum()),
                "podiums": int((d["finish_pos"] <= 3).sum()),
                "poles": int((d["quali_pos"] == 1).sum()),
                "dnf_rate": float(d["dnf"].mean()),
                "avg_finish": float(d["finish_pos"].mean()),
                # cleaned metric: genuine on-track passes only
                "avg_overtakes": float(d["overtakes_clean"].mean()),
            },
            "season": {
                "points": float(d_cur["total_points"].sum()),
                "wins": int((d_cur["finish_pos"] == 1).sum()),
                "podiums": int((d_cur["finish_pos"] <= 3).sum()),
                "avg_finish": float(d_cur["finish_pos"].mean()) if len(d_cur) else None,
                "avg_quali": float(d_cur["quali_pos"].mean()) if len(d_cur) else None,
            },
            # decomposed racecraft/strategy (career averages; additive schema)
            "racecraft": {
                "avg_ontrack_net": float(d["ontrack_net"].mean()),
                "avg_pit_cycle_gain": float(d["pit_cycle_gain"].mean()),
                "avg_pit_stop_delta_ms": float(d["pit_stop_delta_ms"].mean()),
                "avg_stint1_pace_delta_ms": float(d["stint1_pace_delta_ms"].mean()),
                "undercut_attempts": float(d["undercut_attempts"].sum()),
                "undercut_successes": float(d["undercut_successes"].sum()),
            },
            "consistency_ms": float(d["lap_std_ms"].tail(10).mean()),
            "avg_positions_gained": float(gains.mean()),
            "wet_avg_finish": float(wet["finish_pos"].mean()) if len(wet) >= 2 else None,
            "dry_avg_finish": float(dry["finish_pos"].mean()) if len(dry) >= 2 else None,
            "recent_results": recent,
            "track_history": track_history,
        }
    )


def track_list() -> list[dict]:
    races = store.races()
    grouped = (
        races.groupby("location")
        .agg(
            country=("country", "last"),
            event_name=("event_name", "last"),
            races=("round", "size"),
            avg_overtakes=("total_overtakes_clean", "mean"),  # cleaned metric
            sc_rate=("sc_laps", lambda s: float((s > 0).mean())),
            wet_rate=("wet_race", "mean"),
            avg_dnf=("n_dnf", "mean"),
            last_season=("season", "max"),
        )
        .reset_index()
        .sort_values("event_name")
    )
    return store.clean(grouped.to_dict("records"))


def track_stats(location: str) -> dict | None:
    races, res = store.races(), store.results()
    r = races[races["location"] == location]
    if r.empty:
        return None
    t = res[res["location"] == location]

    winners = (
        t[t["finish_pos"] == 1][["season", "driver_name", "team", "team_color", "grid_pos"]]
        .sort_values("season", ascending=False)
        .to_dict("records")
    )
    # how much does grid position stick at this track?
    valid = t.dropna(subset=["grid_pos", "finish_pos"])
    grid_hold = float(np.corrcoef(valid["grid_pos"], valid["finish_pos"])[0, 1]) if len(valid) > 10 else None

    return store.clean(
        {
            "location": location,
            "country": r.iloc[-1]["country"],
            "event_name": r.iloc[-1]["event_name"],
            "races": int(len(r)),
            "avg_overtakes": float(r["total_overtakes_clean"].mean()),  # cleaned metric
            "sc_rate": float((r["sc_laps"] > 0).mean()),
            "vsc_rate": float((r["vsc_laps"] > 0).mean()),
            "wet_rate": float(r["wet_race"].mean()),
            "avg_dnf": float(r["n_dnf"].mean()),
            "avg_track_temp": float(r["track_temp"].mean()),
            "grid_finish_correlation": grid_hold,
            "past_winners": winners,
            # editions keep the total_overtakes field name (frontend contract)
            # but serve the cleaned values
            "editions": r.sort_values("season", ascending=False)[
                ["season", "round", "wet_race", "sc_laps", "total_overtakes_clean", "n_dnf"]
            ]
            .rename(columns={"total_overtakes_clean": "total_overtakes"})
            .to_dict("records"),
        }
    )


def _countback_key(finishes: pd.Series) -> tuple:
    """FIA tie-break: more 1sts, then more 2nds, ... (negated for ascending sort)."""
    counts = finishes.dropna().astype(int).value_counts()
    return tuple(-int(counts.get(p, 0)) for p in range(1, 26))


def standings() -> dict:
    res = store.results()
    season = res["season"].max()
    cur = _with_total_points(res[res["season"] == season])

    drivers = (
        cur.groupby(["driver_code", "driver_name", "team", "team_color"])
        .agg(points=("total_points", "sum"), wins=("finish_pos", lambda s: int((s == 1).sum())),
             podiums=("finish_pos", lambda s: int((s <= 3).sum())))
        .reset_index()
        .to_dict("records")
    )
    cb = cur.groupby("driver_code")["finish_pos"].apply(_countback_key).to_dict()
    drivers.sort(key=lambda r: (-r["points"], cb[r["driver_code"]]))

    teams = (
        cur.groupby(["team", "team_color"])
        .agg(points=("total_points", "sum"), wins=("finish_pos", lambda s: int((s == 1).sum())))
        .reset_index()
        .to_dict("records")
    )
    tcb = cur.groupby("team")["finish_pos"].apply(_countback_key).to_dict()
    teams.sort(key=lambda r: (-r["points"], tcb[r["team"]]))
    return store.clean({"season": int(season), "drivers": drivers, "teams": teams})
