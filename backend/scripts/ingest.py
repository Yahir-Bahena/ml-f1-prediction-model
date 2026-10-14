"""Ingest 2022-2026 F1 race + qualifying data via fastf1 into compact parquet tables.

Resumable: races already present in the output parquet are skipped, and fastf1's
on-disk cache makes re-downloads cheap. Run from the backend/ directory:

    python -m scripts.ingest            # all seasons in config.SEASONS
    python -m scripts.ingest 2026       # a single season
    python -m scripts.ingest normalize  # only canonicalize track labels (no network)
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config

RATE_LIMIT_SLEEP_S = 600  # fastf1 external APIs allow 500 calls/h; wait and resume


def _with_rate_limit_retry(fn, label: str):
    """Run fn(); on fastf1 rate-limit errors sleep and retry until it succeeds."""
    from fastf1 import RateLimitExceededError

    while True:
        try:
            return fn()
        except RateLimitExceededError:
            print(f"  .. rate limited at {label}; sleeping {RATE_LIMIT_SLEEP_S}s", flush=True)
            time.sleep(RATE_LIMIT_SLEEP_S)


def _td_ms(value) -> float:
    """Timedelta -> milliseconds (NaN-safe)."""
    if value is None or pd.isna(value):
        return np.nan
    return value.total_seconds() * 1000.0


def _quali_features(quali) -> pd.DataFrame:
    """Per-driver qualifying position and gap to pole (ms)."""
    res = quali.results
    if res is None or len(res) == 0:
        return pd.DataFrame(columns=["driver_code", "quali_pos", "quali_best_ms", "quali_gap_to_pole_ms"])

    rows = []
    for _, r in res.iterrows():
        best = min(
            (t for t in (r.get("Q1"), r.get("Q2"), r.get("Q3")) if t is not None and not pd.isna(t)),
            default=pd.NaT,
        )
        rows.append(
            {
                "driver_code": r["Abbreviation"],
                "quali_pos": float(r["Position"]) if not pd.isna(r["Position"]) else np.nan,
                "quali_best_ms": _td_ms(best),
            }
        )
    out = pd.DataFrame(rows)
    pole = out["quali_best_ms"].min()
    out["quali_gap_to_pole_ms"] = out["quali_best_ms"] - pole
    return out


SPRINT_COLS = ["sprint_finish_pos", "sprint_grid_pos", "sprint_points"]

# track statuses under which passing is banned (SC, red flag, VSC, VSC ending)
_BLOCKED_STATUS = ("4", "5", "6", "7")


def _clean_overtakes(laps: pd.DataFrame) -> dict[str, float]:
    """Genuine on-track passes per driver, from the per-lap running order.

    Algorithm: for every consecutive lap pair (L-1 -> L), reconstruct the
    running order from the Position column and find *swapped pairs*: driver A
    "passes" driver B when A was behind B at L-1 and ahead at L. Pair-level
    attribution lets us exclude the polluted counterparts cleanly. A swap is
    counted only if:
      - both cars are present on both laps (retiring cars never register a
        pass for the cars that inherit their position);
      - neither car has a PitInTime or PitOutTime on lap L-1 or L (a pit stop
        spans two timing laps, so this drops pit-cycle position changes:
        undercuts, cars pitting ahead, out-lap re-shuffles);
      - neither lap is run under SC / VSC / red flag (passing is banned, so
        recorded swaps there are pit or lapped-car artifacts);
      - L >= 3, matching the raw metric's window (lap-1 chaos is captured by
        the separate lap1_gain metric).
    Relative-order comparison makes the count robust to absolute position
    shifts caused by third cars pitting or retiring between the pair.
    """
    if laps is None or len(laps) == 0:
        return {}
    pos = laps.pivot_table(index="LapNumber", columns="Driver", values="Position", aggfunc="first")
    pit = laps[laps["PitInTime"].notna() | laps["PitOutTime"].notna()]
    pit_laps = set(zip(pit["Driver"], pit["LapNumber"]))
    lap_status = laps.groupby("LapNumber")["TrackStatus"].apply(
        lambda s: "".join(str(v) for v in s.dropna().unique())
    )
    counts = {d: 0.0 for d in pos.columns}
    lapnums = sorted(pos.index)
    for i in range(1, len(lapnums)):
        lap0, lap1 = lapnums[i - 1], lapnums[i]
        if lap1 < 3:
            continue
        status = str(lap_status.get(lap0, "")) + str(lap_status.get(lap1, ""))
        if any(b in status for b in _BLOCKED_STATUS):
            continue
        prev, curr = pos.loc[lap0], pos.loc[lap1]
        ok = prev.notna() & curr.notna()
        drivers = [
            d for d in pos.columns
            if ok[d] and (d, lap0) not in pit_laps and (d, lap1) not in pit_laps
        ]
        p = prev[drivers].to_numpy(dtype=float)
        c = curr[drivers].to_numpy(dtype=float)
        swap = (p[:, None] > p[None, :]) & (c[:, None] < c[None, :])
        for ai, d in enumerate(drivers):
            counts[d] += float(swap[ai].sum())
    return counts


def _is_sprint_event(event) -> bool:
    """True for every sprint weekend format (2022 'sprint', 2023
    'sprint_shootout', 2024+ 'sprint_qualifying')."""
    return "sprint" in str(event.get("EventFormat", "")).lower()


def _sprint_results(season: int, rnd: int) -> pd.DataFrame:
    """Per-driver sprint-race finish and championship points (session 'S')."""
    import fastf1

    sprint = fastf1.get_session(season, rnd, "S")
    sprint.load(laps=False, telemetry=False, weather=False, messages=False)
    res = sprint.results
    if res is None or len(res) == 0:
        return pd.DataFrame(columns=["driver_code", *SPRINT_COLS])
    rows = []
    for _, r in res.iterrows():
        rows.append(
            {
                "driver_code": r["Abbreviation"],
                "sprint_finish_pos": float(r["Position"]) if not pd.isna(r["Position"]) else np.nan,
                "sprint_grid_pos": float(r["GridPosition"]) if not pd.isna(r["GridPosition"]) else np.nan,
                "sprint_points": float(r["Points"]) if not pd.isna(r["Points"]) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _merge_sprint(rows: pd.DataFrame, sdf: pd.DataFrame, label: str) -> pd.DataFrame:
    """Attach sprint columns to race-result rows; race rows without a sprint
    entry get 0 points. Warns if a sprint scorer is missing from the race."""
    rows = rows.drop(columns=[c for c in SPRINT_COLS if c in rows.columns])
    rows = rows.merge(sdf, on="driver_code", how="left")
    rows["sprint_points"] = rows["sprint_points"].fillna(0.0)
    missing = set(sdf[sdf["sprint_points"] > 0]["driver_code"]) - set(rows["driver_code"])
    if missing:
        print(f"  !! {label}: sprint scorers missing from race results: {sorted(missing)}")
    return rows


RACECRAFT_COLS = [
    "ontrack_net",          # net positions from genuine on-track swaps
    "pit_cycle_gain",       # net positions across pit-affected swaps (strategy)
    "sc_cycle_gain",        # net positions through SC/VSC/red-flag laps
    "inherited_gain",       # positions inherited from cars ahead retiring
    "stint1_pace_delta_ms", # stint-1 median lap vs field stint-1 median
    "undercut_attempts",
    "undercut_successes",
    "pit_stop_delta_ms",    # avg per-stop pit-lane duration vs race median
]


def _racecraft_metrics(laps: pd.DataFrame) -> pd.DataFrame:
    """Decomposed racecraft/strategy metrics per driver, from lap data only.

    Position-change decomposition: for every consecutive lap pair, each pair
    swap (A ends ahead of B after being behind) is classified as
      - "pit"     if either car has PitInTime/PitOutTime on one of the two laps
                  (undercuts, overcuts, pit-cycle shuffles),
      - "sc"      if either lap runs under SC/VSC/red flag,
      - "ontrack" otherwise (genuine passes; net = made - suffered).
    Cars ahead that vanish from the lap data (retire) credit one inherited
    position to every car behind them. Together with lap1_gain these buckets
    sum (approximately - lapped-car timing is inherently async) to the total
    grid->finish position change; the residual is reported in testing, not
    stored.

    stint1_pace_delta_ms: driver's stint-1 median lap (accurate laps, lap>=2)
    minus the field median of the same, in ms. Negative = faster than field.

    Undercut: an attempt is a pit entry on lap L while within 2.5s of the car
    directly ahead at the end of lap L-1, where that car stays out on L and
    pits within the next 3 laps. Success: one lap after the rival's out-lap,
    the attacker is ahead. Attempts and successes are stored separately so
    rates can be computed with a volume floor.

    pit_stop_delta_ms: mean pit-lane duration (PitInTime -> next PitOutTime,
    sanity-capped at 120s to exclude red-flag stays) minus the race median
    stop. Negative = faster crew/execution than the field.
    """
    if laps is None or len(laps) == 0:
        return pd.DataFrame(columns=["driver_code", *RACECRAFT_COLS])

    laps = laps.copy()
    laps["_t"] = laps["Time"].dt.total_seconds()
    pos = laps.pivot_table(index="LapNumber", columns="Driver", values="Position", aggfunc="first")
    tim = laps.pivot_table(index="LapNumber", columns="Driver", values="_t", aggfunc="first")
    pit_rows = laps[laps["PitInTime"].notna() | laps["PitOutTime"].notna()]
    pit_laps = set(zip(pit_rows["Driver"], pit_rows["LapNumber"]))
    lap_status = laps.groupby("LapNumber")["TrackStatus"].apply(
        lambda s: "".join(str(v) for v in s.dropna().unique())
    )
    drivers = list(pos.columns)
    acc = {d: dict(ontrack=0.0, pit=0.0, sc=0.0, inh=0.0) for d in drivers}

    lapnums = sorted(pos.index)
    for i in range(1, len(lapnums)):
        lap0, lap1 = lapnums[i - 1], lapnums[i]
        prev, curr = pos.loc[lap0], pos.loc[lap1]
        present0, present1 = prev.notna(), curr.notna()
        both = [d for d in drivers if present0[d] and present1[d]]
        vanished = [d for d in drivers if present0[d] and not present1[d]]
        blocked = any(
            b in (str(lap_status.get(lap0, "")) + str(lap_status.get(lap1, "")))
            for b in _BLOCKED_STATUS
        )
        for d in both:
            acc[d]["inh"] += sum(1.0 for v in vanished if prev[v] < prev[d])
        p = prev[both].to_numpy(dtype=float)
        c = curr[both].to_numpy(dtype=float)
        swap = (p[:, None] > p[None, :]) & (c[:, None] < c[None, :])
        for ai, bi in zip(*swap.nonzero()):
            a, b = both[ai], both[bi]
            a_pit = (a, lap0) in pit_laps or (a, lap1) in pit_laps
            b_pit = (b, lap0) in pit_laps or (b, lap1) in pit_laps
            bucket = "pit" if (a_pit or b_pit) else ("sc" if blocked else "ontrack")
            acc[a][bucket] += 1.0
            acc[b][bucket] -= 1.0

    # stint-1 pace vs field
    stint1: dict[str, float] = {}
    for d in drivers:
        dl = laps[laps["Driver"] == d]
        stints = dl["Stint"].dropna()
        if not len(stints):
            continue
        st = dl[
            (dl["Stint"] == stints.min())
            & (dl["IsAccurate"] == True)  # noqa: E712
            & dl["LapTime"].notna()
            & (dl["LapNumber"] >= 2)
        ]
        if len(st) >= 3:
            stint1[d] = float(st["LapTime"].dt.total_seconds().median() * 1000)
    field_s1 = float(np.median(list(stint1.values()))) if stint1 else np.nan

    # pit-lane durations
    durs: dict[str, list[float]] = {d: [] for d in drivers}
    in_laps: dict[str, list[float]] = {d: [] for d in drivers}
    for d in drivers:
        dl = laps[laps["Driver"] == d].sort_values("LapNumber")
        ins = dl[dl["PitInTime"].notna()][["LapNumber", "PitInTime"]]
        outs = dl[dl["PitOutTime"].notna()][["LapNumber", "PitOutTime"]]
        for _, r in ins.iterrows():
            in_laps[d].append(float(r["LapNumber"]))
            later = outs[outs["LapNumber"] > r["LapNumber"]]
            if len(later):
                ms = (later.iloc[0]["PitOutTime"] - r["PitInTime"]).total_seconds() * 1000
                if 0 < ms < 120_000:
                    durs[d].append(float(ms))
    all_stops = [x for v in durs.values() for x in v]
    med_stop = float(np.median(all_stops)) if all_stops else np.nan

    # undercut attempts / successes
    att = {d: 0.0 for d in drivers}
    suc = {d: 0.0 for d in drivers}
    lapset = set(lapnums)
    for d in drivers:
        for lap in in_laps[d]:
            l0 = lap - 1
            if l0 not in lapset:
                continue
            prev_pos, prev_t = pos.loc[l0], tim.loc[l0]
            my_p = prev_pos.get(d)
            if pd.isna(my_p):
                continue
            ahead = [x for x in drivers if prev_pos.get(x) == my_p - 1]
            if not ahead:
                continue
            rival = ahead[0]
            gap = (prev_t.get(d) or np.nan) - (prev_t.get(rival) or np.nan)
            if pd.isna(gap) or gap > 2.5:
                continue
            if (rival, lap) in pit_laps:  # double-stack same lap: not an undercut
                continue
            rival_in = next((l for l in in_laps[rival] if lap < l <= lap + 3), None)
            if rival_in is None:
                continue
            att[d] += 1.0
            settle = rival_in + 2  # rival out-lap + 1
            if settle in lapset:
                sp = pos.loc[settle]
                if not pd.isna(sp.get(d)) and not pd.isna(sp.get(rival)) and sp[d] < sp[rival]:
                    suc[d] += 1.0

    rows = []
    for d in drivers:
        rows.append(
            {
                "driver_code": d,
                "ontrack_net": acc[d]["ontrack"],
                "pit_cycle_gain": acc[d]["pit"],
                "sc_cycle_gain": acc[d]["sc"],
                "inherited_gain": acc[d]["inh"],
                "stint1_pace_delta_ms": stint1.get(d, np.nan) - field_s1
                if d in stint1
                else np.nan,
                "undercut_attempts": att[d],
                "undercut_successes": suc[d],
                "pit_stop_delta_ms": float(np.mean(durs[d]) - med_stop) if durs[d] else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _lap_metrics(laps: pd.DataFrame, code: str, grid_pos: float) -> dict:
    """Pace, consistency, degradation and overtake metrics for one driver."""
    d = laps[laps["Driver"] == code].sort_values("LapNumber")
    out = {
        "median_lap_ms": np.nan,
        "lap_std_ms": np.nan,
        "deg_slope_s_per_lap": np.nan,
        "n_laps": int(len(d)),
        "lap1_gain": np.nan,
        "overtakes": np.nan,
    }
    if d.empty:
        return out

    acc = d[(d["IsAccurate"] == True) & d["LapTime"].notna()]  # noqa: E712
    if len(acc) >= 3:
        secs = acc["LapTime"].dt.total_seconds()
        out["median_lap_ms"] = float(secs.median() * 1000)
        out["lap_std_ms"] = float(secs.std() * 1000)

        # degradation: mean lap-time slope (s/lap) across stints with >=5 accurate laps
        slopes = []
        for _, stint in acc.groupby("Stint"):
            if len(stint) >= 5:
                slopes.append(np.polyfit(stint["LapNumber"], stint["LapTime"].dt.total_seconds(), 1)[0])
        if slopes:
            out["deg_slope_s_per_lap"] = float(np.mean(slopes))

    pos = d[["LapNumber", "Position"]].dropna()
    if not pos.empty:
        first = pos.iloc[0]
        if not np.isnan(grid_pos) and first["LapNumber"] == 1:
            out["lap1_gain"] = float(grid_pos - first["Position"])
        # positions gained lap-over-lap after lap 1 (proxy for on-track overtakes)
        after = pos[pos["LapNumber"] >= 2]["Position"].to_numpy()
        if len(after) >= 2:
            gains = np.diff(after) * -1  # position number going down = places gained
            out["overtakes"] = float(gains[gains > 0].sum())
    return out


def ingest_race(season: int, rnd: int, event) -> tuple[list[dict], dict] | None:
    import fastf1

    # canonical track labels (schedule renames split history otherwise)
    location = config.normalize_location(event["Location"])
    country = config.normalize_country(location, event["Country"])

    race = fastf1.get_session(season, rnd, "R")
    race.load(laps=True, telemetry=False, weather=True, messages=False)
    res = race.results
    if res is None or len(res) == 0 or res["Position"].isna().all():
        print(f"  !! {season} R{rnd}: no race results, skipping")
        return None

    try:
        quali = fastf1.get_session(season, rnd, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
        qdf = _quali_features(quali)
    except Exception as exc:  # quali data can be missing independently
        from fastf1 import RateLimitExceededError

        if isinstance(exc, RateLimitExceededError):
            raise
        print(f"  !! {season} R{rnd}: quali load failed ({exc}); continuing without")
        qdf = pd.DataFrame(columns=["driver_code", "quali_pos", "quali_best_ms", "quali_gap_to_pole_ms"])

    laps = race.laps
    weather = race.weather_data
    wet = bool(weather["Rainfall"].any()) if weather is not None and len(weather) else False
    # graded wetness: fraction of weather samples (~1/min) reporting rainfall.
    # 0.02 = brief shower, 0.9 = wet throughout; complements the any-rain bool
    wet_fraction = (
        float(weather["Rainfall"].mean()) if weather is not None and len(weather) else np.nan
    )
    track_temp = float(weather["TrackTemp"].mean()) if weather is not None and len(weather) else np.nan
    air_temp = float(weather["AirTemp"].mean()) if weather is not None and len(weather) else np.nan

    sc_laps = vsc_laps = 0
    if laps is not None and len(laps):
        status_by_lap = laps.groupby("LapNumber")["TrackStatus"].apply(
            lambda s: "".join(str(v) for v in s.dropna().unique())
        )
        sc_laps = int(status_by_lap.str.contains("4").sum())
        vsc_laps = int(status_by_lap.str.contains("6").sum())

    n_starters = int(len(res))
    race_median_ms = np.nan

    driver_rows = []
    for _, r in res.iterrows():
        code = r["Abbreviation"]
        grid = float(r["GridPosition"]) if not pd.isna(r["GridPosition"]) else np.nan
        if grid == 0:  # pit-lane start
            grid = float(n_starters)
        status = str(r["Status"])
        finished = ("Finished" in status) or ("Lap" in status)
        # a post-race disqualification is not a reliability event: the car
        # finished on track, so it must not inflate DNF rates
        if status == "Disqualified":
            finished = True
        lm = _lap_metrics(laps, code, grid) if laps is not None else {}
        driver_rows.append(
            {
                "season": season,
                "round": rnd,
                "event_name": event["EventName"],
                "location": location,
                "country": country,
                "date": pd.Timestamp(event["EventDate"]).date().isoformat(),
                "driver_code": code,
                "driver_number": str(r["DriverNumber"]),
                "driver_name": r["FullName"],
                "team": r["TeamName"],
                "team_color": str(r.get("TeamColor", "") or ""),
                "grid_pos": grid,
                "finish_pos": float(r["Position"]) if not pd.isna(r["Position"]) else np.nan,
                "classified": str(r["ClassifiedPosition"]),
                "status": status,
                "points": float(r["Points"]) if not pd.isna(r["Points"]) else 0.0,
                "dnf": not finished,
                "wet_race": wet,
                "wet_fraction": wet_fraction,
                "track_temp": track_temp,
                "air_temp": air_temp,
                **lm,
            }
        )

    rows = pd.DataFrame(driver_rows).merge(qdf, on="driver_code", how="left")
    race_median_ms = rows["median_lap_ms"].median()
    rows["pace_delta_ms"] = rows["median_lap_ms"] - race_median_ms  # negative = faster than field

    clean = _clean_overtakes(laps)
    rows["overtakes_clean"] = rows["driver_code"].map(clean).astype(float)

    # decomposed racecraft/strategy metrics
    rc = _racecraft_metrics(laps)
    rows = rows.drop(columns=[c for c in RACECRAFT_COLS if c in rows.columns])
    rows = rows.merge(rc, on="driver_code", how="left")

    # sprint weekends: attach sprint finish + championship points
    if _is_sprint_event(event):
        try:
            sdf = _sprint_results(season, rnd)
        except Exception as exc:
            from fastf1 import RateLimitExceededError

            if isinstance(exc, RateLimitExceededError):
                raise
            print(f"  !! {season} R{rnd}: sprint load failed ({exc}); continuing without")
            sdf = pd.DataFrame(columns=["driver_code", *SPRINT_COLS])
        rows = _merge_sprint(rows, sdf, f"{season} R{rnd}")
    else:
        rows["sprint_finish_pos"] = np.nan
        rows["sprint_grid_pos"] = np.nan
        rows["sprint_points"] = 0.0

    meta = {
        "season": season,
        "round": rnd,
        "event_name": event["EventName"],
        "location": location,
        "country": country,
        "date": pd.Timestamp(event["EventDate"]).date().isoformat(),
        "wet_race": wet,
        "wet_fraction": wet_fraction,
        "track_temp": track_temp,
        "air_temp": air_temp,
        "sc_laps": sc_laps,
        "vsc_laps": vsc_laps,
        "total_overtakes": float(rows["overtakes"].sum(skipna=True)),
        "total_overtakes_clean": float(rows["overtakes_clean"].sum(skipna=True)),
        "n_starters": n_starters,
        "n_dnf": int(rows["dnf"].sum()),
    }
    return rows.to_dict("records"), meta


def backfill_sprints(results: pd.DataFrame, schedule, season: int) -> pd.DataFrame:
    """Add sprint columns to already-ingested races (resumable, incremental).

    Conventional rounds get sprint_points = 0.0. Sprint rounds whose
    sprint_points are still NaN are fetched one by one; the parquet is saved
    after each so a rate-limit interruption resumes where it left off.
    """
    if "sprint_points" not in results.columns:
        for col in SPRINT_COLS:
            results[col] = np.nan

    for _, event in schedule.iterrows():
        rnd = int(event["RoundNumber"])
        mask = (results["season"] == season) & (results["round"] == rnd)
        if not mask.any():
            continue  # race not ingested at all; the main loop handles it
        if not _is_sprint_event(event):
            results.loc[mask, "sprint_points"] = results.loc[mask, "sprint_points"].fillna(0.0)
            continue
        if results.loc[mask, "sprint_points"].notna().all():
            continue  # already backfilled
        print(f"Backfilling sprint {season} R{rnd:02d} {event['EventName']} ...", flush=True)
        try:
            sdf = _with_rate_limit_retry(
                lambda: _sprint_results(season, rnd), f"sprint {season} R{rnd}"
            )
        except Exception:
            print(f"  !! sprint {season} R{rnd} failed:\n{traceback.format_exc()}")
            continue
        updated = _merge_sprint(results[mask].copy(), sdf, f"{season} R{rnd}")
        for col in SPRINT_COLS:
            results.loc[mask, col] = updated[col].to_numpy()
        results.to_parquet(config.RESULTS_PARQUET, index=False)
    return results


def backfill_clean_overtakes(
    results: pd.DataFrame, races: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reprocess already-ingested races to add the cleaned overtake metric.

    Lap data is served from the fastf1 on-disk cache — no bulk API refetch.
    Races whose total_overtakes_clean is still NaN are processed one at a
    time and both parquets saved after each, so the pass is resumable.
    """
    if "total_overtakes_clean" not in races.columns:
        races["total_overtakes_clean"] = np.nan
    if "overtakes_clean" not in results.columns:
        results["overtakes_clean"] = np.nan

    todo = races[(races["season"] == season) & (races["total_overtakes_clean"].isna())]
    for _, meta in todo.iterrows():
        rnd = int(meta["round"])
        print(f"Reprocessing overtakes {season} R{rnd:02d} {meta['event_name']} ...", flush=True)
        try:
            import fastf1

            sess = _with_rate_limit_retry(
                lambda: fastf1.get_session(season, rnd, "R"), f"overtakes {season} R{rnd}"
            )
            _with_rate_limit_retry(
                lambda: sess.load(laps=True, telemetry=False, weather=False, messages=False),
                f"overtakes load {season} R{rnd}",
            )
            clean = _clean_overtakes(sess.laps)
        except Exception:
            print(f"  !! overtakes {season} R{rnd} failed:\n{traceback.format_exc()}")
            continue
        mask = (results["season"] == season) & (results["round"] == rnd)
        results.loc[mask, "overtakes_clean"] = (
            results.loc[mask, "driver_code"].map(clean).astype(float)
        )
        rmask = (races["season"] == season) & (races["round"] == rnd)
        races.loc[rmask, "total_overtakes_clean"] = float(
            results.loc[mask, "overtakes_clean"].sum(skipna=True)
        )
        results.to_parquet(config.RESULTS_PARQUET, index=False)
        races.to_parquet(config.RACES_PARQUET, index=False)
    return results, races


def backfill_dnf_dsq(results: pd.DataFrame, races: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reclassify post-race disqualifications as finishers (pure pandas,
    idempotent). A DSQ'd car crossed the line - counting it as a DNF inflates
    driver/team reliability rates and the simulation's DNF sampling."""
    mask = (results["status"] == "Disqualified") & results["dnf"]
    if not mask.any():
        return results, races
    results.loc[mask, "dnf"] = False
    for s, r in results.loc[mask, ["season", "round"]].drop_duplicates().itertuples(index=False):
        rm = (results["season"] == s) & (results["round"] == r)
        races.loc[(races["season"] == s) & (races["round"] == r), "n_dnf"] = int(
            results.loc[rm, "dnf"].sum()
        )
    print(f"Reclassified {int(mask.sum())} disqualified finisher(s) as non-DNF")
    results.to_parquet(config.RESULTS_PARQUET, index=False)
    races.to_parquet(config.RACES_PARQUET, index=False)
    return results, races


def backfill_wet_fraction(
    results: pd.DataFrame, races: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add graded wetness (fraction of weather samples with rainfall) to
    already-ingested races. Weather is served from the fastf1 cache; races
    with wet_fraction still NaN are processed one at a time and saved after
    each, so the pass is resumable."""
    if "wet_fraction" not in races.columns:
        races["wet_fraction"] = np.nan
    if "wet_fraction" not in results.columns:
        results["wet_fraction"] = np.nan

    todo = races[(races["season"] == season) & (races["wet_fraction"].isna())]
    for _, meta in todo.iterrows():
        rnd = int(meta["round"])
        print(f"Backfilling wet fraction {season} R{rnd:02d} {meta['event_name']} ...", flush=True)
        try:
            import fastf1

            sess = _with_rate_limit_retry(
                lambda: fastf1.get_session(season, rnd, "R"), f"wet {season} R{rnd}"
            )
            _with_rate_limit_retry(
                lambda: sess.load(laps=False, telemetry=False, weather=True, messages=False),
                f"wet load {season} R{rnd}",
            )
            weather = sess.weather_data
            wf = float(weather["Rainfall"].mean()) if weather is not None and len(weather) else np.nan
        except Exception:
            print(f"  !! wet fraction {season} R{rnd} failed:\n{traceback.format_exc()}")
            continue
        results.loc[(results["season"] == season) & (results["round"] == rnd), "wet_fraction"] = wf
        races.loc[(races["season"] == season) & (races["round"] == rnd), "wet_fraction"] = wf
        results.to_parquet(config.RESULTS_PARQUET, index=False)
        races.to_parquet(config.RACES_PARQUET, index=False)
    return results, races


def backfill_racecraft(
    results: pd.DataFrame, races: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add decomposed racecraft metrics to already-ingested races from the
    fastf1 cache (no bulk refetch). The races-meta marker column
    racecraft_ingested tracks progress; parquets are saved after each race so
    the pass is resumable."""
    if "racecraft_ingested" not in races.columns:
        races["racecraft_ingested"] = np.nan
    for col in RACECRAFT_COLS:
        if col not in results.columns:
            results[col] = np.nan

    todo = races[(races["season"] == season) & (races["racecraft_ingested"].isna())]
    for _, meta in todo.iterrows():
        rnd = int(meta["round"])
        print(f"Backfilling racecraft {season} R{rnd:02d} {meta['event_name']} ...", flush=True)
        try:
            import fastf1

            sess = _with_rate_limit_retry(
                lambda: fastf1.get_session(season, rnd, "R"), f"racecraft {season} R{rnd}"
            )
            _with_rate_limit_retry(
                lambda: sess.load(laps=True, telemetry=False, weather=False, messages=False),
                f"racecraft load {season} R{rnd}",
            )
            rc = _racecraft_metrics(sess.laps).set_index("driver_code")
        except Exception:
            print(f"  !! racecraft {season} R{rnd} failed:\n{traceback.format_exc()}")
            continue
        mask = (results["season"] == season) & (results["round"] == rnd)
        for col in RACECRAFT_COLS:
            results.loc[mask, col] = results.loc[mask, "driver_code"].map(rc[col]).astype(float)
        races.loc[(races["season"] == season) & (races["round"] == rnd), "racecraft_ingested"] = 1.0
        results.to_parquet(config.RESULTS_PARQUET, index=False)
        races.to_parquet(config.RACES_PARQUET, index=False)
    return results, races


def normalize_track_labels(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Rewrite location/country aliases to their canonical values (idempotent,
    pure pandas — no API calls). Unmapped locations pass through untouched."""
    if not len(df) or "location" not in df.columns:
        return df
    loc = df["location"].map(config.normalize_location)
    country = pd.Series(
        [config.normalize_country(l, c) for l, c in zip(loc, df["country"])],
        index=df.index,
    )
    n_changed = int((loc != df["location"]).sum() + (country != df["country"]).sum())
    if n_changed:
        df = df.assign(location=loc, country=country)
        print(f"Normalized {n_changed} track label(s) in {label}")
    return df


def main() -> None:
    import fastf1

    config.setup_fastf1()
    normalize_only = len(sys.argv) > 1 and sys.argv[1] == "normalize"
    seasons = (
        []
        if normalize_only
        else [int(sys.argv[1])] if len(sys.argv) > 1 else config.SEASONS
    )

    results = pd.read_parquet(config.RESULTS_PARQUET) if config.RESULTS_PARQUET.exists() else pd.DataFrame()
    races = pd.read_parquet(config.RACES_PARQUET) if config.RACES_PARQUET.exists() else pd.DataFrame()

    # canonical track labels for rows ingested before normalization existed
    if len(results):
        fixed = normalize_track_labels(results, "race_results")
        if fixed is not results:
            results = fixed
            results.to_parquet(config.RESULTS_PARQUET, index=False)
    if len(races):
        fixed = normalize_track_labels(races, "race_meta")
        if fixed is not races:
            races = fixed
            races.to_parquet(config.RACES_PARQUET, index=False)

    # DSQ-as-DNF correction for rows ingested before the fix (pure pandas)
    if len(results) and len(races):
        results, races = backfill_dnf_dsq(results, races)

    done = set(zip(results["season"], results["round"])) if len(results) else set()

    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    for season in seasons:
        schedule = _with_rate_limit_retry(
            lambda: fastf1.get_event_schedule(season, include_testing=False),
            f"schedule {season}",
        )
        # sprint columns for races ingested before sprint support existed
        if len(results):
            results = backfill_sprints(results, schedule, season)
        # cleaned overtake metric for races ingested before it existed
        if len(results) and len(races):
            results, races = backfill_clean_overtakes(results, races, season)
        # graded wetness for races ingested before it existed
        if len(results) and len(races):
            results, races = backfill_wet_fraction(results, races, season)
        # decomposed racecraft metrics for races ingested before they existed
        if len(results) and len(races):
            results, races = backfill_racecraft(results, races, season)

        for _, event in schedule.iterrows():
            rnd = int(event["RoundNumber"])
            race_end = event["Session5DateUtc"]  # race session UTC time
            if pd.isna(race_end) or race_end + pd.Timedelta(hours=4) > now:
                continue  # race not completed yet
            if (season, rnd) in done:
                continue
            print(f"Ingesting {season} R{rnd:02d} {event['EventName']} ...", flush=True)
            try:
                out = _with_rate_limit_retry(
                    lambda: ingest_race(season, rnd, event), f"{season} R{rnd}"
                )
            except Exception:
                print(f"  !! {season} R{rnd} failed:\n{traceback.format_exc()}")
                continue
            if out is None:
                continue
            rows, meta = out
            results = pd.concat([results, pd.DataFrame(rows)], ignore_index=True)
            races = pd.concat([races, pd.DataFrame([meta])], ignore_index=True)
            results.to_parquet(config.RESULTS_PARQUET, index=False)
            races.to_parquet(config.RACES_PARQUET, index=False)
            done.add((season, rnd))

    print(f"Done. {len(races)} races, {len(results)} driver-race rows.")


if __name__ == "__main__":
    main()
