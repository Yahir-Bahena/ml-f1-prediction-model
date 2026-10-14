"""Overlap analysis: the 23 current winner-misses vs the practice-pace signal.

For each miss race: load the practice session (FP2, FP1 on sprint weekends),
classify race sims (Phase-0-tuned), compute practice_longrun_delta_ms, and
report the eventual winner's practice rank + surprise proxy alongside their
quali_surprise. Rate-limit tolerant: failures are recorded as NO_DATA and the
script is re-runnable (sessions cache locally).

Surprise here is the Phase-0 PROXY (practice rank vs pace_delta_5 expectation);
the real F2 (vs own rolling practice norm) needs the full backfill.

Run from backend/:  ./.venv/bin/python analysis/miss_practice_overlap.py
"""
from __future__ import annotations

import logging

import fastf1
import numpy as np
import pandas as pd

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
fastf1.Cache.enable_cache("data/cache")

MIN_LAPS, BAND_LO, BAND_HI, RESID_STD_MAX, TRIM_PCT = 4, 1.025, 1.12, 0.010, 1.035

# (season, round, winner_code, taxonomy) — 2026-07-18 miss taxonomy
MISSES = [
    (2023, 12, "VER", "race_pace"), (2023, 18, "VER", "race_pace"),
    (2024, 3, "SAI", "reliability"), (2024, 6, "NOR", "strategy"),
    (2024, 11, "RUS", "unpredictable"), (2024, 12, "HAM", "reliability"),
    (2024, 14, "HAM", "unpredictable"), (2024, 16, "LEC", "strategy"),
    (2024, 19, "LEC", "strategy"), (2024, 21, "VER", "weather"),
    (2024, 23, "VER", "race_pace"), (2025, 3, "VER", "quali_overperf"),
    (2025, 5, "PIA", "unpredictable"), (2025, 6, "PIA", "weather"),
    (2025, 7, "VER", "unpredictable"), (2025, 10, "RUS", "quali_overperf"),
    (2025, 12, "NOR", "unpredictable"), (2025, 13, "PIA", "weather"),
    (2025, 22, "VER", "unpredictable"), (2025, 23, "VER", "strategy"),
    (2026, 5, "ANT", "reliability"), (2026, 7, "HAM", "race_pace"),
    (2026, 9, "LEC", "unpredictable"),
]


def classify(session) -> pd.DataFrame:
    laps = session.laps
    sec = laps["LapTime"].dt.total_seconds()
    best = sec.min()
    if np.isnan(best):
        return pd.DataFrame()
    valid = (laps["PitOutTime"].isna() & laps["PitInTime"].isna()
             & laps["Deleted"].ne(True) & laps["IsAccurate"].eq(True)
             & laps["TrackStatus"].astype(str).eq("1") & sec.notna())
    out = []
    for (drv, stint), g in laps[valid].groupby(["Driver", "Stint"]):
        t = g["LapTime"].dt.total_seconds().to_numpy()
        if len(t) < 3:
            continue
        t = t[t <= np.median(t) * TRIM_PCT]
        if len(t) < MIN_LAPS:
            continue
        med = float(np.median(t))
        if not (BAND_LO <= med / best <= BAND_HI):
            continue
        x = np.arange(len(t))
        sl, ic = np.polyfit(x, t, 1)
        if float(np.std(t - (sl * x + ic))) / med > RESID_STD_MAX:
            continue
        out.append({"driver": drv, "compound": g["Compound"].iloc[0],
                    "median_s": med, "laps": int(len(t))})
    return pd.DataFrame(out)


def deltas(st: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp, g in st.groupby("compound"):
        if g["driver"].nunique() < 3:
            continue
        field = g.groupby("driver")["median_s"].min().median()
        for drv, r in g.groupby("driver").agg(m=("median_s", "min"), laps=("laps", "sum")).iterrows():
            rows.append({"driver": drv, "delta_ms": (r["m"] - field) * 1000, "laps": r["laps"]})
    d = pd.DataFrame(rows)
    if not len(d):
        return d
    return (d.groupby("driver").apply(lambda g: np.average(g["delta_ms"], weights=g["laps"]),
                                      include_groups=False)
            .rename("delta_ms").reset_index())


def main() -> None:
    ds = pd.read_parquet("data/processed/dataset.parquet")
    results = []
    for season, rnd, winner, tax in MISSES:
        row = {"season": season, "round": rnd, "winner": winner, "taxonomy": tax}
        try:
            ev = fastf1.get_event(season, rnd)
            sprint = "sprint" in str(ev["EventFormat"]).lower()
            row["event"] = str(ev["EventName"]).replace(" Grand Prix", "")
            row["sprint"] = sprint
            dd = pd.DataFrame()
            for name in (["Practice 1"] if sprint else ["Practice 2", "Practice 3"]):
                s = fastf1.get_session(season, rnd, name)
                s.load(laps=True, telemetry=False, weather=False, messages=False)
                dd = deltas(classify(s))
                if len(dd) >= 5:
                    break
            race = ds[(ds.season == season) & (ds["round"] == rnd)][
                ["driver_code", "quali_surprise", "pace_delta_5"]]
            if not len(dd):
                row["status"] = "NO_SIMS"
            else:
                m = dd.merge(race, left_on="driver", right_on="driver_code")
                m["prank"] = m["delta_ms"].rank()
                m["erank"] = m["pace_delta_5"].rank()
                w = m[m.driver == winner]
                row["status"] = "OK"
                row["n_covered"] = len(m)
                if len(w):
                    r = w.iloc[0]
                    row.update(w_delta_ms=round(float(r["delta_ms"])),
                               w_prank=f"{r['prank']:.0f}/{len(m)}",
                               w_surprise_proxy=round(float(r["erank"] - r["prank"]), 1))
                else:
                    row["status"] = "WINNER_NOT_COVERED"
            wq = race[race.driver_code == winner]["quali_surprise"]
            row["w_qsurp"] = round(float(wq.iloc[0]), 1) if len(wq) and not pd.isna(wq.iloc[0]) else None
        except Exception as e:  # noqa: BLE001
            row["status"] = f"NO_DATA({type(e).__name__})"
        results.append(row)
        print({k: v for k, v in row.items()}, flush=True)
    pd.DataFrame(results).to_csv("analysis/miss_practice_overlap.csv", index=False)
    print("DONE")


if __name__ == "__main__":
    main()
