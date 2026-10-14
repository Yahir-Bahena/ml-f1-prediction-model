"""Coverage audit: race-sim availability across 2022-2026, BEFORE feature build.

For every raced event, classify FP race sims (Phase-0-tuned thresholds) and
record per-driver coverage + confidence. Conventional weekends: FP2, with FP3
fallback when FP2 yields <5 drivers. Sprint weekends: FP1 (the only practice).
Incremental + resumable: appends to analysis/practice_coverage.csv and skips
events already recorded, so rate-limit interruptions just need a re-run.

Run from backend/:  ./.venv/bin/python analysis/practice_coverage_audit.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
fastf1.Cache.enable_cache("data/cache")

OUT = Path("analysis/practice_coverage.csv")

# Phase-0-tuned classifier
MIN_LAPS = 4
BAND_LO, BAND_HI = 1.025, 1.12
RESID_STD_MAX = 0.010
TRIM_PCT = 1.035

STREET = {"Monaco", "Baku", "Marina Bay", "Jeddah", "Las Vegas", "Miami"}


def stints_for(session) -> pd.DataFrame:
    laps = session.laps
    sec = laps["LapTime"].dt.total_seconds()
    session_best = sec.min()
    if np.isnan(session_best):
        return pd.DataFrame()
    valid = (
        laps["PitOutTime"].isna() & laps["PitInTime"].isna()
        & laps["Deleted"].ne(True) & laps["IsAccurate"].eq(True)
        & laps["TrackStatus"].astype(str).eq("1") & sec.notna()
    )
    rows = []
    for (drv, stint), g in laps[valid].groupby(["Driver", "Stint"]):
        t = g["LapTime"].dt.total_seconds().to_numpy()
        if len(t) < 3:
            continue
        med = np.median(t)
        t = t[t <= med * TRIM_PCT]
        if len(t) < MIN_LAPS:
            continue
        med = float(np.median(t))
        ratio = med / session_best
        if not (BAND_LO <= ratio <= BAND_HI):
            continue
        x = np.arange(len(t))
        slope, icpt = np.polyfit(x, t, 1)
        if float(np.std(t - (slope * x + icpt))) / med > RESID_STD_MAX:
            continue
        rows.append({"driver": drv, "laps": int(len(t))})
    return pd.DataFrame(rows)


def confidence(stints: pd.DataFrame) -> pd.Series:
    """Per-driver 0-3: 0 none; 1 one short (<7); 2 one >=7 or two short; 3 more."""
    def score(g):
        n, solid = len(g), (g["laps"] >= 7).sum()
        if n >= 3 or (n >= 2 and solid >= 1):
            return 3
        if solid >= 1 or n >= 2:
            return 2
        return 1
    return stints.groupby("driver").apply(score, include_groups=False)


def main() -> None:
    meta = pd.read_parquet("data/processed/race_meta.parquet")
    ds = pd.read_parquet("data/processed/dataset.parquet")
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(zip(prev["season"], prev["round"]))
    header_needed = not OUT.exists()

    for r in meta.sort_values(["season", "round"]).itertuples():
        key = (int(r.season), int(r.round))
        if key in done:
            continue
        entry = ds[(ds.season == r.season) & (ds["round"] == r.round)]["driver_code"].tolist()
        try:
            ev = fastf1.get_event(int(r.season), int(r.round))
            is_sprint = "sprint" in str(ev["EventFormat"]).lower()
            sessions = ["Practice 1"] if is_sprint else ["Practice 2"]
            st = pd.DataFrame()
            used = None
            for name in sessions + ([] if is_sprint else ["Practice 3"]):
                s = fastf1.get_session(int(r.season), int(r.round), name)
                s.load(laps=True, telemetry=False, weather=False, messages=False)
                cand = stints_for(s)
                if len(cand) and cand["driver"].nunique() >= 5:
                    st, used = cand, name
                    break
                if len(cand) and (len(st) == 0 or cand["driver"].nunique() > st["driver"].nunique()):
                    st, used = cand, name
            conf = confidence(st) if len(st) else pd.Series(dtype=int)
        except Exception as e:  # noqa: BLE001 — record failure, keep going
            st, conf, used, is_sprint = pd.DataFrame(), pd.Series(dtype=int), f"ERROR:{type(e).__name__}", None
        rows = [{
            "season": int(r.season), "round": int(r.round), "location": r.location,
            "sprint": bool(is_sprint) if is_sprint is not None else None,
            "street": r.location in STREET, "wet_race": bool(r.wet_race),
            "session_used": used, "driver": d,
            "confidence": int(conf.get(d, 0)),
        } for d in entry]
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=header_needed, index=False)
        header_needed = False
        n_cov = int((pd.Series([x["confidence"] for x in rows]) > 0).sum())
        print(f"{r.season} R{int(r.round):>2} {r.location:<18} {str(used):<12} "
              f"covered {n_cov}/{len(entry)}", flush=True)

    print("AUDIT DOWNLOAD COMPLETE")


if __name__ == "__main__":
    main()
