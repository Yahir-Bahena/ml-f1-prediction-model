"""Phase 1: extract practice long-run features for every raced event.

Per driver-weekend, writes: practice_longrun_delta_ms (within-compound delta vs
field race-sim median), practice_sample_confidence (0-3, capped at 1 on sprint
FP1 — see miss-overlap findings: FP1 programs are team-biased), n_sim_laps.
Session choice: FP2 then FP3, KEEPING THE BETTER result (bug fix from the
overlap script); sprint weekends: FP1 only.

Resumable: appends to analysis/practice_features.csv, skips recorded events,
records rate-limit failures as status=NO_DATA rows that are retried on re-run.

Run from backend/:  ./.venv/bin/python analysis/practice_features_extract.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
fastf1.Cache.enable_cache("data/cache")

OUT = Path("analysis/practice_features.csv")
MIN_LAPS, BAND_LO, BAND_HI, RESID_STD_MAX, TRIM_PCT = 4, 1.025, 1.12, 0.010, 1.035


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


def per_driver(st: pd.DataFrame) -> pd.DataFrame:
    """Within-compound deltas vs field median + per-driver confidence."""
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
    agg = d.groupby("driver").agg(
        delta_ms=("delta_ms", "mean"), laps=("laps", "sum"), n_runs=("delta_ms", "size"))
    conf = []
    for _, r in agg.iterrows():
        if r["n_runs"] >= 3 or (r["n_runs"] >= 2 and r["laps"] >= 10):
            conf.append(3)
        elif r["laps"] >= 7 or r["n_runs"] >= 2:
            conf.append(2)
        else:
            conf.append(1)
    agg["confidence"] = conf
    return agg.reset_index()


def main() -> None:
    meta = pd.read_parquet("data/processed/race_meta.parquet")
    ds = pd.read_parquet("data/processed/dataset.parquet")
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        ok_prev = prev[prev["status"] != "NO_DATA"]
        done = set(zip(ok_prev["season"], ok_prev["round"]))
        prev[prev["status"] != "NO_DATA"].to_csv(OUT, index=False)  # drop NO_DATA -> retried
    header = not OUT.exists()

    for r in meta.sort_values(["season", "round"]).itertuples():
        key = (int(r.season), int(r.round))
        if key in done:
            continue
        entry = ds[(ds.season == r.season) & (ds["round"] == r.round)]["driver_code"].tolist()
        try:
            ev = fastf1.get_event(int(r.season), int(r.round))
            sprint = "sprint" in str(ev["EventFormat"]).lower()
            best_dd, best_n = pd.DataFrame(), -1
            for name in (["Practice 1"] if sprint else ["Practice 2", "Practice 3"]):
                try:
                    s = fastf1.get_session(int(r.season), int(r.round), name)
                    s.load(laps=True, telemetry=False, weather=False, messages=False)
                    dd = per_driver(classify(s))
                    if len(dd) > best_n:          # keep the better session
                        best_dd, best_n = dd, len(dd)
                    if best_n >= 10:
                        break
                except Exception:  # noqa: BLE001 — session-level failure, try next
                    continue
            rows = []
            for d in entry:
                m = best_dd[best_dd.driver == d] if len(best_dd) else pd.DataFrame()
                if len(m):
                    conf = int(m.iloc[0]["confidence"])
                    if sprint:
                        conf = min(conf, 1)  # FP1 programs are team-biased (VER false negatives)
                    rows.append({"season": key[0], "round": key[1], "driver": d, "status": "OK",
                                 "sprint": sprint, "delta_ms": float(m.iloc[0]["delta_ms"]),
                                 "confidence": conf, "n_sim_laps": int(m.iloc[0]["laps"])})
                else:
                    rows.append({"season": key[0], "round": key[1], "driver": d, "status": "OK",
                                 "sprint": sprint, "delta_ms": np.nan, "confidence": 0,
                                 "n_sim_laps": 0})
        except Exception as e:  # noqa: BLE001 — event-level failure (rate limit)
            rows = [{"season": key[0], "round": key[1], "driver": d, "status": "NO_DATA",
                     "sprint": None, "delta_ms": np.nan, "confidence": 0, "n_sim_laps": 0}
                    for d in entry]
            print(f"{key} FAILED: {type(e).__name__}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=header, index=False)
        header = False
        n_cov = sum(1 for x in rows if x["confidence"] > 0)
        print(f"{key[0]} R{key[1]:>2} {r.location:<18} covered {n_cov}/{len(entry)}", flush=True)
    print("EXTRACTION COMPLETE")


if __name__ == "__main__":
    main()
