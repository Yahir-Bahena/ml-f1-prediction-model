"""Phase 0: hand-validation of the race-sim classifier on three 2025 weekends.

Answers ONE question before any pipeline is built: can race-relevant pace be
identified from public FP2 data? Prints every candidate stint with its
accept/reject reason, the per-driver practice_longrun_delta_ms, a Phase-0 proxy
for practice_pace_surprise (vs pace_delta_5 expectations from the dataset — the
real F2 needs a rolling backfill that doesn't exist yet), and the comparison
against the actual race result.

Run from backend/:  ./.venv/bin/python analysis/phase0_practice_validation.py
Weekends: 2025 R3 Suzuka (unexpected winner pace), R5 Jeddah (false-signal
expected — race decided by a penalty), R4 Bahrain (normal dominant weekend).
"""
from __future__ import annotations

import logging

import fastf1
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.getLogger("fastf1").setLevel(logging.ERROR)
fastf1.Cache.enable_cache("data/cache")

# classifier tunables (Phase 0 defaults; hand-tuning them is the point)
MIN_LAPS = 5
BAND_LO, BAND_HI = 1.025, 1.12   # stint median vs session-overall best lap
RESID_STD_MAX = 0.010            # detrended residual robust-std vs stint median
TRIM_PCT = 1.025                 # in-stint traffic trim: laps > median * this

WEEKENDS = [(2025, 3, "Suzuka"), (2025, 5, "Jeddah"), (2025, 4, "Bahrain")]


def classify_session(year: int, rnd: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stint table with verdicts, per-driver deltas) for one FP2."""
    s = fastf1.get_session(year, rnd, "FP2")
    s.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = s.laps
    sec = laps["LapTime"].dt.total_seconds()
    session_best = sec.min()
    mid = laps["LapStartTime"].median()

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
        keep = t <= med * TRIM_PCT           # traffic/cooldown trim
        t_clean = t[keep]
        med = np.median(t_clean)
        reasons = []
        if len(t_clean) < MIN_LAPS:
            reasons.append(f"short ({len(t_clean)} clean laps)")
        ratio = med / session_best
        if ratio < BAND_LO:
            reasons.append(f"too fast ({ratio:.3f} of best -> quali trim)")
        if ratio > BAND_HI:
            reasons.append(f"too slow ({ratio:.3f} of best)")
        resid_std = np.nan
        if len(t_clean) >= 3:
            x = np.arange(len(t_clean))
            slope, icpt = np.polyfit(x, t_clean, 1)
            resid_std = float(np.std(t_clean - (slope * x + icpt)))
            if resid_std / med > RESID_STD_MAX:
                reasons.append(f"inconsistent (resid {resid_std:.2f}s)")
        late = bool(g["LapStartTime"].median() > mid)
        rows.append({
            "driver": drv, "stint": int(stint),
            "compound": g["Compound"].iloc[0], "tyre_age0": float(g["TyreLife"].min()),
            "laps_clean": int(len(t_clean)), "median_s": round(float(med), 3),
            "pct_of_best": round(float(ratio), 3),
            "resid_std_s": round(resid_std, 3) if not np.isnan(resid_std) else None,
            "late_fp2": late,
            "verdict": "ACCEPT" if not reasons else "reject",
            "reason": "; ".join(reasons) or "race sim",
        })
    st = pd.DataFrame(rows).sort_values(["verdict", "median_s"])

    acc = st[st["verdict"] == "ACCEPT"].copy()
    # within-compound delta vs field median of accepted sims on that compound
    deltas = []
    for comp, g in acc.groupby("compound"):
        if g["driver"].nunique() < 3:
            continue  # too few sims on this compound to define a field
        field_med = g.groupby("driver")["median_s"].min().median()
        per = g.groupby("driver").agg(median_s=("median_s", "min"), laps=("laps_clean", "sum"))
        for drv, r in per.iterrows():
            deltas.append({"driver": drv, "compound": comp,
                           "delta_ms": (r["median_s"] - field_med) * 1000, "laps": r["laps"]})
    dd = pd.DataFrame(deltas)
    if len(dd):
        dd = (dd.groupby("driver")
                .apply(lambda g: np.average(g["delta_ms"], weights=g["laps"]),
                       include_groups=False)
                .rename("practice_longrun_delta_ms").reset_index()
                .sort_values("practice_longrun_delta_ms"))
    return st, dd


def main() -> None:
    ds = pd.read_parquet("data/processed/dataset.parquet")
    for year, rnd, name in WEEKENDS:
        print(f"\n{'='*80}\n{year} R{rnd} {name} - FP2\n{'='*80}")
        st, dd = classify_session(year, rnd)
        n_acc = (st['verdict'] == 'ACCEPT').sum()
        print(f"candidate stints: {len(st)} | accepted: {n_acc}")
        print(st.to_string(index=False))
        if not len(dd):
            print("NO usable race sims -> feature would be NaN this weekend")
            continue
        race = ds[(ds.season == year) & (ds["round"] == rnd)][
            ["driver_code", "target_pos", "quali_pos", "pace_delta_5"]]
        m = dd.merge(race, left_on="driver", right_on="driver_code", how="inner")
        m["practice_rank"] = m["practice_longrun_delta_ms"].rank()
        m["expected_rank"] = m["pace_delta_5"].rank()          # pre-weekend expectation
        m["surprise_proxy"] = m["expected_rank"] - m["practice_rank"]  # + = faster than expected
        print("\nper-driver feature values vs actual race:")
        print(m[["driver", "practice_longrun_delta_ms", "practice_rank",
                 "expected_rank", "surprise_proxy", "quali_pos", "target_pos"]]
              .sort_values("practice_rank").round(1).to_string(index=False))
        rho = spearmanr(m["practice_rank"], m["target_pos"]).statistic
        print(f"\nSpearman(practice rank, race finish): {rho:.3f} (n={len(m)})")
        winner = m[m.target_pos == 1]
        if len(winner):
            w = winner.iloc[0]
            print(f"actual winner {w.driver}: practice rank {w.practice_rank:.0f}/{len(m)}, "
                  f"surprise proxy {w.surprise_proxy:+.1f}")


if __name__ == "__main__":
    main()
